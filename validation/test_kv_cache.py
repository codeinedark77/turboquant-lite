"""
validation/test_kv_cache.py — pytest suite for inference/kv_cache.py.
Uses a tiny synthetic (random-weight) Llama model -- see quality_probe.py's
docstring for why that's the honest ceiling on what's testable in this
sandbox (no GPU, no network path to gated real weights).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from transformers import LlamaConfig, LlamaForCausalLM
from inference.kv_cache import build_turboquant_cache, TurboQuantCacheLayer
from inference.quality_probe import build_synthetic_model, divergence_at


@pytest.fixture(scope="module")
def tiny_model():
    return build_synthetic_model(seed=0)


def test_generate_completes_and_matches_shape(tiny_model):
    model, cfg = tiny_model
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    tq_cache = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=6, do_sample=False, past_key_values=tq_cache)
    assert out.shape == (1, 11)  # 5 prompt + 6 new
    assert out.dtype == torch.long


def test_generate_is_deterministic(tiny_model):
    model, cfg = tiny_model
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        cache_a = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
        out_a = model.generate(input_ids, max_new_tokens=8, do_sample=False, past_key_values=cache_a)
        cache_b = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
        out_b = model.generate(input_ids, max_new_tokens=8, do_sample=False, past_key_values=cache_b)
    assert torch.equal(out_a, out_b)


def test_high_bits_closely_tracks_baseline(tiny_model):
    """At 8/8-bit (near-lossless), divergence from the unquantized baseline
    should be small -- catches a fundamentally broken pipeline, not just a
    lossy one."""
    model, cfg = tiny_model
    r = divergence_at(model, cfg, seq_len=30, k_bits=8, v_bits=8, seed=1)
    assert r["kl_div"] < 0.01
    assert r["argmax_match"]


def test_divergence_monotonic_in_bits(tiny_model):
    """More bits should never produce *more* divergence from baseline, on
    average across seeds -- this is the structural check that the
    quantization knob is actually wired correctly, not just present."""
    model, cfg = tiny_model
    bit_settings = [(2, 3), (3, 4), (4, 4), (8, 8)]
    mean_kls = []
    for k_bits, v_bits in bit_settings:
        kls = [divergence_at(model, cfg, seq_len=40, k_bits=k_bits, v_bits=v_bits, seed=s)["kl_div"]
               for s in range(4)]
        mean_kls.append(sum(kls) / len(kls))
    # non-increasing within a small tolerance for Monte Carlo noise at n=4 seeds
    for i in range(len(mean_kls) - 1):
        assert mean_kls[i + 1] <= mean_kls[i] + 1e-4, (bit_settings, mean_kls)


def test_batched_generation_matches_solo_generation(tiny_model):
    """Every other test in this suite uses batch_size=1. This checks the
    thing that actually matters for concurrent serving (Phase 0 flagged
    this as untested): generating N sequences together as one batch must
    produce identical results to generating each one alone -- any
    cross-batch-item leakage in the cache would show up as a mismatch here.

    Comparison is truncated to each sequence's own (solo) length, not a
    full-length exact match: a sequence that hits EOS early gets
    pad-continued in the batched run to match the batch's longest member --
    expected HF batching behavior, unrelated to cache correctness. An
    earlier version of this test used full-length comparison and passed
    only by luck of the seed (1) it happened to run at; a follow-up sweep
    across 10 seeds found seed=5 failing under that stricter comparison,
    traced to exactly this EOS-padding effect, not a real bug. Verified
    robust across all 10 seeds under the corrected (truncated) comparison."""
    model, cfg = tiny_model
    model.set_attn_implementation("eager")
    torch.manual_seed(1)
    seqs = [torch.randint(0, cfg.vocab_size, (1, 8)) for _ in range(3)]

    solo_outputs = []
    for s in seqs:
        with torch.no_grad():
            cache = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
            solo_outputs.append(model.generate(s, max_new_tokens=5, do_sample=False, past_key_values=cache))

    batch_input = torch.cat(seqs, dim=0)
    with torch.no_grad():
        cache = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
        batch_out = model.generate(batch_input, max_new_tokens=5, do_sample=False, past_key_values=cache)

    for i in range(3):
        solo_len = solo_outputs[i].shape[1]
        assert torch.equal(solo_outputs[i][0], batch_out[i][:solo_len]), \
            f"sequence {i} diverged between solo and batched generation (up to its natural stop length)"


def test_variable_length_batch_with_attention_mask_matches_solo(tiny_model):
    """Closes a gap flagged earlier as untested: the same-length batch test
    doesn't exercise padding/attention_mask at all. Left-pads 3
    different-length sequences into one batch, generates with an explicit
    attention_mask, and checks each sequence's new tokens match generating
    it alone (unpadded) -- real evidence padding doesn't leak into the
    quantized cache's per-vector scale/quantization computation."""
    model, cfg = tiny_model
    model.set_attn_implementation("eager")
    torch.manual_seed(3)
    seqs = [torch.randint(0, cfg.vocab_size, (7,)), torch.randint(0, cfg.vocab_size, (12,)), torch.randint(0, cfg.vocab_size, (4,))]
    pad_id = 0
    maxlen = max(len(s) for s in seqs)
    padded = torch.stack([torch.cat([torch.full((maxlen - len(s),), pad_id), s]) for s in seqs])
    mask = torch.stack([torch.cat([torch.zeros(maxlen - len(s)), torch.ones(len(s))]) for s in seqs]).long()

    solo = []
    for s in seqs:
        with torch.no_grad():
            cache = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
            out = model.generate(s.unsqueeze(0), max_new_tokens=4, do_sample=False, past_key_values=cache)
        solo.append(out[0, len(s):])

    with torch.no_grad():
        cache = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
        batch_out = model.generate(padded, attention_mask=mask, max_new_tokens=4, do_sample=False, past_key_values=cache)

    for i in range(3):
        solo_len = solo[i].shape[0]
        assert torch.equal(solo[i], batch_out[i, maxlen:maxlen + solo_len]), \
            f"sequence {i} (len {len(seqs[i])}) diverged (up to its natural stop length)"


def test_cache_seq_length_tracks_updates():
    layer = TurboQuantCacheLayer(head_dim=8, k_bits=4, v_bits=4)
    k1 = torch.randn(1, 2, 5, 8)
    v1 = torch.randn(1, 2, 5, 8)
    layer.update(k1, v1)
    assert layer.get_seq_length() == 5
    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)
    layer.update(k2, v2)
    assert layer.get_seq_length() == 6


def test_cache_reset_clears_state():
    layer = TurboQuantCacheLayer(head_dim=8, k_bits=4, v_bits=4)
    layer.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
    assert layer.get_seq_length() == 5
    layer.reset()
    assert layer.get_seq_length() == 0
