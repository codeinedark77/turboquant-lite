"""
validation/test_attention_hook.py — pytest suite for inference/attention_hook.py.
Deliberately does NOT assert Prod beats plain MSE end-to-end -- that's not
what the real measurements showed (see ARCHITECTURE.md Phase 2b status).
These tests check the mechanism is wired correctly and behaves sanely, which
is what's actually confirmed.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from inference.quality_probe import build_synthetic_model
from inference.attention_hook import (
    build_turboquant_prod_cache,
    register_turboquant_prod_attention,
    repeat_kv_tensor,
)


@pytest.fixture(scope="module")
def tiny_model():
    return build_synthetic_model(seed=0)


def test_repeat_kv_tensor_shape_and_values():
    x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    out = repeat_kv_tensor(x, n_rep=2)
    assert out.shape == (2, 6, 4, 5)
    assert torch.equal(out[:, 0], out[:, 1])  # each kv head repeated adjacently
    assert torch.equal(out[:, 2], out[:, 3])


def test_repeat_kv_tensor_noop_at_n_rep_1():
    x = torch.randn(2, 3, 4, 5)
    assert torch.equal(repeat_kv_tensor(x, 1), x)


def test_prod_attention_generate_completes(tiny_model):
    model, cfg = tiny_model
    tq_cache, layers = build_turboquant_prod_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
    impl_name = register_turboquant_prod_attention(layers, name="test_prod_generate")
    model.set_attn_implementation(impl_name)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=4, do_sample=False, past_key_values=tq_cache)
    model.set_attn_implementation("eager")
    assert out.shape == (1, 9)


def test_prod_attention_output_not_nan_or_exploding(tiny_model):
    model, cfg = tiny_model
    tq_cache, layers = build_turboquant_prod_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits=3, v_bits=4)
    impl_name = register_turboquant_prod_attention(layers, name="test_prod_sanity")
    model.set_attn_implementation(impl_name)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        out = model(input_ids, use_cache=True, past_key_values=tq_cache)
    model.set_attn_implementation("eager")
    assert torch.isfinite(out.logits).all()
    assert out.logits.abs().max() < 1e4  # sane range, not blown up
