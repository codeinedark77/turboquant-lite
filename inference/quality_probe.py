"""
inference/quality_probe.py — Phase 2 quantitative probe.

Honest scope: this sandbox has no GPU and no network path to Hugging Face
(gated Llama weights need auth this environment doesn't have either), so
there is no trained model available here. A random-weight model can't tell
you anything about *semantic* quality loss — perplexity on a model that
hasn't learned language is meaningless. What it CAN honestly tell you:
whether the quantization mechanism behaves the way it's supposed to --
does more bits mean less divergence, does divergence stay bounded as
context grows, is the effect size in a sane numeric range (not zero, not
exploding/NaN). That's what this measures. The real quality gate
(perplexity delta, needle-in-haystack) from ARCHITECTURE.md Phase 2 still
needs to run on real hardware with real weights -- this de-risks the
mechanism first.

Method: single prefill forward pass over `seq_len` random tokens, comparing
final-position next-token distributions between an unquantized baseline
cache and a TurboQuant-quantized cache built from the SAME model weights.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from inference.kv_cache import build_turboquant_cache


def build_synthetic_model(seed: int = 0) -> tuple[LlamaForCausalLM, LlamaConfig]:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        hidden_size=64, num_attention_heads=8, num_key_value_heads=2,
        head_dim=8, num_hidden_layers=2, intermediate_size=128,
        vocab_size=100, max_position_embeddings=1024,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model, cfg


def divergence_at(model, cfg, seq_len: int, k_bits: int, v_bits: int, seed: int) -> dict:
    torch.manual_seed(seed)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with torch.no_grad():
        out_baseline = model(input_ids, use_cache=True)
        tq_cache = build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits, v_bits)
        out_tq = model(input_ids, use_cache=True, past_key_values=tq_cache)

    lb = out_baseline.logits[0, -1]
    lt = out_tq.logits[0, -1]
    log_pb = F.log_softmax(lb, dim=-1)
    log_pt = F.log_softmax(lt, dim=-1)
    kl = F.kl_div(log_pt, log_pb, log_target=True, reduction="sum").item()
    return {
        "seq_len": seq_len, "k_bits": k_bits, "v_bits": v_bits, "seed": seed,
        "kl_div": kl,
        "max_abs_logit_diff": (lb - lt).abs().max().item(),
        "argmax_match": bool((lb.argmax() == lt.argmax()).item()),
    }


def run_sweep(n_seeds: int = 3) -> list[dict]:
    model, cfg = build_synthetic_model(seed=42)
    results = []
    for seq_len in (10, 50, 200):
        for k_bits, v_bits in ((8, 8), (4, 4), (3, 4), (2, 3)):
            for seed in range(n_seeds):
                results.append(divergence_at(model, cfg, seq_len, k_bits, v_bits, seed))
    return results


def summarize(results: list[dict]):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["seq_len"], r["k_bits"], r["v_bits"])].append(r)

    print(f"{'seq_len':>8} {'k/v bits':>10} {'mean KL':>12} {'mean max|Δlogit|':>18} {'argmax match rate':>18}")
    for (seq_len, kb, vb), rs in sorted(grouped.items()):
        mean_kl = sum(r["kl_div"] for r in rs) / len(rs)
        mean_maxdiff = sum(r["max_abs_logit_diff"] for r in rs) / len(rs)
        match_rate = sum(r["argmax_match"] for r in rs) / len(rs)
        print(f"{seq_len:>8} {f'{kb}/{vb}':>10} {mean_kl:>12.5f} {mean_maxdiff:>18.4f} {match_rate:>18.0%}")


if __name__ == "__main__":
    results = run_sweep()
    summarize(results)
