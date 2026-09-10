"""
benchmarks/vram_budget.py — Phase 0: pre-flight VRAM headroom calculator.

Answers the question that decides whether the rest of this project is worth
building: at 4-bit weights, how much VRAM is actually left over for
KV-cache on a 6GB card, and how much does TurboQuant's K=3bit/V=4bit scheme
change the usable context length in that leftover space?

Architecture numbers for the two preset models are copied from their live
config.json on Hugging Face (checked directly, not from memory). Parameter
counts are then *computed* from those architecture fields rather than
hardcoded from a model card, as a built-in cross-check — see the docstring
on count_params for how closely that lines up with each model's commonly
reported size.

Use --custom for any other model: pull the same fields straight out of its
config.json into a small JSON file and point this at it.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass


@dataclass
class ModelArch:
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


# Verified against each model's config.json on Hugging Face directly (not
# from training-data memory) — see chat history for the fetched values.
PRESETS = {
    "qwen2.5-3b-instruct": ModelArch(
        name="Qwen2.5-3B-Instruct", hidden_size=2048, num_hidden_layers=36,
        num_attention_heads=16, num_key_value_heads=2, intermediate_size=11008,
        vocab_size=151936, head_dim=128, tie_word_embeddings=True,
    ),
    "llama-3.2-3b-instruct": ModelArch(
        name="Llama-3.2-3B-Instruct", hidden_size=3072, num_hidden_layers=28,
        num_attention_heads=24, num_key_value_heads=8, intermediate_size=8192,
        vocab_size=128256, head_dim=128, tie_word_embeddings=True,
    ),
}


def count_params(a: ModelArch) -> int:
    """Total parameter count computed from architecture fields (SwiGLU MLP,
    RMSNorm, tied or untied embeddings) rather than copied from a model
    card. Ignores attention-bias terms — a few thousand params against a
    multi-billion total, not worth the extra config fields. Computed values
    for both presets below land within ~0.1% of each model's commonly
    reported size (3.09B / 3.21B), which is the actual cross-check this
    function is for.
    """
    embed = a.vocab_size * a.hidden_size
    per_layer_attn = (
        a.hidden_size * (a.num_attention_heads * a.head_dim)    # q_proj
        + a.hidden_size * (a.num_key_value_heads * a.head_dim)  # k_proj
        + a.hidden_size * (a.num_key_value_heads * a.head_dim)  # v_proj
        + (a.num_attention_heads * a.head_dim) * a.hidden_size  # o_proj
    )
    per_layer_mlp = 3 * a.hidden_size * a.intermediate_size  # gate+up+down
    per_layer_norms = 2 * a.hidden_size
    per_layer = per_layer_attn + per_layer_mlp + per_layer_norms
    total = embed + a.num_hidden_layers * per_layer + a.hidden_size  # + final norm
    if not a.tie_word_embeddings:
        total += a.vocab_size * a.hidden_size  # separate lm_head
    return total


def weight_bytes(a: ModelArch, bits: int) -> float:
    return count_params(a) * bits / 8


def kv_bytes_per_token(a: ModelArch, k_bits: float, v_bits: float) -> float:
    """Raw bytes/token of KV-cache summed across all layers, at the given
    per-element bit-widths. Quantization metadata overhead is added
    separately in report() since it's per-vector, not per-element."""
    per_layer_k = a.num_key_value_heads * a.head_dim * (k_bits / 8)
    per_layer_v = a.num_key_value_heads * a.head_dim * (v_bits / 8)
    return a.num_hidden_layers * (per_layer_k + per_layer_v)


def max_context(vram_bytes: float, w_bytes: float, cuda_overhead: float, per_token_bytes: float) -> int:
    remaining = vram_bytes - w_bytes - cuda_overhead
    return int(remaining // per_token_bytes) if remaining > 0 else 0


def report(a: ModelArch, vram_gb: float, weight_bits: int, cuda_overhead_mb: float):
    vram_bytes = vram_gb * 1024**3
    cuda_overhead = cuda_overhead_mb * 1024**2
    params = count_params(a)
    w_bytes = weight_bytes(a, weight_bits)

    # Per-vector TurboQuant metadata: scale (fp16, 2B) on both K and V;
    # K (Prod variant) additionally carries a residual norm (fp16, 2B) and
    # proj_bits sign bits, packed -- default proj_bits = head_dim.
    k_overhead_per_vec = 2 + 2 + (a.head_dim / 8)
    v_overhead_per_vec = 2

    baseline_per_token = kv_bytes_per_token(a, k_bits=16, v_bits=16)
    tq_per_token = (
        kv_bytes_per_token(a, k_bits=3, v_bits=4)
        + a.num_hidden_layers * (k_overhead_per_vec + v_overhead_per_vec)
    )

    print(f"\n=== {a.name} | {vram_gb:.0f}GB VRAM budget ===")
    print(f"  computed params:        {params/1e9:.3f}B")
    print(f"  weights @ {weight_bits}-bit:        {w_bytes/1024**3:.2f} GB")
    print(f"  assumed CUDA/runtime overhead: {cuda_overhead_mb:.0f} MB")
    remaining = vram_bytes - w_bytes - cuda_overhead
    print(f"  headroom for KV-cache:   {remaining/1024**3:.2f} GB")
    if remaining <= 0:
        print("  ** NEGATIVE headroom at these settings — this model doesn't fit. **")
        return

    ctx_baseline = max_context(vram_bytes, w_bytes, cuda_overhead, baseline_per_token)
    ctx_tq = max_context(vram_bytes, w_bytes, cuda_overhead, tq_per_token)
    print(f"  baseline BF16 KV:  {baseline_per_token:6.1f} B/token -> ~{ctx_baseline:,} tokens max")
    print(f"  TurboQuant K3/V4:  {tq_per_token:6.1f} B/token -> ~{ctx_tq:,} tokens max")
    if ctx_baseline > 0:
        print(f"  -> {ctx_tq/ctx_baseline:.2f}x usable context at these settings")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(PRESETS.keys()) + ["all"], default="all")
    p.add_argument("--vram-gb", type=float, default=6.0)
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--cuda-overhead-mb", type=float, default=600,
                    help="Placeholder estimate for CUDA context + framework allocator "
                         "overhead — replace with a measured number in Phase 3.")
    p.add_argument("--custom", type=str, default=None,
                    help="Path to a JSON file with hidden_size, num_hidden_layers, "
                         "num_attention_heads, num_key_value_heads, intermediate_size, "
                         "vocab_size (+ optionally head_dim, tie_word_embeddings) — "
                         "copy these straight out of the model's config.json.")
    args = p.parse_args()

    if args.custom:
        with open(args.custom) as f:
            cfg = json.load(f)
        cfg.setdefault("name", "custom")
        models = [ModelArch(**cfg)]
    elif args.model == "all":
        models = list(PRESETS.values())
    else:
        models = [PRESETS[args.model]]

    for m in models:
        report(m, vram_gb=args.vram_gb, weight_bits=args.weight_bits, cuda_overhead_mb=args.cuda_overhead_mb)
