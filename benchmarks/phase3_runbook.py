"""
benchmarks/phase3_runbook.py — run this on the actual 3050, not in the dev
sandbox this project was built in. That sandbox measured out at 1 CPU core,
no GPU, and no network path to Hugging Face (gated Llama weights need auth
this environment doesn't have) -- so nothing below has been executed for
real. It's built directly on top of code that HAS been tested end-to-end
(inference/kv_cache.py, 35/35 passing) against a synthetic model; what's new
here is real weights and a real GPU, both first-run on your machine.

Setup:
    pip install torch transformers accelerate bitsandbytes --break-system-packages
    huggingface-cli login   # Llama-3.2-3B-Instruct is gated, needs accepted license + token

Usage:
    python3 benchmarks/phase3_runbook.py --model meta-llama/Llama-3.2-3B-Instruct
    python3 benchmarks/phase3_runbook.py --model meta-llama/Llama-3.2-3B-Instruct --text-file my_holdout.txt

Runs, in order:
    1. VRAM profile      -- actual torch.cuda.max_memory_allocated(), baseline vs TurboQuant,
                             replacing Phase 0's cuda_overhead_mb *estimate* with a measured number.
    2. Perplexity delta   -- teacher-forced loss on a held-out text sample (built-in default is a
                             short original paragraph, not copied from anywhere -- swap in your own
                             via --text-file for a meaningful result; the built-in default is only
                             here so the script runs out of the box).
    3. Needle-in-haystack -- a numeric needle inserted into synthetic filler text at increasing
                             depths, checking whether greedy generation recovers it.

Uses TurboQuantMSE for both K and V (Phase 2's integration) as the default --
NOT the Prod path from Phase 2b, which underperformed plain MSE in every
full-model test run during development (see ARCHITECTURE.md Phase 2b
status). Pass --k-bits/--v-bits/--use-prod to override and check whether
that holds on real weights too -- it's an open question, not a closed one.
"""
from __future__ import annotations
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

DEFAULT_HOLDOUT_TEXT = (
    "The old lighthouse keeper climbed the spiral stairs every evening at dusk, "
    "counting each of the two hundred and twelve steps out of habit rather than "
    "necessity. From the top, the sea looked less like water and more like a "
    "restless grey animal, breathing in slow tidal pulls against the rocks below. "
    "He had kept this rhythm for thirty-one years, long enough that his knees "
    "knew the stairs better than his mind did, and long enough that the town "
    "below had stopped asking when he might finally retire."
)


def load_model(model_name: str, device: str = "cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading {model_name} at 4-bit (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device
    )
    model.eval()
    return model, tokenizer


def build_cache(model, cfg, k_bits: int, v_bits: int, use_prod: bool):
    if use_prod:
        from inference.attention_hook import build_turboquant_prod_cache, register_turboquant_prod_attention
        cache, layers = build_turboquant_prod_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits, v_bits)
        impl = register_turboquant_prod_attention(layers, name="phase3_prod")
        model.set_attn_implementation(impl)
        return cache
    else:
        from inference.kv_cache import build_turboquant_cache
        model.set_attn_implementation("eager")
        return build_turboquant_cache(cfg.num_hidden_layers, cfg.head_dim, k_bits, v_bits)


def vram_profile(model, tokenizer, cfg, k_bits, v_bits, use_prod, context_lengths=(512, 2048, 8192)):
    print("\n=== 1. VRAM profile (measured, not estimated) ===")
    device = next(model.parameters()).device
    for ctx_len in context_lengths:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx_len), device=device)

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model.set_attn_implementation("eager")
            model(input_ids, use_cache=True)
        baseline_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            cache = build_cache(model, cfg, k_bits, v_bits, use_prod)
            model(input_ids, use_cache=True, past_key_values=cache)
        tq_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        print(f"  ctx={ctx_len:>6}: baseline peak {baseline_peak:.2f} GB | "
              f"TurboQuant K{k_bits}/V{v_bits} peak {tq_peak:.2f} GB | "
              f"saved {baseline_peak - tq_peak:.2f} GB")


def perplexity_delta(model, tokenizer, cfg, k_bits, v_bits, use_prod, text: str):
    print("\n=== 2. Perplexity delta (teacher-forced, held-out text) ===")
    device = next(model.parameters()).device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        model.set_attn_implementation("eager")
        baseline_loss = model(input_ids, labels=input_ids, use_cache=True).loss.item()

        cache = build_cache(model, cfg, k_bits, v_bits, use_prod)
        tq_loss = model(input_ids, labels=input_ids, use_cache=True, past_key_values=cache).loss.item()

    import math
    ppl_base, ppl_tq = math.exp(baseline_loss), math.exp(tq_loss)
    print(f"  baseline perplexity: {ppl_base:.3f}")
    print(f"  TurboQuant K{k_bits}/V{v_bits} perplexity: {ppl_tq:.3f}")
    print(f"  delta: {ppl_tq - ppl_base:+.3f} ({(ppl_tq/ppl_base - 1)*100:+.1f}%)")
    if len(text) < 2000:
        print("  NOTE: built-in default text is short and generic -- swap in a real held-out "
              "sample via --text-file for a result worth trusting.")


def needle_in_haystack(model, tokenizer, cfg, k_bits, v_bits, use_prod, depths=(0.1, 0.5, 0.9), haystack_tokens=4000):
    print("\n=== 3. Needle-in-haystack ===")
    device = next(model.parameters()).device
    filler = "The quick brown fox jumps over the lazy dog. "
    filler_ids = tokenizer(filler, return_tensors="pt").input_ids[0]
    n_repeats = haystack_tokens // filler_ids.shape[0] + 1
    haystack = filler_ids.repeat(n_repeats)[:haystack_tokens]

    for depth in depths:
        needle_num = torch.randint(10000, 99999, (1,)).item()
        needle_text = f" The secret number is {needle_num}. "
        needle_ids = tokenizer(needle_text, return_tensors="pt").input_ids[0]

        insert_at = int(len(haystack) * depth)
        full_ids = torch.cat([haystack[:insert_at], needle_ids, haystack[insert_at:]]).unsqueeze(0)
        prompt = torch.cat([
            full_ids[0],
            tokenizer("\n\nWhat is the secret number mentioned above? Answer with just the number:",
                       return_tensors="pt").input_ids[0],
        ]).unsqueeze(0).to(device)

        cache = build_cache(model, cfg, k_bits, v_bits, use_prod)
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=10, do_sample=False, past_key_values=cache)
        answer = tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)
        found = str(needle_num) in answer
        print(f"  depth={depth:.0%}: needle={needle_num}, model said {answer!r} -> {'FOUND' if found else 'MISSED'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--k-bits", type=int, default=3)
    p.add_argument("--v-bits", type=int, default=4)
    p.add_argument("--use-prod", action="store_true",
                    help="Use Phase 2b's bias-corrected K path instead of Phase 2's plain MSE-K default. "
                         "Underperformed MSE-K in every synthetic-model test this session -- "
                         "worth checking whether that holds on real weights, not assumed.")
    p.add_argument("--text-file", type=str, default=None, help="Path to held-out text for perplexity eval.")
    p.add_argument("--skip-vram", action="store_true")
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--skip-needle", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU detected. This script needs the actual 3050 -- "
              "it will not give meaningful results on CPU.")
        sys.exit(1)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    model, tokenizer = load_model(args.model)

    text = DEFAULT_HOLDOUT_TEXT
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()

    t0 = time.time()
    if not args.skip_vram:
        vram_profile(model, tokenizer, cfg, args.k_bits, args.v_bits, args.use_prod)
    if not args.skip_perplexity:
        perplexity_delta(model, tokenizer, cfg, args.k_bits, args.v_bits, args.use_prod, text)
    if not args.skip_needle:
        needle_in_haystack(model, tokenizer, cfg, args.k_bits, args.v_bits, args.use_prod)
    print(f"\nDone in {time.time()-t0:.1f}s")
