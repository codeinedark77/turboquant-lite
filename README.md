# turboquant-lite

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Part of Agentic_](https://img.shields.io/badge/Part_of-Agentic__Super__OS-8A2BE2.svg)](#)

A standalone, pluggable implementation of TurboQuant (Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026) KV-cache compression, built and validated for consumer-VRAM local inference (developed against a 6GB RTX 3050) rather than the 24–32GB rigs the paper's and reference benchmarks used.

Point any OpenAI-client-compatible agent at this server's `base_url` and get a memory-cheaper local LLM backend, toggleable on/off via config. That's the whole point — it's meant to be attached and detached from other projects, not baked into any one of them.

**Read `ARCHITECTURE.md` for the full design rationale, the honest test-by-test findings, and exactly what's confirmed vs. still open.** This file is just how to run it.

## Status at a glance

56/56 tests pass, all on a synthetic (random-weight) model in a sandbox with no GPU — that setup validates *mechanism*, not language quality. Nothing here has run on real weights or a real GPU yet. Default config is Phase 2's plain MSE-K/V cache; Phase 2b's bias-corrected K path (`--use-prod`) exists and is tested but underperformed MSE in every full-model comparison run during development. This isn't just an under-tuned hyperparameter — a 2048x range of the relevant knob (`proj_bits`) produced no change, so it's a real open question, not a tuning gap. See `ARCHITECTURE.md` §18 for the investigation.

## Install

```bash
pip install torch transformers accelerate bitsandbytes fastapi uvicorn --break-system-packages
# or: pip install -e ".[serve]"
huggingface-cli login   # gated Llama weights need an accepted license + token
```

## Run the tests

```bash
pytest -q          # core algorithm + cache + attention hook + server, all synthetic-model
```

## Validate on your actual hardware (Phase 3 — not run anywhere yet)

```bash
python3 benchmarks/vram_budget.py --vram-gb 6                      # pre-flight estimate, no GPU needed
python3 benchmarks/phase3_runbook.py --model meta-llama/Llama-3.2-3B-Instruct
```

The runbook measures real VRAM (baseline vs. TurboQuant), perplexity delta on held-out text, and needle-in-haystack recall. Swap in your own text via `--text-file` — the built-in default is a short filler paragraph only so the script runs out of the box.

## Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# or: python3 server/app.py --model meta-llama/Llama-3.2-3B-Instruct --k-bits 3 --v-bits 4
```

Then point any OpenAI-compatible client at `http://localhost:8000/v1`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 64}'
```

Flags: `--no-turboquant` (serve baseline BF16 for comparison), `--use-prod` (Phase 2b's path, off by default), `--k-bits`/`--v-bits`.

## Layout

```
core/         rotation, Lloyd-Max codebook, TurboQuantMSE/Prod quantizers, bit-packing
validation/   theorem checks + pytest suite (44 tests, all passing)
inference/    HF Transformers Cache integration (kv_cache.py) + attention-forward
              patching for the Prod path (attention_hook.py) + a quality/divergence probe
benchmarks/   VRAM budget calculator (Phase 0) + the real-hardware Phase 3 runbook
server/       OpenAI-compatible FastAPI server — the actual pluggable deliverable
```

## What's real vs. what needs your GPU

Everything checkable without a GPU has been checked: the core algorithm's theoretical properties (empirically validated, not just derived), the Cache/AttentionInterface integrations against transformers' *current* API (fetched from source, not assumed), batch correctness, and the server's request/response plumbing including input validation. Triton kernels (Phase 4) were confirmed un-runnable in the dev sandbox — no GPU means no CPU fallback, so they weren't written rather than shipped untested.

What's still open, and needs real hardware to close: does Phase 2b's Prod path ever beat plain MSE on real weights, and what does Phase 3's actual perplexity/VRAM/latency data say about whether Phase 4 is worth building at all.
