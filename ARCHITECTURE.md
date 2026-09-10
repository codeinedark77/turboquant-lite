# TurboQuant-Lite — Architecture Plan

**Status:** Pre-implementation design doc
**Target hardware:** RTX 3050, 6GB VRAM / 16GB DDR5 (Garuda Linux)
**Reference:** Zandieh, Daliri, Hadian, Mirrokni — *TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate* (arXiv:2504.19874, ICLR 2026)

## TL;DR

A standalone, pluggable KV-cache quantization module implementing TurboQuant, built and validated on a 6GB card instead of the 24–32GB rigs the paper and reference implementations used. Ships as an OpenAI-compatible FastAPI server so any agent in the Empire (or any other project) can point `base_url` at it and get a memory-cheaper local LLM backend. Correctness and real measured numbers come before speed; Triton optimization is a conditional phase, not a default one.

---

## 1. Scope

**In scope (v1):**
- Dense, GQA-attention decoder models only (Qwen2.5-3B-Instruct / Llama-3.2-3B-Instruct class).
- Single-GPU, single-model serving.
- Config-toggleable TurboQuant KV-cache, independently bit-width-configurable for K and V.
- OpenAI-compatible `/v1/chat/completions` server as the integration surface.

**Explicitly out of scope (v1):**
- MoE / linear-attention / Mamba-hybrid layers — TurboQuant's own validation (vLLM/Red Hat study) only covers standard attention; unsupported layers get skipped, not force-quantized.
- vLLM production serving — deferred, see §6.
- Multi-GPU / tensor parallelism — irrelevant at this VRAM budget.

## 2. Design Principles

1. **Additive-only.** TurboQuant wraps the KV-cache; it never forks model internals. Disabled via config, the model must reproduce baseline BF16 output byte-for-byte.
2. **Measure, don't borrow.** Every number in this project gets a benchmark script next to it, run on *your* 3050. The paper's numbers and the reference repo's numbers were measured on 24–32GB cards — they set expectations, not results.
3. **Correctness before speed.** Theoretical validation → functional validation on a real model → resource validation on real hardware → optimize only what Phase 3 proves is worth optimizing.
4. **No vanity numbers.** Every headline claim ("Nx compression," "context extended to N") gets an adversarial-audit line: what's actually being counted, was it quality-checked or just non-crashing, is it a real multi-run average or N=1.

## 3. Algorithm Recap

Each K/V vector entering the cache goes through:

1. **PolarQuant rotation** — a random orthogonal rotation spreads the vector's energy evenly across dimensions, pushing per-coordinate values toward a concentrated Beta distribution that's easy to quantize near-optimally.
2. **Lloyd-Max scalar quantization** — an optimal b-bit scalar quantizer, precomputed offline for that specific Beta distribution, applied per coordinate post-rotation.
3. **QJL residual correction** (*Prod variant, Keys only*) — a 1-bit Quantized Johnson-Lindenstrauss projection on the leftover quantization error, removing the systematic bias that MSE-optimal quantizers otherwise introduce into inner-product estimates.

**Why keys and values are treated differently:** attention scores are `Q·Kᵀ` — an inner product — so a biased quantizer on K directly distorts softmax weighting. That's what step 3 fixes, and it's why keys need the *Prod* variant. Values are combined via a weighted sum, not used as an inner-product operand, so plain MSE quantization is sufficient — just at a higher bit-width, since empirical data (reference implementation) shows V is the accuracy bottleneck: 2-bit V lands at cos_sim 0.94, 4-bit V at 0.997. **Default: K=3bit, V=4bit, both overridable per experiment.**

**Open implementation decision:** rotation can be a dense random orthogonal matrix (O(d²) per apply — simple, fine at typical head_dim 64–128) or a structured/Hadamard-based fast rotation (O(d log d) — more code, matters more at large head_dim). Start dense. Only build the fast path if Phase 3 profiling shows rotation is actually the bottleneck.

## 4. Module Layout

```
turboquant-lite/
├── core/
│   ├── rotation.py        # random orthogonal rotation + QJL projection matrices
│   ├── codebook.py        # Lloyd-Max scalar quantizer for Beta-distributed coords
│   ├── quantizer.py       # TurboQuantMSE + TurboQuantProd — quantize/dequantize
│   └── bitpack.py         # sub-byte packing/unpacking (2/3/4-bit groups)
├── validation/
│   ├── theorems.py        # MSE distortion bound, unbiasedness, 1/4^b scaling — CPU only
│   └── test_quantizer.py  # pytest unit tests for core/
├── inference/
│   ├── kv_cache.py        # quantize-on-write / dequantize-on-read cache manager
│   ├── attention_hook.py  # wraps HF Transformers attention forward pass
│   └── model_loader.py    # loads target model at 4-bit weights
├── kernels/
│   └── triton_kernels.py  # Phase 4 stretch — fused quantize/attention kernels
├── server/
│   ├── app.py              # FastAPI, OpenAI-compatible /v1/chat/completions
│   └── config.py           # bit-width + on/off config, per K/V
├── benchmarks/
│   ├── vram_budget.py      # Phase 0 — pre-flight headroom calculator
│   ├── vram_profile.py     # Phase 3 — actual measured VRAM, baseline vs TQ
│   └── quality_eval.py     # perplexity / cos-sim / needle-recall checks
├── pyproject.toml
└── ARCHITECTURE.md         # this doc
```

## 5. Phased Roadmap

### Phase 0 — Budget & Scaffold
`benchmarks/vram_budget.py` reads the target model's `config.json`, computes weight memory at 4-bit plus per-token KV-cache bytes from `num_layers × num_kv_heads × head_dim`, and prints exactly how much VRAM headroom remains on a 6GB card at a given target context length. **This number decides whether the rest of the project is worth it — not the paper's numbers.** Ballpark to sanity-check against: a 3B model at 4-bit is roughly 1.5–2GB of weights, leaving real headroom; a 7B at 4-bit is closer to 3.5–4GB, leaving very little. Repo scaffold + pytest wired up alongside this.

### Phase 1 — Core Algorithm (pure PyTorch, CPU-only, no GPU needed)
Build `core/rotation.py`, `codebook.py`, `quantizer.py`, `bitpack.py`. Gate: `validation/theorems.py` must pass all three checks — MSE distortion bound, unbiasedness of the Prod estimator, and the 1/4^b distortion-vs-bitwidth scaling law — before any GPU work starts.

### Phase 2 — Functional Integration
`inference/attention_hook.py` wraps a small HF Transformers model's attention forward pass, swapping the live KV-cache for the quantized store via `inference/kv_cache.py`. Correctness gate: perplexity delta vs. BF16 baseline on a held-out sample, plus needle-in-haystack retrieval at increasing context lengths — run on the actual 3050, actual chosen model.

### Phase 3 — Resource Validation (the real answer)
`vram_profile.py` measures actual VRAM before/after and max context before OOM. `quality_eval.py` reports cos-sim/perplexity/needle-recall per bit-width config. Output is a results table with real numbers — this is the checkpoint that decides whether Phase 4 happens at all.

### Phase 4 — Triton Kernels (conditional)
Only pursued if Phase 3 shows dequantize-on-read is a meaningful latency tax for your actual usage pattern. Fused quantize + dequantize + attention-score kernels in `kernels/triton_kernels.py`.

### Phase 5 — Pluggable Server
`server/app.py`: FastAPI, OpenAI-compatible, streaming, config flag for TurboQuant on/off and per-K/V bit-width. This is the artifact that plugs into the Empire — swap `base_url`, nothing else changes.

## 6. Why HF Transformers, Not vLLM (v1)

vLLM's PagedAttention allocator is tuned against 24GB+ deployments — the exact regime both the paper's and the reference implementation's benchmarks ran in. On 6GB it's fighting you for every megabyte while you're also patching its internals, which is the wrong place to debug a new quantization scheme. HF Transformers gives a direct, inspectable attention forward pass, which matters more than production throughput at this stage. vLLM integration is a legitimate later phase once the algorithm is proven correct and raw serving throughput actually becomes the bottleneck.

## 7. Known Unknowns / Risks

- **Model architecture check required before Phase 2.** Confirm the chosen model uses plain GQA with no sliding-window or hybrid-attention layers — TurboQuant doesn't cover those yet.
- **6GB may just be tight, and that's a valid result.** Phase 0 might reveal there isn't enough headroom above a 3B model's weights to make KV-cache compression visibly matter. If so, the correct move is dropping to a 1.5B model or narrowing the context range being tested — not forcing the numbers.
- **Speed may regress before it improves.** Dequantize-on-read is real overhead; the win here is capacity/context-length, not tokens/sec, until and unless Phase 4 lands.
- **Codebooks are distribution-specific.** Lloyd-Max codebooks are tied to the Beta distribution induced by rotation at a given `(head_dim, bit-width)` pair — generate once per combination and cache them, don't regenerate per request.

## 8. Licensing Note

The reference implementation (0xSero/turboquant) is GPL-3.0 — fine to read and learn from. Reimplementing from the paper's algorithm description (rather than copying its code) keeps this module license-clean for reuse across other projects, which matters since the whole point is a pluggable component.

## 9. Phase 2 Status (updated post-implementation)

Built and passing (29/29 tests total): `inference/kv_cache.py` — a real `transformers.cache_utils.CacheLayerMixin` subclass (`TurboQuantCacheLayer`), verified against the *current* Cache API (fetched from source — it moved to a per-layer `CacheLayerMixin` design at some point after this model's training cutoff, so this was checked live rather than assumed). Drops into `model.generate(past_key_values=...)` with zero changes to the model's forward pass.

**Scope correction, found while integrating, not anticipated in §3:** the standard `CacheLayerMixin.update()` contract returns plain tensors that flow into an unmodified attention matmul. TurboQuantProd's QJL bias-correction needs the query available at attention-score time — realizing it means intercepting `Q @ Kᵀ` itself, not just the cache. So Phase 2 as built uses **TurboQuantMSE for both K and V** (dequantize-on-read); the Prod path validated in Phase 1 is real and tested, but wiring it into live attention is separate, harder work (model-specific attention-forward patching) — pushed to a Phase 2b rather than silently dropped or overclaimed.

**What's actually testable here vs. what needs your hardware:** no GPU and no network path to gated Llama weights in this sandbox, so validation runs on a tiny synthetic (random-weight, real-architecture) Llama model. That can't measure semantic quality — perplexity on a model that hasn't learned language is meaningless — but it can and does check that the mechanism behaves correctly: a divergence probe (`inference/quality_probe.py`) comparing quantized-cache vs. baseline next-token distributions confirmed KL divergence is (a) strictly monotonic in bit-width across every context length tested, and (b) at the default K3/V4 setting, argmax matched baseline in 100% of trials at 10/50/200-token contexts, dropping to 67% only at the more aggressive K2/V3.

One finding worth flagging rather than trusting at face value: divergence *shrank* with longer context (10→200 tokens) instead of compounding, plausibly because softmax-weighted averaging over more positions partially cancels independent per-token quantization noise. That's measured on a random model with flat, unpeaked attention. Real trained attention is often much more peaked (a few positions dominate), which would weaken this averaging effect — so "error doesn't compound" should be treated as untested-on-real-weights, not confirmed, until Phase 3 runs on the actual model.

## 10. Phase 2b Status — realizing the Prod path (mixed, honestly reported)

Built: `inference/attention_hook.py`, registering a custom score function via `transformers.AttentionInterface` (the real, documented extension point — `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`, verified from current docs/source rather than assumed). `TurboQuantProdCacheLayer` keeps K in Prod form and exposes the raw packet directly to the registered function via closure, since `Cache.update()`'s tensor-only return contract can't carry the residual/sign bits Prod needs.

**A real design bug caught before it shipped:** the Phase 1 estimator (`estimate_inner_product`) only computed *aligned* dot products (query[i] against key[i]) — correct for its own Monte Carlo test, wrong for real attention, which scores every query position against every key position. Generalizing to the full `[S_q, S_k]` matrix (`estimate_attention_scores`, matmul-based) required re-deriving the shape of the computation, and it was re-validated for unbiasedness independently (0.4% median relative bias, tighter than the aligned version, averaged over 200 trials) rather than assumed correct because the aligned version was.

**The honest, currently-unresolved finding:** unbiased-in-expectation-over-200-trials is not the same claim as low-error-in-the-single-draw a real deployment actually gets (one rotation, one projection matrix, drawn once and reused for the cache's lifetime). Measured head-to-head against Phase 2's plain MSE-K, at matched K=3-bit, averaged over 6 seeds:

| context | MSE-K mean KL vs. baseline | Prod-K mean KL vs. baseline |
|---|---|---|
| 20 tok | 0.00001 | 0.00038 |
| 100 tok | 0.00001 | 0.00030 |

Prod-K is *worse* in this test, not better. Isolated at the estimator level (no model in the loop), single-draw error does shrink with `proj_bits` as the sqrt(1/m) variance-reduction predicts (0.477 → 0.070 mean abs error, 8→512 proj_bits) — so the mechanism itself is behaving correctly. But re-running the *full-model* comparison at proj_bits from 32 up to 2048 (a 64x range) produced an unchanged 0.00055 mean KL — no improvement at all, which contradicts the isolated result and is not yet explained. Logged here as an open discrepancy rather than papered over with an untested explanation.

**Practical recommendation given what's actually confirmed:** ship Phase 2's plain MSE-K as the default. It's simpler, and it won a real head-to-head against Prod-K in every full-model test run this session. Prod's bias-correction is proven real and unbiased at the estimator level, but doesn't yet demonstrate a practical win end-to-end — treat it as a documented, real, unresolved research gap (same spirit as the Empire project's known-gaps list), not a finished feature.

## 11. Phase 5 Status — the pluggable server (built, this is the original ask)

Built and tested (39/39 total): `server/app.py`, an OpenAI-compatible `/v1/chat/completions` server (streaming + non-streaming) plus `/v1/models`, defaulting to Phase 2's plain MSE-K/V cache (the config that actually won every head-to-head this session — see §10). `use_prod` and `use_turboquant` are both plain config flags, not code changes, so the honest, unresolved Prod-vs-MSE question stays checkable rather than getting silently decided by whatever shipped first.

Tested via FastAPI's `TestClient` against the same synthetic model used throughout, with a minimal mock tokenizer standing in for real vocabulary (the synthetic model has none). This validates the actual deliverable's plumbing — routing, OpenAI schema shape, SSE streaming format, the on/off toggle actually switching which attention implementation runs, not just accepting the flag and ignoring it. It does not validate generation quality, same limitation as every other phase in this sandbox.

Real use (needs real weights + your GPU):
```
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
then point any OpenAI-client-compatible agent in the Empire at `http://localhost:8000/v1` — this is the "plug in whenever I want" piece from the original ask.

## 12. Honest Overall Status

| Phase | Status | Confidence |
|---|---|---|
| 0 — VRAM budget | Done | High — real configs, computed params match published sizes to ~0.1% |
| 1 — Core algorithm | Done, 100% tested | High — theorems validated empirically, not just derived |
| 2 — Cache integration (MSE) | Done, tested on synthetic model | High on mechanism; quality needs real weights |
| 2b — Attention patching (Prod) | Done, tested, underperforms MSE | Mechanism confirmed correct in isolation; full-model result real but unexplained |
| 3 — Real-hardware validation | Runbook written, not run | Untested — needs your 3050 |
| 4 — Triton kernels | Not started | Correctly gated behind Phase 3 data that doesn't exist yet |
| 5 — Pluggable server | Done, tested on synthetic model | High on plumbing; quality needs real weights |

## 13. Final Hardening Pass

**Phase 4 confirmed blocked, not just assumed.** Actually tried running a trivial Triton kernel in this sandbox rather than inferring from "no GPU" alone: fails with `RuntimeError: 0 active drivers`. Triton has no CPU fallback — it JIT-compiles to a real GPU backend or it doesn't run at all. Writing Triton kernels here would produce zero-percent-tested code, which is a worse artifact than not writing them; Phase 4 stays correctly untouched until Phase 3 runs on real hardware.

**Batch_size>1, untested anywhere else this session, checked for real:** every prior test used batch_size=1. Generating 3 sequences together as one batch produces token-for-token identical output to generating each alone, across 5 autoregressive decode steps each (not just a single prefill) — real evidence against cross-batch-item leakage in the cache, which is exactly the kind of bug that's easy to introduce and easy to miss. Caveat worth keeping honest: this used same-length sequences with no padding; variable-length batches need `attention_mask` threaded through properly, untested here, relevant if server-side request batching gets built later (Phase 6 territory, not attempted).

**Caught and fixed a real bug in my own previous edit**, not just in the "new" code: a `str_replace` had silently merged `test_cache_seq_length_tracks_updates`'s body into a different test, deleting its `def` line — passing test count stayed misleadingly unchanged (39→39) while test *identity* was wrong. Found by checking collected test names against what should exist, not by trusting the headline pass/fail count alone. Fixed; both tests now exist correctly. 40/40 pass.

## 14. Honest Overall Status (superseded by §18 — kept for the record, not re-edited to stay accurate)

| Phase | Status | Confidence |
|---|---|---|
| 0 — VRAM budget | Done | High — real configs, computed params match published sizes to ~0.1% |
| 1 — Core algorithm | Done, 100% tested | High — theorems validated empirically |
| 2 — Cache integration (MSE) | Done, tested on synthetic model | High on mechanism; quality needs real weights |
| 2b — Attention patching (Prod) | Done, underperforms MSE | Mechanism correct in isolation; full-model result real but unexplained *(see §18 — investigated further, not still a mystery)* |
| 3 — Real-hardware validation | Runbook written, not run | Untested — needs your 3050 |
| 4 — Triton kernels | Not started, confirmed blocked here | Correctly gated — verified un-runnable in this sandbox, not just deferred |
| 5 — Pluggable server | Done, tested on synthetic model | High on plumbing; quality needs real weights |
| — Batch correctness | Verified same-length batches | *(see §18 — variable-length + attention_mask now verified too)* |

Two things need your hardware to close: whether Prod ever beats MSE on real weights, and whether Phase 4 is worth building once Phase 3 has real latency numbers to gate it on. Everything gate-able without a GPU has been gated and checked.

## 15. Repo Hygiene Pass

Tested the server's actual failure modes rather than trusting pydantic's type checking alone: empty `messages` was a raw 500 with a stack trace, not a clean error. Added validators (empty messages, non-positive or excessive `max_tokens`) — all now return structured 422s; a control valid-request case confirmed normal operation still works. Added `README.md` as a real getting-started entry point separate from this file's design-doc depth. 44/44 tests pass.

## 16. Empire Integration — `empire_integration/local_llm_backend.py`

Honest scope: designed from memory of the Empire's architecture (LangGraph harness, A2A bus, Safety Guardian gate, unified capability registry, 19 agents), not tested against the real harness since it isn't available in this sandbox. What's tested is the thing underneath it — a real `langchain_openai.ChatOpenAI` client talking to a real running turboquant-lite server (wired via `httpx.ASGITransport`, no actual network needed) — and that testing caught a real, non-obvious bug:

**Real bug: `ChatOpenAI` sends `max_completion_tokens`, not `max_tokens`.** The server only recognized the older field name; pydantic silently drops unrecognized fields by default, so every LangChain-driven request would have ignored the caller's token limit entirely and generated up to the 256-token default regardless of what was asked for — a real, silent, costly bug that only a real client library surfaces (every one of my own hand-rolled tests all session used `max_tokens` because that's what I expected, so none of them could have caught this). Fixed: both field names now accepted, `max_completion_tokens` wins if both are present, matching OpenAI's own migration precedent. Regression-tested at both the raw-payload level and with the actual `ChatOpenAI` client that found it.

**Adapter design:** `get_llm_with_fallback(local_cfg, groq_llm)` — one fallback path, not a per-call-site try/except pattern repeated across agents. That's a deliberate callback to the Empire's own documented bug history: the hardcoded `== 19` agent count spread across four files because the same check got copy-pasted instead of centralized. A health check (`is_local_backend_healthy`) actually hits `/v1/models` rather than assuming the server is up.

**Flagged, not decided:** where backend selection should actually live (the unified capability registry seems right, so it stays a single checked source rather than a sixth drifted list); whether Safety Guardian needs to care about backend identity at all (my read from memory is it gates by agent behavior — github/content/browser/computer-use — not by which model reasons for it, but that's inference, not a read of the real gate code); whether A2A bus messages carry any latency/format assumption tuned to Groq's 70B that a local 3B model's different response profile might violate. All three need your actual code to resolve, not more reasoning from memory here.

**Not a quality-equivalent swap.** This serves whatever's loaded locally (3B-class, per this project's testing) against Groq's 70B elsewhere in the stack. Which agents, if any, are reasonable candidates for the tradeoff is a call for your capability registry, not this module.

## 17. Test Count at Empire-Integration Handoff (superseded by §18)

51/51 pass — core algorithm, both cache integrations, batch correctness, server plumbing and validation, and the Empire integration adapter's testable surface (health check + fallback logic).
## 18. Gap-Closing Pass

Explicit request to verify everything and close what's closeable without a GPU. Results:

**Bug audit, systematic:** parsed every file in the repo with `ast`, flagged any function over 30 lines for manual review (the two orphaned-body bugs found earlier were both symptoms of a specific `str_replace` failure mode — using a bare `def name():` line as the match target, which can silently delete that line while splicing another function's content in). Four functions flagged; all four manually confirmed coherent, no further instances found. The bug class appears fully swept, not just patched where it happened to be noticed.

**The proj_bits discrepancy — genuinely investigated, not re-guessed.** Previously logged as "unexplained." Ruled out a registration bug first (confirmed `model._attn_implementation` actually switches per call). Then measured the correction term directly inside a real forward pass: its magnitude is real and non-negligible (~16-18% of the direct term), and does shrink slightly with proj_bits — so the mechanism isn't inert. Tested and *rejected* a specific hypothesis (that the correction is roughly constant across key positions, which softmax would cancel regardless of precision) — the correction's variation *across* key positions is ~10x larger than its near-constant component, so that's not it.

Re-ran the full sweep at finer granularity (16 through 2048) and it reproduces to the 5th decimal place — this is real, not noise. Extended down to proj_bits=1 (the theoretical minimum): identical result. The clean theoretical picture: MSE-only's error vs. the true unquantized baseline is a *fixed, deterministic* miss (the omitted Q·residual term, for a given rotation draw) that never shrinks; Prod's error is a *zero-mean, shrinking-variance* estimate of that same term. In principle Prod should cross below MSE once its variance drops under MSE's fixed miss. Empirically, across a 2048x range of m, that crossover never happens on this synthetic model at k_bits=3 — Prod's downstream KL is statistically indistinguishable from m=1 to m=2048, and consistently worse than MSE throughout.

This *strengthens* the Phase 2b recommendation rather than weakening it: "maybe more proj_bits fixes it" is now empirically closed off up to a 2048x range, not an open possibility. What's not resolved: why the crossover doesn't happen even at m=2048 — the working theory is that per-layer variance compounds across the 2 attention layers and subsequent nonlinearities in a way this single-layer estimator-accuracy analysis doesn't capture, but that's a hypothesis, not a traced mechanism. Real hardware with real weights (and more layers to actually observe compounding across) is what would resolve it, not more synthetic-model debugging in this sandbox.

**Variable-length batching — closed.** Previously flagged as untested (only same-length batches had been checked). Left-padded 3 different-length sequences (7/12/4 tokens) into one batch with an explicit `attention_mask`, generated, and confirmed each sequence's output matches generating it alone unpadded — real evidence padding doesn't leak into the quantized cache's per-vector scale computation. Now a permanent regression test.

**Server's `use_prod=True` path — closed.** The config flag existed and was documented but had never actually been exercised through the HTTP layer (all prior server tests used the MSE default). Confirmed working for both streaming and non-streaming through real requests. Now permanent tests.

54/54 pass.

## 19. Status As Of Right Now (the actually-current one — sections 12, 14, and 17 are progressively superseded snapshots, kept as an honest record of how this evolved rather than edited after the fact to look like it was always this clean)

| Phase | Status |
|---|---|
| 0 — VRAM budget | Done, verified |
| 1 — Core algorithm | Done, 100% tested |
| 2 — Cache integration (MSE, default) | Done, tested — same-length **and** variable-length/padded batches verified |
| 2b — Attention patching (Prod, off by default) | Done. Underperforms MSE; investigated in real depth (§18), not just flagged — crossover doesn't occur across a 2048x proj_bits range, so this isn't a tuning gap, it's a real open question needing real weights |
| 3 — Real-hardware validation | Runbook written, not run — needs your 3050 |
| 4 — Triton kernels | Not started — empirically confirmed un-runnable in this sandbox (no GPU driver), correctly gated behind Phase 3 data that doesn't exist yet |
| 5 — Pluggable server | Done, tested — including the `use_prod=True` path through the actual HTTP layer, streaming and non-streaming |
| Empire integration adapter | Built, honestly scoped as design-from-memory; its testable surface (health check, fallback) is tested, the real harness wiring isn't (not available here) |
| Repo-wide bug audit | Done — swept for the specific `str_replace`-orphaned-body pattern that bit this project twice; no further instances found |

54/54 tests. Two things need your actual GPU to close, and only your actual GPU: real Prod-vs-MSE quality, and real Phase 3 numbers to decide Phase 4.

## 20. Second Gap-Closing Pass

**Empire integration happy path — the actual gap this time.** Re-reading `test_empire_integration.py` found all three existing tests only exercised the *unhealthy* server path — `is_local_backend_healthy` returning `True`, and `get_llm_with_fallback` actually returning the local model, had never been tested. Closing it surfaced a real design gap: the sync health-check hits a real socket directly, which `httpx.ASGITransport` can't stand in for (async-only). Fixed by making `is_local_backend_healthy`/`get_local_llm`/`get_llm_with_fallback` accept injectable HTTP clients, and adding an async variant reliably testable via ASGI transport.

Also spot-checked the sync path against an actual live server in this sandbox — genuinely, not via a shortcut: 27 seconds to come up on the successful run; a prior attempt didn't respond within 40 seconds at all. Confirmed working when it came up (`is_local_backend_healthy` → True, `get_llm_with_fallback` correctly returned the local model over a Groq sentinel, `llm.invoke()` got a real response). That timing variance is why it's not in the automated suite — a flaky test is worse than no test — but it did run for real, once, successfully, in this session.

(Side finding, mechanical not architectural: this sandbox's bash tool returns an opaque `-1` with zero captured output on certain background-process-management command shapes, even when the command actually succeeded — confirmed by checking process state independently after the fact. Worked around by keeping background-process commands simpler and moving Python logic into separate files rather than inline `-c` strings.)

**README.md and pyproject.toml were stale** — test count (44 vs. actual 56) and the transformers version floor (`>=4.40`, which predates the CacheLayerMixin/AttentionInterface API this code actually requires — bumped to `>=5.14`, the version actually tested against, not a guessed-lower number).

**Phase 3's needle-in-haystack tensor logic** — never run at all before now (needs real weights). Exercised the actual concatenation/insertion/shape logic with a mock tokenizer: all three depth positions (10%/50%/90%) produce exactly the expected sequence length. Doesn't validate retrieval quality (needs a real model for that), but rules out an off-by-one or shape bug in code that had zero coverage until now.

56/56 automated tests, plus one manually-verified live-server confirmation outside the automated suite by design.

## 21. Final Verification — the actual deliverable, not the working copy

Everything above was checked against my working directory throughout the session. Before calling this done: extracted the actual zip fresh to a clean location and ran `pytest -q` exactly as the README instructs, from a cold extraction with none of the session's accumulated state. 56/56, clean. Cross-checked every file path the README references against the real package layout — all present, no stale references. `pyproject.toml` parses as valid TOML.

This is the artifact being handed off, verified as itself, not inferred from having verified something upstream of it.

## 22. Final Robustness Sweep — a real bug this time, in the tests themselves

Every theorem check and every batch-correctness test up to this point had only ever run at its one hardcoded default seed. Last check: sweep fresh seeds across all of them, since a test that only passes at the seed it was written against isn't verified, it's coincidental.

**Theorem checks (Lloyd-Max vs. uniform, distortion scaling, both Prod unbiasedness checks): robust.** All four pass cleanly across 5 fresh seeds never used during development (1, 7, 13, 99, 2026). Real confidence these are genuine properties, not artifacts of seed=0.

**Batch correctness: found a real bug, in the test, not the cache.** Sweeping 10 fresh seeds, seed=5 failed the same-length batch test. Traced it: that seed's third sequence hit EOS naturally after 3 new tokens in solo generation and correctly stopped there; in the batched run, the same sequence generates the same correct tokens up to EOS, then gets pad-continued with repeated EOS to match the batch's longest member, since HF's batched generation only stops once *every* sequence in the batch is done. Comparing full-length output was never the right check — it conflates "did the cache leak across batch items" (the actual thing worth testing, and true) with "does batched generation stop at exactly the same point as solo generation" (expected to differ, well-documented HF behavior, unrelated to this project).

Fixed both the same-length and variable-length batch tests to compare only up to each sequence's own natural stopping length. Re-swept: 10/10 seeds robust on both, including the seed that exposed the flaw. The earlier "batch correctness verified" claim (§13, §18) was true in substance — no cross-batch-item leakage was ever found, at any seed — but rested on a test that would have shown false negatives at other seeds. Worth being exact about: not a retraction, a correction to how it was being checked.

56/56, and for the first time, checked in a way that would have caught the one real gap in that number if it existed.
