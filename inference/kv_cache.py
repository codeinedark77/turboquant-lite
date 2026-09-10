"""
inference/kv_cache.py — TurboQuantCacheLayer: a real transformers.cache_utils
.CacheLayerMixin subclass, so it drops into model.generate(past_key_values=...)
with zero changes to the model's forward pass.

Honest scope note (see ARCHITECTURE.md and the Phase 2 writeup this shipped
with): this layer quantizes BOTH keys and values with TurboQuantMSE
(dequantize-on-read, returns plain tensors). It does NOT wire in
TurboQuantProd's QJL bias-correction — that estimator needs the query
available at attention-score time, which means intercepting the attention
module's Q@K^T computation itself, not just the cache. Standard `update()`
returns tensors that flow into an unmodified attention matmul, so using a
Prod-quantized K here without patching attention would just store extra
residual/sign bits that nothing reads. Realizing the Prod path is a
separate, harder piece of work (attention-forward patching, model-specific)
and is explicitly not attempted in this phase.

Verified against transformers' current cache_utils.py (fetched directly,
not from training-data memory — the API changed to a per-layer
CacheLayerMixin design at some point after this model's training cutoff).
Required abstract methods: lazy_initialization, update, get_mask_sizes,
get_seq_length, get_max_cache_shape.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers.cache_utils import CacheLayerMixin, Cache

from core.quantizer import TurboQuantMSE


class TurboQuantCacheLayer(CacheLayerMixin):
    """Quantizes [batch, num_kv_heads, seq, head_dim] K/V states with
    TurboQuantMSE before storing; dequantizes the full accumulated cache on
    every `update()` call (simple and correct; not the fast path — Phase 4's
    Triton kernels are where a real incremental/fused version belongs)."""

    is_sliding = False

    def __init__(self, head_dim: int, k_bits: int = 3, v_bits: int = 4, codebook_seed: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.k_quant = TurboQuantMSE(head_dim, k_bits, seed=0, codebook_seed=codebook_seed)
        self.v_quant = TurboQuantMSE(head_dim, v_bits, seed=1, codebook_seed=codebook_seed + 1)
        # accumulated per-token quantized state, kept as lists of the dicts
        # TurboQuantMSE.quantize() returns -- concatenation happens at
        # dequant time along the sequence dim
        self._k_store: list[dict] = []
        self._v_store: list[dict] = []
        self._seq_len = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_kv_heads = key_states.shape[0], key_states.shape[1]
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """key_states/value_states: [batch, num_kv_heads, new_seq, head_dim]."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._k_store.append(self.k_quant.quantize(key_states))
        self._v_store.append(self.v_quant.quantize(value_states))
        self._seq_len += key_states.shape[-2]

        k_full = torch.cat([self.k_quant.dequantize(q) for q in self._k_store], dim=-2)
        v_full = torch.cat([self.v_quant.dequantize(q) for q in self._v_store], dim=-2)
        # Keep dtype consistent with what the attention module expects --
        # quantize/dequantize runs in float32 internally.
        return k_full.to(self.dtype), v_full.to(self.dtype)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1  # dynamic, no fixed max

    def get_max_length(self) -> int | None:
        return None  # dynamic, no fixed max -- same semantics as get_max_cache_shape

    def reset(self) -> None:
        self._k_store, self._v_store, self._seq_len = [], [], 0


def build_turboquant_cache(num_layers: int, head_dim: int, k_bits: int = 3, v_bits: int = 4) -> Cache:
    """Real transformers.cache_utils.Cache, pre-populated with one
    TurboQuantCacheLayer per model layer -- pass straight into
    model.generate(past_key_values=build_turboquant_cache(...))."""
    layers = [
        TurboQuantCacheLayer(head_dim, k_bits=k_bits, v_bits=v_bits, codebook_seed=i)
        for i in range(num_layers)
    ]
    return Cache(layers=layers)
