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

class TurboQuantCacheLayer(CacheLayerMixin):
    """Stores Keys and Values in natively packed 4-bit NF4 bytes using bitsandbytes,
    with blockwise scaling factors. No float tensors are ever materialized here."""

    is_sliding = False

    def __init__(self, head_dim: int, k_bits: int = 4, v_bits: int = 4, codebook_seed: int = 0):
        super().__init__()
        self.head_dim = head_dim
        
        # NF4 parameters for Keys and Values
        self.blocksize = 64
        
        self._k_packed: list[torch.Tensor] = []
        self._k_absmax: list[torch.Tensor] = []
        
        self._v_packed: list[torch.Tensor] = []
        self._v_absmax: list[torch.Tensor] = []
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

        import bitsandbytes.functional as bnbF
        
        # Quantize key_states to NF4
        k_flat = key_states.contiguous().view(-1)
        if k_flat.is_cuda:
            k_packed, k_quant_state = bnbF.quantize_4bit(
                k_flat, blocksize=self.blocksize, quant_type="nf4", compress_statistics=False
            )
            self._k_packed.append(k_packed)
            self._k_absmax.append(k_quant_state.absmax)
        else:
            self._k_packed.append(k_flat.cpu())
            self._k_absmax.append(torch.tensor([], device="cpu"))
            
        # Quantize value_states to NF4
        v_flat = value_states.contiguous().view(-1)
        if v_flat.is_cuda:
            v_packed, v_quant_state = bnbF.quantize_4bit(
                v_flat, blocksize=self.blocksize, quant_type="nf4", compress_statistics=False
            )
            self._v_packed.append(v_packed)
            self._v_absmax.append(v_quant_state.absmax)
        else:
            self._v_packed.append(v_flat.cpu())
            self._v_absmax.append(torch.tensor([], device="cpu"))
            
        self._seq_len += key_states.shape[-2]

        # Return dummy tensors. FlashAttention will bypass these and read the packed cache directly.
        k_dummy = torch.empty((self.batch_size, self.num_kv_heads, self._seq_len, self.head_dim), device=self.device, dtype=self.dtype)
        v_dummy = torch.empty((self.batch_size, self.num_kv_heads, self._seq_len, self.head_dim), device=self.device, dtype=self.dtype)

        return k_dummy, v_dummy

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1  # dynamic, no fixed max

    def get_max_length(self) -> int | None:
        return None  # dynamic, no fixed max -- same semantics as get_max_cache_shape

    def reset(self) -> None:
        self._seq_len = 0
        self._k_packed, self._k_absmax = [], []
        self._v_packed, self._v_absmax = [], []


def build_turboquant_cache(num_layers: int, head_dim: int, k_bits: int = 3, v_bits: int = 4) -> Cache:
    """Real transformers.cache_utils.Cache, pre-populated with one
    TurboQuantCacheLayer per model layer -- pass straight into
    model.generate(past_key_values=build_turboquant_cache(...))."""
    layers = [
        TurboQuantCacheLayer(head_dim, k_bits=k_bits, v_bits=v_bits, codebook_seed=i)
        for i in range(num_layers)
    ]
    return Cache(layers=layers)
