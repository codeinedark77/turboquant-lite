"""
inference/attention_hook.py — Phase 2b: wiring TurboQuantProd's bias
correction into live attention.

Phase 2's cache-only integration (inference/kv_cache.py) dequantizes on
read and hands back plain tensors, so the model's normal attention matmul
runs unmodified -- which meant the Prod/QJL path validated in Phase 1 was
never actually exercised (see ARCHITECTURE.md Phase 2 status). Fixing that
means intercepting the score computation itself.

The real hook for this, verified against current docs/source rather than
assumed: transformers' `AttentionInterface` registry. Models resolve their
score function via `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`
and call it as `fn(module, query_states, key_states, value_states,
attention_mask, dropout=..., scaling=..., **kwargs)` — this is the
documented, first-class extension point (transformers/docs/attention_interface.md),
not a raw monkeypatch of a model's forward method.

Design: `query_states`/`key_states`/`value_states` arriving here already
have RoPE applied and have already been through `Cache.update()` — but
`Cache.update()`'s contract only returns *dequantized* tensors, discarding
the residual/sign bits Prod needs. So `TurboQuantProdCacheLayer` below
keeps the raw Prod packet accessible (`get_k_packet()`), and the registered
attention function reaches it directly via `module.layer_idx` — a closure
over the same cache-layer list used to build the Cache, rather than trying
to thread the raw packet through Cache.update()'s tensor-only return
contract.

IMPORTANT correction made while building this: the estimator validated in
Phase 1 (`estimate_inner_product`) only computed *aligned* dot products
(query[i] against key[i]) — fine for the Phase 1 Monte Carlo check, wrong
for real attention, which needs every query position scored against every
key position. `core/quantizer.py::TurboQuantProd.estimate_attention_scores`
generalizes it to the full [S_q, S_k] matrix (matmul-based outer product)
and is independently re-validated for unbiasedness in
validation/theorems.py — the aligned-pair version being correct doesn't
automatically make the all-pairs version correct, so it was checked again
rather than assumed.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.cache_utils import CacheLayerMixin

from core.quantizer import TurboQuantProd, TurboQuantMSE


def repeat_kv_tensor(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Mirrors transformers' repeat_kv for GQA: [B, num_kv_heads, S, D] ->
    [B, num_kv_heads * n_rep, S, D]. Used here to expand the K packet's
    components (idx/scale/r_norm/signs), not just plain K/V tensors."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class TurboQuantProdCacheLayer(CacheLayerMixin):
    """Like Phase 2's TurboQuantCacheLayer, but keeps K in TurboQuantProd
    form and exposes the raw packet via get_k_packet() for the attention
    function below to consume directly. V is unchanged from Phase 2
    (TurboQuantMSE, dequantize-on-read) -- values are a reconstruction
    target, not an inner-product operand, so Prod's correction has nothing
    to fix there (see core/quantizer.py's module docstring)."""

    is_sliding = False

    def __init__(self, head_dim: int, k_bits: int = 3, v_bits: int = 4, proj_bits: int | None = None, codebook_seed: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.k_quant = TurboQuantProd(head_dim, k_bits, proj_bits=proj_bits, seed=0, codebook_seed=codebook_seed)
        self.v_quant = TurboQuantMSE(head_dim, v_bits, seed=1, codebook_seed=codebook_seed + 1)
        self._k_store: list[dict] = []
        self._v_store: list[dict] = []
        self._seq_len = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self._k_store.append(self.k_quant.quantize(key_states))
        self._v_store.append(self.v_quant.quantize(value_states))
        self._seq_len += key_states.shape[-2]
        # Fallback plain tensors -- shape/dtype-compatible with a standard
        # attention path; the registered attention fn below ignores the K
        # tensor it's handed and uses get_k_packet() instead, but V here IS
        # the real path (Prod has nothing to offer V; see class docstring).
        k_full = torch.cat([self.k_quant.dequantize(q) for q in self._k_store], dim=-2)
        v_full = torch.cat([self.v_quant.dequantize(q) for q in self._v_store], dim=-2)
        return k_full.to(self.dtype), v_full.to(self.dtype)

    def get_k_packet(self) -> dict:
        return {
            "idx": torch.cat([p["idx"] for p in self._k_store], dim=-2),
            "scale": torch.cat([p["scale"] for p in self._k_store], dim=-2),
            "r_norm": torch.cat([p["r_norm"] for p in self._k_store], dim=-2),
            "signs": torch.cat([p["signs"] for p in self._k_store], dim=-2),
        }

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_max_length(self) -> int | None:
        return None

    def reset(self) -> None:
        self._k_store, self._v_store, self._seq_len = [], [], 0


def build_turboquant_prod_cache(num_layers: int, head_dim: int, k_bits: int = 3, v_bits: int = 4, proj_bits: int | None = None):
    from transformers.cache_utils import Cache
    layers = [
        TurboQuantProdCacheLayer(head_dim, k_bits=k_bits, v_bits=v_bits, proj_bits=proj_bits, codebook_seed=i)
        for i in range(num_layers)
    ]
    return Cache(layers=layers), layers


def register_turboquant_prod_attention(cache_layers: list, name: str = "turboquant_prod") -> str:
    """Registers a custom attention function that scores queries against
    keys via TurboQuantProd's bias-corrected estimator instead of a plain
    matmul, closing over `cache_layers` (the SAME list used to build the
    Cache passed to generate()) since the raw K packet isn't retrievable
    through Cache.update()'s tensor-only return contract.
    Returns the implementation name to pass as attn_implementation=... .
    """

    def turboquant_prod_attention(module, query_states, key_states, value_states,
                                    attention_mask, scaling, dropout=0.0, **kwargs):
        layer = cache_layers[module.layer_idx]
        n_rep = query_states.shape[1] // key_states.shape[1]

        k_packet = layer.get_k_packet()
        if n_rep > 1:
            k_packet = {k: repeat_kv_tensor(v, n_rep) for k, v in k_packet.items()}
        value_states_r = repeat_kv_tensor(value_states, n_rep)

        q_rot = layer.k_quant.rotation.apply(query_states)
        scores = layer.k_quant.estimate_attention_scores(k_packet, q_rot) * scaling

        if attention_mask is not None:
            scores = scores + attention_mask[..., : k_packet["idx"].shape[-2]]

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        attn_output = torch.matmul(weights, value_states_r)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, weights

    AttentionInterface.register(name, turboquant_prod_attention)
    return name
