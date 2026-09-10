"""
core/bitpack.py — Vectorized sub-byte bit-packing for quantized indices.

Fully vectorized via reshape + weighted-sum (no Python loops over the batch
or sequence dimensions) — cheap enough for correctness testing on CPU here.
Phase 4 replaces the hot path with fused Triton kernels; this stays as the
reference implementation those get checked against.
"""
from __future__ import annotations
import torch


def pack_bits(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """idx: integer tensor, values in [0, 2**bits - 1], shape [..., n].
    Returns uint8 tensor, shape [..., ceil(n*bits/8)]."""
    orig_shape = idx.shape
    n = orig_shape[-1]
    flat = idx.reshape(-1, n).to(torch.int64)

    bit_range = torch.arange(bits, dtype=torch.int64)
    bitstream = (flat.unsqueeze(-1) >> bit_range) & 1  # [B, n, bits]
    bitstream = bitstream.reshape(flat.shape[0], n * bits)

    total_bits = n * bits
    pad = (-total_bits) % 8
    if pad:
        bitstream = torch.cat(
            [bitstream, torch.zeros(flat.shape[0], pad, dtype=torch.int64)], dim=-1
        )
    n_bytes = bitstream.shape[-1] // 8
    bitstream = bitstream.reshape(flat.shape[0], n_bytes, 8)

    byte_weights = 2 ** torch.arange(8, dtype=torch.int64)
    packed = (bitstream * byte_weights).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*orig_shape[:-1], n_bytes)


def unpack_bits(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Inverse of pack_bits. n = number of original values packed per row."""
    orig_shape = packed.shape
    flat = packed.reshape(-1, orig_shape[-1]).to(torch.int64)

    bit_range8 = torch.arange(8, dtype=torch.int64)
    bitstream = (flat.unsqueeze(-1) >> bit_range8) & 1  # [B, n_bytes, 8]
    bitstream = bitstream.reshape(flat.shape[0], -1)[:, : n * bits]
    bitstream = bitstream.reshape(flat.shape[0], n, bits)

    val_weights = 2 ** torch.arange(bits, dtype=torch.int64)
    vals = (bitstream * val_weights).sum(dim=-1)
    return vals.reshape(*orig_shape[:-1], n)
