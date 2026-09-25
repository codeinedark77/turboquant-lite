"""
Milestone 2.5: fully-tiled fused kernel -- tiles M, N, *and* K, with proper
tail masking on all three. Removes both limitations `fused_kernel.py` (v1)
carries: no more full-N-per-program requirement, no more K%BLOCK_K==0
requirement.

STATUS: validated under the Triton interpreter across seven shapes chosen
specifically to be awkward: M-tail only, K-tail only, N-tail only, all three
tails at once, M=1 (GEMV-like), and a "33x97x41"-style set of shapes with no
common small factors. All seven matched a dequant-then-matmul reference to
within float32 tolerance. Real GPU compilation and bitsandbytes comparison:
still untested, same caveat as everywhere else in this repo.

HOW THIS DIFFERS FROM v1 (fused_kernel.py), MECHANICALLY: v1 keeps N whole
per program specifically so the packed-weight tile stays a CONTIGUOUS range
in the flat packed buffer -- a plain load. This version tiles N too, which
means an arbitrary (BLOCK_K, BLOCK_N) tile is generally NOT contiguous in
the row-major-flattened packed buffer (a tile that starts partway through a
row breaks contiguity). The fix here is a direct GATHER: compute the flat
index of every element in the tile individually
(`flat_offs = k_offs[:, None] * N + n_offs[None, :]`) and load via that
computed-offset tensor rather than a base pointer + contiguous range.
Triton's `tl.load` supports this -- it's the same mechanism the absmax
lookup already uses in v1, just applied to the (much larger) packed-weight
load too.

THE HONEST TRADEOFF: this is correct and general, but a gather is not a
coalesced memory access pattern. Real production kernels (Marlin) avoid
this by reshuffling the packed weight layout OFFLINE (once, at load time)
so that every tile IS contiguous post-reshuffle, turning the hot-path load
back into a plain contiguous one. That reshuffle is not implemented here.
Whether the gather in this version is fast enough to bother with vs. v1's
untiled-N simplicity, or whether the reshuffle is worth building, is
exactly the kind of question the benchmark script is for -- this repo
currently has no evidence either way, only the reasoning above.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._lut import codebook_kwargs, nf4_lut_lookup
from .codebook import DEFAULT_BLOCKSIZE


@triton.jit
def _fused_nf4_matmul_tiled_kernel(
    x_ptr, packed_ptr, absmax_ptr, out_ptr,
    M, N, K, blocksize, n_packed_bytes,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    C8: tl.constexpr, C9: tl.constexpr, C10: tl.constexpr, C11: tl.constexpr,
    C12: tl.constexpr, C13: tl.constexpr, C14: tl.constexpr, C15: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        x_tile = tl.load(
            x_ptr + m_offs[:, None] * K + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)

        # genuine gather: this (k0, n0) tile is not generally contiguous in
        # the flat packed buffer once N is tiled (see module docstring).
        flat_offs = k_offs[:, None] * N + n_offs[None, :]
        w_valid = k_mask[:, None] & n_mask[None, :]

        packed_offs = flat_offs // 2
        pmask = w_valid & (packed_offs < n_packed_bytes)
        is_high = (flat_offs % 2) == 0
        packed = tl.load(packed_ptr + packed_offs, mask=pmask, other=0).to(tl.int32)
        nib = tl.where(is_high, (packed >> 4) & 0x0F, packed & 0x0F)

        val = nf4_lut_lookup(nib, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15)

        blk = flat_offs // blocksize
        scale = tl.load(absmax_ptr + blk, mask=w_valid, other=1.0)
        # X is already zeroed in the invalid K region above, so any garbage
        # here from an out-of-range gather can't corrupt the accumulation
        # numerically -- the explicit tl.where is still needed though, since
        # `other=1.0` on the scale load combined with an unmasked nib could
        # otherwise produce inf/nan (e.g. 0 * inf) rather than a clean 0.
        w_tile = tl.where(w_valid, val * scale, 0.0)

        acc += tl.dot(x_tile, w_tile)

    tl.store(
        out_ptr + m_offs[:, None] * N + n_offs[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fused_nf4_matmul_tiled(
    x: torch.Tensor,
    packed: torch.Tensor,
    absmax: torch.Tensor,
    K: int,
    N: int,
    blocksize: int = DEFAULT_BLOCKSIZE,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Fully-tiled fused NF4 dequant+matmul. No shape restrictions -- M, N, K
    can be anything >= 1, none need to divide their BLOCK size.

    `x`: (M, K). `packed`/`absmax`: same convention as everywhere else in
    this repo (row-major-flattened (K, N) weight, blockwise NF4). Returns
    (M, N) float32.
    """
    M = x.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_nf4_matmul_tiled_kernel[grid](
        x, packed, absmax, out, M, N, K, blocksize, packed.numel(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        **codebook_kwargs(),
    )
    return out
