"""
Shared NF4 lookup-table helper, callable from other @triton.jit kernels.

Verified that jit-calling-jit composes correctly under the interpreter
(a standalone 16-value round-trip test matched `codebook.NF4_LUT` exactly)
before this was pulled out of naive_kernel.py / fused_kernel.py's inlined
copies. naive_kernel.py and fused_kernel.py still carry their own inlined
copies rather than being refactored to call this -- they were already
validated as-is, and touching working, tested code to deduplicate it is a
worse trade than a bit of repetition. fused_kernel_tiled.py and the
autotuned variant use this one.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .codebook import NF4_LUT


def codebook_kwargs() -> dict[str, float]:
    return {f"C{i}": v for i, v in enumerate(NF4_LUT)}


@triton.jit
def nf4_lut_lookup(
    idx,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    C8: tl.constexpr, C9: tl.constexpr, C10: tl.constexpr, C11: tl.constexpr,
    C12: tl.constexpr, C13: tl.constexpr, C14: tl.constexpr, C15: tl.constexpr,
):
    """idx: int tensor of 4-bit codes (0-15). Returns the dequantized float
    value, unscaled (caller applies the per-block absmax scale)."""
    v = tl.where(idx == 0, C0, 0.0)
    v = tl.where(idx == 1, C1, v)
    v = tl.where(idx == 2, C2, v)
    v = tl.where(idx == 3, C3, v)
    v = tl.where(idx == 4, C4, v)
    v = tl.where(idx == 5, C5, v)
    v = tl.where(idx == 6, C6, v)
    v = tl.where(idx == 7, C7, v)
    v = tl.where(idx == 8, C8, v)
    v = tl.where(idx == 9, C9, v)
    v = tl.where(idx == 10, C10, v)
    v = tl.where(idx == 11, C11, v)
    v = tl.where(idx == 12, C12, v)
    v = tl.where(idx == 13, C13, v)
    v = tl.where(idx == 14, C14, v)
    v = tl.where(idx == 15, C15, v)
    return v
