"""
Ground-truth NF4 codebook and dequant semantics.

These constants and the nibble-order convention below were extracted directly
from bitsandbytes-foundation/bitsandbytes @ main:

  - Codebook values: csrc/kernels.cu, `nf4_dequantization_lut` (hardcoded,
    used by the actual CUDA dequant kernel).
  - Cross-checked against the live `bitsandbytes.functional.create_normal_map()`
    output (scipy.stats.norm.ppf-based) on bitsandbytes==0.50.0 -- the two
    sources agree to full float32 precision.
  - Nibble order and block-scale indexing: csrc/kernels.cu,
    `kDequantizeBlockwise`, the `NF4` case (around the `dDequantizeNF4` calls).

Verified 2026-08-04. If you're reading this months later, bitsandbytes may
have moved things around -- re-run the same grep against current `main` and
diff against NF4_LUT below before trusting it blindly.
"""

from __future__ import annotations

import torch

# index (4-bit code) -> dequantized value in [-1, 1]
NF4_LUT: list[float] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

DEFAULT_BLOCKSIZE = 64
VALID_BLOCKSIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def nf4_lut_tensor(device="cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(NF4_LUT, device=device, dtype=dtype)


def unpack_nibbles(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed uint8 bytes into (high_nibble, low_nibble) 4-bit codes.

    bitsandbytes convention (csrc/kernels.cu, kDequantizeBlockwise, NF4 case):

        out[2*j]     = LUT[byte[j] >> 4]     # high nibble -> even output index
        out[2*j + 1] = LUT[byte[j] & 0x0F]   # low nibble  -> odd output index

    Get this backwards and every dequantized value will be *plausible*
    (still a valid NF4 codebook entry) but silently wrong -- the kind of bug
    that survives a shape check and a "does it look like weights" glance.
    """
    packed = packed.to(torch.int32)
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    return high.to(torch.long), low.to(torch.long)


def reference_dequant_nf4(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int = DEFAULT_BLOCKSIZE,
) -> torch.Tensor:
    """Pure PyTorch (no Triton, no bitsandbytes, no GPU required) reference
    dequantization. Deliberately dumb and slow -- this is the thing every
    kernel in this repo gets checked against.

    `packed`: 1-D uint8 tensor, 2 codes per byte.
    `absmax`: 1-D float tensor, one scale per `blocksize`-sized block of
        *output* (unpacked) elements.

    Runs fine on CPU, which is how this was actually tested (see tests/) --
    the CUDA-kernel comparison in reference.py still needs a real GPU.
    """
    device = packed.device
    lut = nf4_lut_tensor(device=device, dtype=torch.float32)
    high, low = unpack_nibbles(packed)

    n_out = packed.numel() * 2
    out = torch.empty(n_out, device=device, dtype=torch.float32)
    out[0::2] = lut[high]
    out[1::2] = lut[low]

    block_idx = torch.arange(n_out, device=device) // blocksize
    return out * absmax[block_idx]
