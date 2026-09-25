"""
core/rotation.py — Random orthogonal rotation and QJL-style sign projection.

TurboQuant's first stage (PolarQuant) randomly rotates each K/V vector before
quantizing. Key fact this relies on: if R is a Haar-random orthogonal matrix
and x is *any* fixed vector, then R @ x is uniformly distributed on the
sphere of radius ||x||. So after rotation, the coordinate distribution of
*any* input vector collapses to the same universal shape (up to the unknown
scale ||x||), regardless of what the original, un-rotated vector looked like.
That's what lets a single precomputed scalar codebook (codebook.py) work well
across many different real K/V vectors.

Rotation also preserves inner products exactly: (R@a)·(R@b) == a·b, since
R^T R = I. quantizer.py relies on this — it estimates q·k by rotating both
q and k the same way and working entirely in rotated coordinates.
"""
from __future__ import annotations
import torch


def random_orthogonal_matrix(
    d: int, seed: int | None = None, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    Haar-random d x d orthogonal matrix via QR decomposition of a Gaussian
    matrix, with the sign correction needed for a *uniform* distribution over
    O(d) (plain QR without it is biased — see Mezzadri, "How to generate
    random matrices from the classical compact groups", 2007).
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    A = torch.randn(d, d, generator=gen, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    d_sign = torch.sign(torch.diagonal(R))
    d_sign = torch.where(d_sign == 0, torch.ones_like(d_sign), d_sign)
    Q = Q * d_sign.unsqueeze(0)
    return Q


class RandomRotation:
    """Applies / inverts a fixed random orthogonal rotation on the last dim."""

    def __init__(self, dim: int, seed: int | None = None, device="cpu", dtype=torch.float32):
        self.dim = dim
        self.device = device
        self.R = random_orthogonal_matrix(dim, seed=seed, device=device, dtype=dtype)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.R.device != x.device or self.R.dtype != x.dtype:
            self.R = self.R.to(device=x.device, dtype=x.dtype)
        return x @ self.R

    def invert(self, x_rot: torch.Tensor) -> torch.Tensor:
        if self.R.device != x_rot.device or self.R.dtype != x_rot.dtype:
            self.R = self.R.to(device=x_rot.device, dtype=x_rot.dtype)
        # R is orthogonal: R^-1 == R^T
        return x_rot @ self.R.T

    def to(self, device):
        self.R = self.R.to(device)
        self.device = device
        return self


def sign_projection_matrix(
    d: int, m: int, seed: int | None = None, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    m x d matrix with iid N(0,1) entries — deliberately UNnormalized. The QJL
    residual-correction estimator in quantizer.py is derived assuming raw
    standard-normal projection rows (see the docstring there); scaling this
    matrix would require re-deriving the estimator's constant.
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return torch.randn(m, d, generator=gen, device=device, dtype=dtype)
