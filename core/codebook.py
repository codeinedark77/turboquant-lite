"""
core/codebook.py — Empirical Lloyd-Max scalar quantizer.

After PolarQuant's rotation (rotation.py), a coordinate of a rotated vector
with norm r behaves like r times a coordinate of a uniform random point on
the unit sphere in R^d (true for *any* fixed input, since a Haar rotation of
a fixed point is uniform on its sphere). That induced marginal is a
symmetric, concentrated, Beta-like distribution on [-1, 1] whose exact shape
depends on d.

Rather than deriving and hardcoding the closed-form Beta(alpha, beta)
parameterization for that marginal (real risk of getting the parameters
subtly wrong without the full derivation in front of me), this builds the
codebook *empirically*: sample many coordinates of true random unit vectors
in R^d directly — which is easy and exact, no approximation — and run the
classical Lloyd-Max algorithm on those samples. At the sample sizes used
here this converges to the same codebook the closed-form approach would
produce, without needing the closed form at all.

The resulting codebook is dimensionless (built for unit-norm input); real
K/V vectors are rescaled by their own norm before lookup (quantizer.py).
"""
from __future__ import annotations
import torch


def sample_unit_sphere_coordinates(
    d: int, n_vectors: int, seed: int | None = None, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """Draw n_vectors uniform random points on the unit sphere in R^d and
    return all d * n_vectors coordinates pooled into one 1-D sample."""
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    g = torch.randn(n_vectors, d, generator=gen, device=device, dtype=dtype)
    g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return g.reshape(-1)


def fit_lloyd_max(samples: torch.Tensor, bits: int, iters: int = 50, tol: float = 1e-7) -> torch.Tensor:
    """
    Classical Lloyd-Max scalar quantizer fit to `samples` (1-D tensor):
    iterate [nearest-level assignment] -> [conditional-mean update] to
    convergence. Returns the 2**bits reconstruction levels, sorted ascending.
    """
    levels = 2 ** bits
    samples = samples.flatten().to(torch.float64)  # fp64: this is a one-time
    # offline fit, not a hot path, and low-bit conditional-mean updates are
    # numerically touchy enough to be worth the precision.

    qs = torch.linspace(0, 1, levels + 2, dtype=torch.float64)[1:-1]
    # torch.quantile has an input-size ceiling (~16.7M elements) we blow past
    # at realistic (head_dim * n_vectors) sizes. Init is just a starting
    # point for the iteration below, not the fit itself -- a subsample is
    # plenty representative.
    init_sample = samples if samples.numel() <= 1_000_000 else samples[
        torch.randperm(samples.numel())[:1_000_000]
    ]
    recon, _ = torch.sort(torch.quantile(init_sample, qs))

    prev_distortion = None
    for _ in range(iters):
        boundaries = (recon[:-1] + recon[1:]) / 2
        idx = torch.bucketize(samples, boundaries)  # values in [0, levels-1]

        sums = torch.zeros(levels, dtype=torch.float64)
        counts = torch.zeros(levels, dtype=torch.float64)
        sums.scatter_add_(0, idx, samples)
        counts.scatter_add_(0, idx, torch.ones_like(samples))

        empty = counts == 0
        new_recon = torch.where(empty, recon, sums / counts.clamp(min=1))
        new_recon, _ = torch.sort(new_recon)

        distortion = ((samples - recon[idx]) ** 2).mean().item()
        recon = new_recon
        if prev_distortion is not None and abs(prev_distortion - distortion) < tol:
            break
        prev_distortion = distortion

    return recon.to(torch.float32)


_CODEBOOK_CACHE: dict[tuple[int, int, int], dict] = {}


def build_codebook(head_dim: int, bits: int, n_vectors: int = 10_000, seed: int = 0) -> dict:
    """End-to-end: sample the induced coordinate distribution for this
    head_dim, fit Lloyd-Max, return the codebook. Cached per (head_dim, bits,
    seed) — this is meant to be computed once per model config and reused,
    not regenerated per call (see ARCHITECTURE.md Known Unknowns).

    n_vectors=10_000 (-> 10_000*head_dim pooled samples, e.g. 1.28M at
    head_dim=128) is sized for this dev sandbox, measured at exactly 1 CPU
    core: ~2-3s per fit at these settings, vs. 60s+ at the 200k-vector size
    tried first. Statistically this is still comfortably oversized for
    fitting <=256 levels (8-bit) — even at 16 levels that's tens of
    thousands of samples per bucket. Bump it if running on real (multi-core)
    hardware and chasing a tighter fit."""
    key = (head_dim, bits, seed)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]
    samples = sample_unit_sphere_coordinates(head_dim, n_vectors, seed=seed)
    levels = fit_lloyd_max(samples, bits)
    boundaries = (levels[:-1] + levels[1:]) / 2
    cb = {"levels": levels, "boundaries": boundaries, "head_dim": head_dim, "bits": bits}
    _CODEBOOK_CACHE[key] = cb
    return cb
