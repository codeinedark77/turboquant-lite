"""
core/quantizer.py — TurboQuantMSE and TurboQuantProd.

TurboQuantMSE: rotate -> per-vector scale -> Lloyd-Max scalar quantize.
Minimizes reconstruction error. Used for Values: they're combined via a
weighted sum (attention_weights @ V), never used as an inner-product operand
themselves, so reconstruction fidelity is what actually matters.

TurboQuantProd: everything TurboQuantMSE does, plus a QJL (Quantized
Johnson-Lindenstrauss) bias correction on the *residual* left over after
Lloyd-Max quantization. Used for Keys, because attention scores are inner
products (Q . K), and a plain MSE-optimal reconstruction is not automatically
the right thing to plug into a downstream inner product — the systematic
error needs correcting, not just minimizing.

Derivation of the correction (worth writing out since it's the one part of
this module reconstructed from a description rather than the paper itself):

  Let k_rot = k_hat + r, where k_hat is the exact Lloyd-Max reconstruction
  and r is the (small, known-at-encode-time) residual. We want an unbiased
  estimate of q_rot . k_rot = q_rot . k_hat + q_rot . r. The first term is
  exact. For the second: let g be a row of a random N(0, I_d) projection
  matrix S. For FIXED q_rot, r, the pair (g.q_rot, g.r) is jointly Gaussian
  with Cov = q_rot . r and sigma_{g.r} = ||r||. The classical Gaussian
  sign-correlation identity (Price's theorem / a Stein's-lemma corollary),

      E[A * sign(B)] = sqrt(2/pi) * Cov(A,B) / sigma_B      for jointly
                                                              Gaussian A, B,

  gives E[(g.q_rot) * sign(g.r)] = sqrt(2/pi) * (q_rot.r) / ||r||. Solving
  for q_rot.r and averaging over m independent projection rows to cut
  variance:

      q_rot . r  ~=  ||r|| * sqrt(pi/2) / m * sum_j (g_j.q_rot) * sign(g_j.r)

  This is exact in expectation over the random projection, for any fixed
  q_rot and r — i.e. unbiased by construction, *given the derivation above is
  correct*. That claim is checked empirically, not just trusted, in
  validation/theorems.py::check_prod_unbiasedness.
"""
from __future__ import annotations
import math
import torch

from .rotation import RandomRotation, sign_projection_matrix
from .codebook import build_codebook


class TurboQuantMSE:
    def __init__(self, head_dim: int, bits: int, seed: int = 0, codebook_seed: int = 0, device="cpu"):
        """`seed` controls the per-instance random rotation (and, in
        TurboQuantProd, the projection matrix) -- vary this freely across
        instances/trials. `codebook_seed` controls the Lloyd-Max codebook,
        which approximates a fixed population distribution for this
        (head_dim, bits) and should normally stay FIXED across instances so
        build_codebook's cache actually hits instead of refitting from
        scratch on every construction."""
        self.head_dim = head_dim
        self.bits = bits
        self.rotation = RandomRotation(head_dim, seed=seed, device=device)
        cb = build_codebook(head_dim, bits, seed=codebook_seed)
        self.levels = cb["levels"].to(device)
        self.boundaries = cb["boundaries"].to(device)

    def _quantize_rotated(self, x: torch.Tensor):
        """x: [..., head_dim], already rotated. Returns (idx, scale)."""
        scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm = x / scale
        if self.boundaries.device != x_norm.device:
            self.boundaries = self.boundaries.to(x_norm.device)
        if self.levels.device != x_norm.device:
            self.levels = self.levels.to(x_norm.device)
        idx = torch.bucketize(x_norm.contiguous(), self.boundaries)
        return idx, scale

    def quantize(self, x: torch.Tensor) -> dict:
        """x: [..., head_dim], NOT yet rotated (raw K or V vectors)."""
        x_rot = self.rotation.apply(x)
        idx, scale = self._quantize_rotated(x_rot)
        return {"idx": idx, "scale": scale}

    def reconstruct_rotated(self, q: dict) -> torch.Tensor:
        """Dequantized value in *rotated* space (skips the inverse rotation;
        used internally when the caller is going to work in rotated space
        anyway, e.g. TurboQuantProd)."""
        return self.levels[q["idx"]] * q["scale"]

    def dequantize(self, q: dict) -> torch.Tensor:
        return self.rotation.invert(self.reconstruct_rotated(q))


class TurboQuantProd(TurboQuantMSE):
    def __init__(
        self,
        head_dim: int,
        bits: int,
        proj_bits: int | None = None,
        seed: int = 0,
        codebook_seed: int = 0,
        device="cpu",
    ):
        super().__init__(head_dim, bits, seed=seed, codebook_seed=codebook_seed, device=device)
        # Default m = head_dim: "roughly one residual bit per original
        # dimension," matching how the source material frames the QJL step's
        # cost. Not independently verified against the paper's exact
        # hyperparameter choice -- treat as a tunable default, not a
        # reproduction. See ARCHITECTURE.md.
        self.proj_bits = proj_bits or head_dim
        self.S = sign_projection_matrix(head_dim, self.proj_bits, seed=seed + 2, device=device)

    def quantize(self, x: torch.Tensor) -> dict:
        x_rot = self.rotation.apply(x)
        idx, scale = self._quantize_rotated(x_rot)
        if self.levels.device != idx.device:
            self.levels = self.levels.to(idx.device)
        k_hat_rot = self.levels[idx] * scale
        residual = x_rot - k_hat_rot
        r_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        if self.S.device != residual.device or self.S.dtype != residual.dtype:
            self.S = self.S.to(device=residual.device, dtype=residual.dtype)
        signs = torch.sign(residual @ self.S.T)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return {"idx": idx, "scale": scale, "r_norm": r_norm, "signs": signs}

    def estimate_inner_product(self, q: dict, query_rot: torch.Tensor) -> torch.Tensor:
        """query_rot: [..., head_dim], the QUERY vector already rotated by
        this SAME instance's rotation (call self.rotation.apply(query) — a
        different rotation instance will silently give a meaningless
        result, since k_hat/residual live in this instance's rotated frame).
        Returns an estimate of query . key for each *aligned* position
        (query[i] against key[i]), unbiased over the random draw of self.S
        (see module docstring). For real attention's all-pairs score matrix
        (every query position against every key position), use
        estimate_attention_scores instead — this method alone would silently
        compute only the diagonal, which is a different (wrong) thing."""
        if self.levels.device != q["idx"].device:
            self.levels = self.levels.to(q["idx"].device)
        k_hat_rot = self.reconstruct_rotated(q)
        direct = (query_rot * k_hat_rot).sum(dim=-1)

        if self.S.device != query_rot.device or self.S.dtype != query_rot.dtype:
            self.S = self.S.to(device=query_rot.device, dtype=query_rot.dtype)
        q_proj = query_rot @ self.S.T  # [..., proj_bits]
        m = self.proj_bits
        correction = (
            math.sqrt(math.pi / 2) / m
            * q["r_norm"].squeeze(-1)
            * (q_proj * q["signs"]).sum(dim=-1)
        )
        return direct + correction

    def estimate_attention_scores(self, q: dict, query_rot: torch.Tensor) -> torch.Tensor:
        """query_rot: [..., S_q, head_dim]; q's tensors: [..., S_k, ...].
        Returns the full [..., S_q, S_k] score matrix — every query position
        against every key position — which is what real attention needs
        (softmax is over S_k for each query position, not just a single
        aligned pair). Same unbiased-per-entry derivation as
        estimate_inner_product, applied pairwise instead of aligned:
        direct[i,j] = q_rot[i] . k_hat_rot[j], correction[i,j] built the same
        way via two matmuls instead of one reduction."""
        if self.levels.device != q["idx"].device:
            self.levels = self.levels.to(q["idx"].device)
        k_hat_rot = self.reconstruct_rotated(q)  # [..., S_k, D]
        direct = query_rot @ k_hat_rot.transpose(-1, -2)  # [..., S_q, S_k]

        if self.S.device != query_rot.device or self.S.dtype != query_rot.dtype:
            self.S = self.S.to(device=query_rot.device, dtype=query_rot.dtype)
        q_proj = query_rot @ self.S.T  # [..., S_q, proj_bits]
        corr_raw = q_proj @ q["signs"].transpose(-1, -2)  # [..., S_q, S_k]
        r_norm_row = q["r_norm"].transpose(-1, -2)  # [..., 1, S_k]
        m = self.proj_bits
        correction = (math.sqrt(math.pi / 2) / m) * corr_raw * r_norm_row
        return direct + correction
