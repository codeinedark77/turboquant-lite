"""
validation/test_quantizer.py — pytest suite for core/.
Run: pytest validation/ -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from core.rotation import RandomRotation, random_orthogonal_matrix
from core.bitpack import pack_bits, unpack_bits
from core.quantizer import TurboQuantMSE, TurboQuantProd
from validation.theorems import (
    check_lloyd_max_beats_uniform,
    check_distortion_scaling,
    check_prod_unbiasedness,
    check_prod_attention_scores_unbiasedness,
)


# ---------- rotation ----------

def test_rotation_is_orthogonal():
    R = random_orthogonal_matrix(64, seed=1)
    I = torch.eye(64)
    assert torch.allclose(R @ R.T, I, atol=1e-4)
    assert torch.allclose(R.T @ R, I, atol=1e-4)


def test_rotation_preserves_inner_products():
    rot = RandomRotation(64, seed=2)
    a, b = torch.randn(10, 64), torch.randn(10, 64)
    ip_before = (a * b).sum(dim=-1)
    ip_after = (rot.apply(a) * rot.apply(b)).sum(dim=-1)
    assert torch.allclose(ip_before, ip_after, atol=1e-3, rtol=1e-3)


def test_rotation_preserves_norms():
    rot = RandomRotation(64, seed=3)
    x = torch.randn(20, 64)
    assert torch.allclose(x.norm(dim=-1), rot.apply(x).norm(dim=-1), atol=1e-3, rtol=1e-3)


def test_rotation_invert_recovers_input():
    rot = RandomRotation(32, seed=4)
    x = torch.randn(15, 32)
    recovered = rot.invert(rot.apply(x))
    assert torch.allclose(x, recovered, atol=1e-4)


# ---------- bitpack ----------

@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("n", [16, 37, 128])  # 37: deliberately not byte-aligned
def test_bitpack_roundtrip(bits, n):
    torch.manual_seed(0)
    idx = torch.randint(0, 2 ** bits, (5, n))
    packed = pack_bits(idx, bits)
    unpacked = unpack_bits(packed, bits, n=n)
    assert torch.equal(idx, unpacked)


def test_bitpack_actually_compresses():
    idx = torch.randint(0, 8, (100,))  # 3-bit values
    packed = pack_bits(idx, bits=3)
    assert packed.numel() < idx.numel()  # 300 bits -> 38 bytes vs 100 int64s


# ---------- quantizer round-trips ----------

def test_mse_quantizer_output_shape():
    tq = TurboQuantMSE(head_dim=64, bits=4, seed=5)
    x = torch.randn(3, 7, 64)
    q = tq.quantize(x)
    recon = tq.dequantize(q)
    assert recon.shape == x.shape


def test_mse_quantizer_error_shrinks_with_more_bits():
    torch.manual_seed(6)
    x = torch.randn(500, 64)
    errors = {}
    for bits in (2, 4, 6):
        tq = TurboQuantMSE(head_dim=64, bits=bits, seed=7, codebook_seed=7)
        recon = tq.dequantize(tq.quantize(x))
        errors[bits] = ((x - recon) ** 2).mean().item()
    assert errors[4] < errors[2]
    assert errors[6] < errors[4]


def test_prod_direct_term_consistent_with_mse_reconstruction():
    """TurboQuantProd's direct term (query_rot . k_hat_rot) should equal what
    you'd get by just dot-producting the query against a plain MSE
    dequantization -- the correction is additive on top, not a replacement."""
    head_dim, bits, seed = 64, 4, 8
    mse = TurboQuantMSE(head_dim, bits, seed=seed, codebook_seed=seed)
    prod = TurboQuantProd(head_dim, bits, seed=seed, codebook_seed=seed)

    k = torch.randn(10, head_dim)
    q = torch.randn(10, head_dim)

    mse_q = mse.quantize(k)
    k_hat_rot_via_mse = mse.reconstruct_rotated(mse_q)
    q_rot = mse.rotation.apply(q)
    direct_via_mse = (q_rot * k_hat_rot_via_mse).sum(dim=-1)

    prod_q = prod.quantize(k)
    prod_k_hat_rot = prod.reconstruct_rotated(prod_q)
    assert torch.allclose(k_hat_rot_via_mse, prod_k_hat_rot, atol=1e-5)
    assert torch.allclose(
        direct_via_mse, (q_rot * prod_k_hat_rot).sum(dim=-1), atol=1e-5
    )


# ---------- theorem checks, wired in as real pass/fail assertions ----------

def test_theorem_lloyd_max_beats_uniform():
    result = check_lloyd_max_beats_uniform()
    assert result["passed"], result


def test_theorem_distortion_scaling():
    result = check_distortion_scaling()
    assert result["passed"], result


def test_theorem_prod_unbiasedness():
    result = check_prod_unbiasedness()
    assert result["passed"], result


def test_theorem_prod_attention_scores_unbiasedness():
    result = check_prod_attention_scores_unbiasedness()
    assert result["passed"], result


def test_single_draw_correction_error_shrinks_with_proj_bits():
    """Estimator-level check, isolated from the full model: a SINGLE draw
    (not averaged over many trials, unlike check_prod_unbiasedness) of the
    Prod correction should show shrinking error as proj_bits grows, per the
    sqrt(1/m) variance-reduction the derivation predicts. This passing does
    NOT imply Prod beats plain MSE end-to-end in a real model at a fixed
    proj_bits budget -- see ARCHITECTURE.md Phase 2b status for the
    (unresolved) discrepancy between this isolated result and the
    full-model measurement."""
    from core.quantizer import TurboQuantProd
    torch.manual_seed(0)
    head_dim, bits = 8, 3
    q = torch.randn(1, head_dim)
    k = torch.randn(50, head_dim)
    true_scores = q @ k.T

    errors = []
    for proj_bits in (8, 32, 128, 512):
        torch.manual_seed(7)
        tq = TurboQuantProd(head_dim, bits, proj_bits=proj_bits, seed=7)
        packed = tq.quantize(k)
        q_rot = tq.rotation.apply(q)
        est = tq.estimate_attention_scores(packed, q_rot)
        errors.append((est - true_scores).abs().mean().item())

    assert errors[-1] < errors[0] * 0.5, errors
