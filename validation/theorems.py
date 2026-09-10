"""
validation/theorems.py — CPU-only checks of the core algorithm's claimed
properties. This is the Phase 1 gate: nothing in inference/ gets touched
until these pass for real, on real (if synthetic) data — not asserted from
the derivation alone.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.codebook import sample_unit_sphere_coordinates, fit_lloyd_max
from core.quantizer import TurboQuantProd


def check_lloyd_max_beats_uniform(head_dim=128, bits=4, n_samples=10_000, seed=0) -> dict:
    """Lloyd-Max is MSE-optimal by construction; on held-out samples from the
    true induced distribution it must match or beat a naive uniform
    quantizer at the same bit-width."""
    train = sample_unit_sphere_coordinates(head_dim, n_samples, seed=seed)
    test = sample_unit_sphere_coordinates(head_dim, n_samples, seed=seed + 999)

    lm_levels = fit_lloyd_max(train, bits)
    lm_boundaries = (lm_levels[:-1] + lm_levels[1:]) / 2
    lm_idx = torch.bucketize(test, lm_boundaries)
    lm_mse = ((test - lm_levels[lm_idx]) ** 2).mean().item()

    lo, hi = train.min(), train.max()
    n_levels = 2 ** bits
    uni_levels = torch.linspace(lo, hi, n_levels)
    uni_boundaries = (uni_levels[:-1] + uni_levels[1:]) / 2
    uni_idx = torch.bucketize(test, uni_boundaries)
    uni_mse = ((test - uni_levels[uni_idx]) ** 2).mean().item()

    passed = lm_mse <= uni_mse * 1.01  # 1% slack for Monte Carlo noise
    return {
        "name": "lloyd_max_beats_uniform",
        "passed": passed,
        "lloyd_max_mse": lm_mse,
        "uniform_mse": uni_mse,
    }


def check_distortion_scaling(head_dim=128, bit_range=(2, 3, 4, 5), n_samples=10_000, seed=0) -> dict:
    """High-rate scalar quantization theory: MSE distortion falls roughly 4x
    per extra bit (Bennett / Panter-Dite). Checked loosely — 2-3 bit results
    sit outside the asymptotic 'high-rate' regime the theorem strictly
    assumes, so this checks direction + rough magnitude, not a tight ratio."""
    results = {}
    for b in bit_range:
        train = sample_unit_sphere_coordinates(head_dim, n_samples, seed=seed)
        test = sample_unit_sphere_coordinates(head_dim, n_samples, seed=seed + 999)
        levels = fit_lloyd_max(train, b)
        boundaries = (levels[:-1] + levels[1:]) / 2
        idx = torch.bucketize(test, boundaries)
        results[b] = ((test - levels[idx]) ** 2).mean().item()

    monotonic = all(results[bit_range[i + 1]] < results[bit_range[i]] for i in range(len(bit_range) - 1))
    ratios = [results[bit_range[i]] / results[bit_range[i + 1]] for i in range(len(bit_range) - 1)]
    reasonable = all(1.5 <= r <= 8.0 for r in ratios)
    return {
        "name": "distortion_scaling",
        "passed": monotonic and reasonable,
        "mse_by_bits": results,
        "ratios": ratios,
    }


def check_prod_unbiasedness(head_dim=64, bits=3, proj_bits=64, n_keys=200, n_trials=200, seed=0) -> dict:
    """Monte Carlo check: for fixed (query, key) pairs, average the Prod
    estimator over many independent random rotations+projections and confirm
    the mean converges toward the true q.k. This checks *systematic* bias —
    any single draw is expected to be noisy (it's a 1-bit-per-projection
    randomized estimator); the claim under test is that the noise doesn't
    have a directional skew."""
    torch.manual_seed(seed)
    q = torch.randn(n_keys, head_dim)
    k = torch.randn(n_keys, head_dim)
    true_ip = (q * k).sum(dim=-1)

    estimates = torch.zeros(n_trials, n_keys)
    for t in range(n_trials):
        tq = TurboQuantProd(head_dim, bits, proj_bits=proj_bits, seed=seed * 10_000 + t)
        packed = tq.quantize(k)
        q_rot = tq.rotation.apply(q)
        estimates[t] = tq.estimate_inner_product(packed, q_rot)

    mean_est = estimates.mean(dim=0)
    bias = mean_est - true_ip
    rel_bias = (bias.abs() / true_ip.abs().clamp(min=1e-6)).median().item()
    # generous band: this flags a gross derivation/sign/constant error, it's
    # not meant to certify tight production-grade accuracy
    passed = rel_bias < 0.15
    return {
        "name": "prod_unbiasedness",
        "passed": passed,
        "median_relative_bias": rel_bias,
        "mean_abs_true_ip": true_ip.abs().mean().item(),
    }


def check_prod_attention_scores_unbiasedness(
    head_dim=32, bits=4, proj_bits=32, n_q=5, n_k=7, n_trials=200, seed=0
) -> dict:
    """Same claim as check_prod_unbiasedness, but for estimate_attention_scores
    -- the all-pairs [S_q, S_k] version actually used in real attention,
    which is a materially different computation (matmul-based outer product,
    not an aligned per-position reduction) built from the same identity."""
    torch.manual_seed(seed)
    q = torch.randn(n_q, head_dim)
    k = torch.randn(n_k, head_dim)
    true_scores = q @ k.T

    estimates = torch.zeros(n_trials, n_q, n_k)
    for t in range(n_trials):
        tq = TurboQuantProd(head_dim, bits, proj_bits=proj_bits, seed=seed * 10_000 + t)
        packed = tq.quantize(k)
        q_rot = tq.rotation.apply(q)
        estimates[t] = tq.estimate_attention_scores(packed, q_rot)

    mean_est = estimates.mean(dim=0)
    bias = mean_est - true_scores
    rel_bias = (bias.abs() / true_scores.abs().clamp(min=1e-6)).median().item()
    passed = rel_bias < 0.15
    return {
        "name": "prod_attention_scores_unbiasedness",
        "passed": passed,
        "median_relative_bias": rel_bias,
    }


def run_all(verbose: bool = True) -> list:
    checks = [
        check_lloyd_max_beats_uniform(),
        check_distortion_scaling(),
        check_prod_unbiasedness(),
        check_prod_attention_scores_unbiasedness(),
    ]
    if verbose:
        for c in checks:
            status = "PASS" if c["passed"] else "FAIL"
            detail = {k: v for k, v in c.items() if k not in ("name", "passed")}
            print(f"[{status}] {c['name']}: {detail}")
    return checks


if __name__ == "__main__":
    results = run_all()
    n_pass = sum(r["passed"] for r in results)
    print(f"\n{n_pass}/{len(results)} theorem checks passed")
    sys.exit(0 if n_pass == len(results) else 1)
