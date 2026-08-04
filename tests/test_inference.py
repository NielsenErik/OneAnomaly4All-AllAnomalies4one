"""
Tests for the exact-inference routines, the new leaf types, the two extra
property validators, and the SOS (squared-circuit) mode.

The marginal / partition tests check exactness numerically: marginals must
agree with brute-force grid integration of the modeled density, and every
density must integrate to 1.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probabilistic_circuits import (
    CategoricalLeaf,
    DensityPC,
    GaussianLeaf,
    GaussianMixtureLeaf,
    SquaredPC,
    SumNode,
    VtreeInternal,
    VtreeLeaf,
    eval_log_marginal,
    log_partition,
    mpe,
    random_balanced_vtree,
    validate_circuit,
    validate_decomposability,
    validate_determinism,
    validate_smoothness,
    validate_structured_decomposability,
)

torch.manual_seed(0)


def vtree2():
    return VtreeInternal(VtreeLeaf(0), VtreeLeaf(1))


def vtree4():
    return VtreeInternal(
        left=VtreeInternal(VtreeLeaf(0), VtreeLeaf(1)),
        right=VtreeInternal(VtreeLeaf(2), VtreeLeaf(3)),
    )


# ─── Partition function ───────────────────────────────────────────────────────

@pytest.mark.parametrize("leaf", [None, GaussianLeaf, lambda i: GaussianMixtureLeaf(i, 3)])
def test_monotone_partition_is_one(leaf):
    pc = DensityPC(vtree4(), n_sum_components=3, leaf_factory=leaf)
    assert abs(float(pc.log_partition())) < 1e-5


def test_density_integrates_to_one_grid():
    """Brute-force 2-D grid integration of the modeled density ≈ 1."""
    pc = DensityPC(vtree2(), n_sum_components=2, leaf_factory=GaussianLeaf)
    pc.fit_leaves(torch.randn(100, 2))
    g = torch.linspace(-12, 12, 401)
    dx = float(g[1] - g[0])
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    with torch.no_grad():
        total = torch.exp(pc.log_prob(pts)).sum() * dx * dx
    assert abs(float(total) - 1.0) < 1e-3


# ─── Marginals ────────────────────────────────────────────────────────────────

def test_marginal_none_equals_log_prob():
    pc = DensityPC(vtree4(), n_sum_components=3)
    x = torch.randn(7, 4)
    assert torch.allclose(pc.log_marginal(x, []), pc.log_prob(x), atol=1e-5)


def test_marginal_all_is_zero():
    """Marginalizing everything yields the partition function = 1."""
    pc = DensityPC(vtree4(), n_sum_components=3)
    x = torch.randn(5, 4)
    out = pc.log_marginal(x, [0, 1, 2, 3])
    assert torch.allclose(out, torch.zeros(5), atol=1e-5)


def test_marginal_matches_grid_integration():
    """p(x0) from the circuit ≈ ∫ p(x0, t) dt computed on a grid."""
    pc = DensityPC(vtree2(), n_sum_components=2, leaf_factory=GaussianLeaf)
    pc.fit_leaves(torch.randn(100, 2))
    x0 = torch.tensor([0.3])
    exact = pc.log_marginal(torch.tensor([[0.3, 999.0]]), [1])  # value at idx 1 ignored

    g = torch.linspace(-12, 12, 4001)
    dx = float(g[1] - g[0])
    pts = torch.stack([x0.repeat(len(g)), g], dim=1)
    with torch.no_grad():
        brute = torch.log(torch.exp(pc.log_prob(pts)).sum() * dx)
    assert abs(float(exact) - float(brute)) < 1e-3


def test_marginal_ignores_marginalized_values():
    pc = DensityPC(vtree4(), n_sum_components=2)
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b = torch.tensor([[1.0, 2.0, -50.0, 4.0]])
    assert torch.allclose(pc.log_marginal(a, [2]), pc.log_marginal(b, [2]), atol=1e-6)


# ─── MPE ──────────────────────────────────────────────────────────────────────

def test_mpe_deterministic_returns_leaf_modes():
    """K=1 (no sum nodes) is deterministic: MPE = leaf modes, exactly."""
    pc = DensityPC(vtree4(), n_sum_components=1, leaf_factory=GaussianLeaf)
    X = torch.randn(50, 4) + torch.tensor([1.0, -2.0, 0.5, 3.0])
    pc.fit_leaves(X)
    assignment, log_val = pc.mpe()
    expected = torch.tensor([float(m.mu) for m in sorted(
        (mm for mm in pc.modules() if isinstance(mm, GaussianLeaf)),
        key=lambda mm: mm.feature_idx)])
    assert torch.allclose(assignment, expected, atol=1e-5)
    # MPE log value equals the density at the returned assignment
    assert abs(log_val - float(pc.log_prob(assignment.unsqueeze(0)))) < 1e-4


def test_mpe_respects_evidence():
    pc = DensityPC(vtree4(), n_sum_components=1, leaf_factory=GaussianLeaf)
    assignment, _ = pc.mpe(evidence={2: 7.5})
    assert abs(float(assignment[2]) - 7.5) < 1e-6


def test_mpe_at_least_as_good_as_random_completion():
    pc = DensityPC(vtree4(), n_sum_components=3, leaf_factory=GaussianLeaf)
    pc.fit_leaves(torch.randn(100, 4))
    assignment, log_val = pc.mpe()
    rand_val = float(pc.log_prob(torch.randn(1, 4) * 3))
    assert log_val >= rand_val - 1e-6 or log_val > rand_val


# ─── Categorical leaves ───────────────────────────────────────────────────────

def test_categorical_pc_sums_to_one():
    pc = DensityPC(vtree2(), n_sum_components=2,
                   leaf_factory=lambda i: CategoricalLeaf(i, n_categories=3))
    states = torch.tensor([[a, b] for a in range(3) for b in range(3)], dtype=torch.float32)
    total = float(torch.exp(pc.log_prob(states)).sum())
    assert abs(total - 1.0) < 1e-5


def test_categorical_fit_tracks_frequencies():
    leaf = CategoricalLeaf(0, n_categories=2)
    X = torch.cat([torch.zeros(90, 1), torch.ones(10, 1)], dim=0)
    leaf.fit(X)
    p0 = float(torch.exp(leaf.log_density(torch.tensor(0.0))))
    assert 0.8 < p0 < 0.95


# ─── Validators 3 and 4 ───────────────────────────────────────────────────────

def test_determinism_holds_for_k1():
    pc = DensityPC(vtree4(), n_sum_components=1)
    validate_determinism(pc.root, torch.randn(10, 4))  # no sum nodes → trivially ok


def test_determinism_fails_for_gaussian_mixture():
    pc = DensityPC(vtree4(), n_sum_components=3, leaf_factory=GaussianLeaf)
    with pytest.raises(AssertionError, match="Determinism"):
        validate_determinism(pc.root, torch.randn(10, 4))


def test_structured_decomposability_passes_own_vtree():
    vt = vtree4()
    pc = DensityPC(vt, n_sum_components=3)
    validate_structured_decomposability(pc.root, vt)


def test_structured_decomposability_fails_other_vtree():
    pc = DensityPC(vtree4(), n_sum_components=2)
    other = VtreeInternal(  # interleaved partition (0,2)|(1,3) — incompatible
        left=VtreeInternal(VtreeLeaf(0), VtreeLeaf(2)),
        right=VtreeInternal(VtreeLeaf(1), VtreeLeaf(3)),
    )
    with pytest.raises(AssertionError, match="Structured decomposability"):
        validate_structured_decomposability(pc.root, other)


def test_individual_validators_pass_on_valid_circuit():
    pc = DensityPC(vtree4(), n_sum_components=2)
    validate_smoothness(pc.root)
    validate_decomposability(pc.root)
    pc.validate()


# ─── SOS / squared circuits ───────────────────────────────────────────────────

def test_sos_partition_finite_and_validates():
    pc = SquaredPC(vtree4(), n_sum_components=2, seed=3)
    assert math.isfinite(float(pc.log_partition()))
    pc.validate()


def test_sos_density_integrates_to_one_grid():
    pc = SquaredPC(vtree2(), n_sum_components=2, seed=1)
    pc.fit_leaves(torch.randn(100, 2))
    g = torch.linspace(-12, 12, 401)
    dx = float(g[1] - g[0])
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    with torch.no_grad():
        total = torch.exp(pc.log_prob(pts)).sum() * dx * dx
    assert abs(float(total) - 1.0) < 1e-3


def test_sos_marginal_matches_grid_integration():
    pc = SquaredPC(vtree2(), n_sum_components=2, seed=2)
    pc.fit_leaves(torch.randn(80, 2))
    exact = pc.log_marginal(torch.tensor([[0.5, 0.0]]), [1])
    g = torch.linspace(-12, 12, 4001)
    dx = float(g[1] - g[0])
    pts = torch.stack([torch.full((len(g),), 0.5), g], dim=1)
    with torch.no_grad():
        brute = torch.log(torch.exp(pc.log_prob(pts)).sum() * dx)
    assert abs(float(exact) - float(brute)) < 1e-3


def test_sos_marginal_none_equals_log_prob():
    pc = SquaredPC(vtree4(), n_sum_components=2, seed=4)
    x = torch.randn(6, 4)
    assert torch.allclose(pc.log_marginal(x, []), pc.log_prob(x), atol=1e-4)


def test_sos_trains_by_nll():
    """A few NLL steps must not break normalization."""
    pc = SquaredPC(vtree2(), n_sum_components=2, seed=5)
    X = torch.randn(128, 2)
    pc.fit_leaves(X)
    opt = torch.optim.Adam(pc.parameters(), lr=0.05)
    for _ in range(10):
        loss = -pc.log_prob(X).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = torch.linspace(-12, 12, 301)
    dx = float(g[1] - g[0])
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    with torch.no_grad():
        total = torch.exp(pc.log_prob(pts)).sum() * dx * dx
    assert abs(float(total) - 1.0) < 5e-3


def test_sos_gmm_leaves():
    pc = SquaredPC(vtree2(), n_sum_components=2,
                   leaf_factory=lambda i: GaussianMixtureLeaf(i, 2), seed=6)
    pc.fit_leaves(torch.randn(60, 2))
    out = pc.log_prob(torch.randn(8, 2))
    assert out.shape == (8,) and torch.isfinite(out).all()


def test_sos_categorical_leaves_sum_to_one():
    pc = SquaredPC(vtree2(), n_sum_components=2,
                   leaf_factory=lambda i: CategoricalLeaf(i, 2), seed=7)
    states = torch.tensor([[a, b] for a in range(2) for b in range(2)], dtype=torch.float32)
    total = float(torch.exp(pc.log_prob(states)).sum())
    assert abs(total - 1.0) < 1e-4


# ─── Module-level inference helpers on hand-built circuits ───────────────────

def test_log_partition_module_level():
    pc = DensityPC(random_balanced_vtree(list(range(8)), seed=0), n_sum_components=2)
    assert abs(float(log_partition(pc.root))) < 1e-5


def test_eval_log_marginal_module_level():
    pc = DensityPC(vtree2(), n_sum_components=2)
    x = torch.randn(4, 2)
    out = eval_log_marginal(pc.root, x, [0])
    assert out.shape == (4,) and torch.isfinite(out).all()
