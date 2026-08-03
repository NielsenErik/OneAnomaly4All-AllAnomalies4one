"""
Tests for the ProbMoE-style cardinality-constrained router (the SIMPLE
primitives in src/probabilistic_circuits.py) and the encoder-free
ProbRoutedRawPC detector (Direction 2 + ProbMoE) in src/directions.py.

Two things are pinned down:
  1. Exactness of the routing machinery — normalisers, cardinality posterior
     and selection marginals are checked against brute-force enumeration.
  2. The exactness INVARIANT of the project — adding the router does not touch
     the density: ProbRoutedRawPC.log_prob equals the plain RoutedRawPC mixture
     density, each sub-circuit stays normalised, and validate() passes.
"""
from __future__ import annotations

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.directions import ProbRoutedRawPC, RoutedRawPC, auroc
from src.probabilistic_circuits import (
    cardinality_log_normalizers,
    cardinality_log_posterior,
    cardinality_moments,
    selection_marginals,
)

torch.manual_seed(0)


def _brute(p_row: torch.Tensor):
    """Brute-force Z_k and per-k selection numerators by full enumeration."""
    n = len(p_row)
    Zk = torch.zeros(n + 1)
    num = torch.zeros(n + 1, n)
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            w = 1.0
            for i in range(n):
                w *= float(p_row[i]) if i in S else float(1 - p_row[i])
            Zk[r] += w
            for j in S:
                num[r, j] += w
    return Zk, num


# ─── SIMPLE primitives: exactness vs brute force ────────────────────────────

def test_normalizers_match_brute_force_and_sum_to_one():
    logits = torch.randn(5, 6) * 1.5
    p = torch.sigmoid(logits)
    logZ = cardinality_log_normalizers(torch.log(p), torch.log(1 - p))
    # Σ_k Z_k == 1
    assert torch.allclose(logZ.exp().sum(dim=1), torch.ones(5), atol=1e-5)
    for b in range(5):
        Zk, _ = _brute(p[b])
        assert torch.allclose(logZ[b].exp(), Zk, atol=1e-5)


def test_selection_marginals_match_brute_force():
    logits = torch.randn(4, 6) * 1.5
    p = torch.sigmoid(logits)
    for b in range(4):
        Zk, num = _brute(p[b])
        for kmin, kmax in [(1, 6), (1, 1), (2, 2), (3, 3), (2, 4)]:
            m = selection_marginals(logits[b:b + 1], kmin, kmax)[0]
            m_bf = num[kmin:kmax + 1].sum(0) / Zk[kmin:kmax + 1].sum()
            assert torch.allclose(m, m_bf, atol=1e-4)


def test_exact_k_marginals_sum_to_k_and_are_finite():
    logits = torch.randn(8, 5) * 2.0
    for k in range(1, 5):
        m = selection_marginals(logits, k, k)
        assert torch.isfinite(m).all()
        assert torch.allclose(m.sum(dim=1), torch.full((8,), float(k)), atol=1e-4)


def test_cardinality_posterior_and_moments():
    logits = torch.randn(7, 5)
    post = cardinality_log_posterior(logits, 1, 5).exp()
    assert torch.allclose(post.sum(dim=1), torch.ones(7), atol=1e-5)
    mom = cardinality_moments(logits, 1, 5)
    assert mom["expected"].shape == (7,)
    assert (mom["expected"] >= 1).all() and (mom["expected"] <= 5).all()
    assert (mom["entropy"] >= -1e-5).all()
    assert (mom["map"] >= 1).all() and (mom["map"] <= 5).all()


def test_marginals_work_under_no_grad():
    """Routing analysis is typically called at eval time inside no_grad."""
    logits = torch.randn(3, 4)
    with torch.no_grad():
        m = selection_marginals(logits, 1, 4)
    assert torch.isfinite(m).all() and m.shape == (3, 4)


# ─── ProbRoutedRawPC detector ───────────────────────────────────────────────

def _fit_detector(d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = {f"m{i}": torch.randn(150, d, generator=g) + (i * 3.0) for i in range(3)}
    det = ProbRoutedRawPC({m: d for m in data}, n_sum_components=3,
                          shared_structure=True, seed=seed)
    det.fit(data, epochs=15, lr=0.1)
    return det, data


def test_density_invariant_preserved():
    """The router must not alter the density: per-expert partition stays ~0 and
    log_prob equals an identically-built RoutedRawPC mixture."""
    det, data = _fit_detector()
    det.validate()
    for m in det.pcs:
        assert abs(float(det.pcs[m].log_partition().detach())) < 1e-3
    x = data["m1"][:10]
    # same weights/experts ⇒ unknown-modality mixture density is identical
    lp_prob = det.log_prob(x)
    plain = RoutedRawPC.log_prob(det, x)  # inherited mixture path, unchanged
    assert torch.allclose(lp_prob, plain, atol=1e-6)
    assert torch.isfinite(det.log_prob(x, modality="m1")).all()


def test_routing_queries_shapes_and_ranges():
    det, data = _fit_detector()
    x = data["m0"][:12]
    n = len(det.pcs)
    post = det.cardinality_log_posterior(x)
    assert post.shape == (12, n)  # default range [1, N]
    assert torch.allclose(post.exp().sum(1), torch.ones(12), atol=1e-5)
    m, names = det.selection_marginals(x)
    assert m.shape == (12, n) and names == list(det.pcs.keys())
    resp, _ = det.mixture_responsibilities(x)
    assert torch.allclose(resp.sum(1), torch.ones(12), atol=1e-5)


def test_localization_recovers_source_modality():
    det, data = _fit_detector()
    import numpy as np
    for target in ["m0", "m1", "m2"]:
        loc, names = det.localize(data[target][:40])
        acc = float((np.array(names)[loc.numpy()] == target).mean())
        assert acc > 0.8  # well-separated synthetic ⇒ near-perfect routing


def test_router_params_are_trainable():
    det, data = _fit_detector()
    ent = det.routing_entropy(data["m1"][:20]).mean()
    ent.backward()
    assert det.router_log_temp.grad is not None
    assert det.router_bias["m0"].grad is not None


def test_exact_k_configuration():
    g = torch.Generator().manual_seed(1)
    data = {f"m{i}": torch.randn(120, 5, generator=g) + (i * 4.0) for i in range(3)}
    det = ProbRoutedRawPC({m: 5 for m in data}, k_min=2, k_max=2, seed=1)
    det.fit(data, epochs=10, lr=0.1)
    m, _ = det.selection_marginals(data["m0"][:8])
    assert torch.allclose(m.sum(1), torch.full((8,), 2.0), atol=1e-4)


def test_add_modality_grows_router():
    det, data = _fit_detector()
    g = torch.Generator().manual_seed(2)
    det.add_modality("m3", torch.randn(80, 6, generator=g) - 5.0, epochs=8, lr=0.1)
    assert "m3" in det.router_bias and "m3" in det._m_bar
    m, names = det.selection_marginals(data["m0"][:5])
    assert m.shape == (5, 4) and "m3" in names
    det.validate()


def test_routing_scores_run():
    det, data = _fit_detector()
    clean = data["m2"][:30]
    for sig in ["shift", "entropy", "expected_k", "neg_max_marginal"]:
        s = det.routing_score(clean, signal=sig)
        assert s.shape == (30,) and torch.isfinite(s).all()
