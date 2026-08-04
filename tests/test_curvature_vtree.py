"""
Tests for the curvature-guided vtree learners (Idea A, 2026-06-11 eval).

The mechanistic claim is tested directly on block-structured data: the
cross-block edges of the sparsified MI graph must be the LEAST curved
(bottlenecks), every learner must recover the block split at the top of the
vtree, and — since any vtree is valid by construction — the resulting
circuits must pass all four property validators unchanged.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probabilistic_circuits import (
    VTREE_METHODS,
    DensityPC,
    curvature_sign_stability,
    curvature_vtree,
    forman_curvature,
    learned_vtree,
    mutual_information_matrix,
    ollivier_ricci_curvature,
    sparsify_mi_graph,
    spectral_vtree,
    validate_circuit,
    vtree_leaves,
)
from tests.test_chow_liu import block_data

torch.manual_seed(0)

BLOCK_A, BLOCK_B = {0, 1, 2}, {3, 4, 5}


def block_graph():
    M = mutual_information_matrix(block_data())
    return sparsify_mi_graph(M)


# ─── sparsification ──────────────────────────────────────────────────────────

def test_sparsify_is_symmetric_connected_and_sparse():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    A = block_graph()
    assert np.allclose(A, A.T)
    assert np.all(np.diag(A) == 0)
    n_comp, _ = connected_components(csr_matrix(A), directed=False)
    assert n_comp == 1, "MST union must keep the graph connected"
    assert (A > 0).sum() < A.size  # strictly sparser than the dense MI matrix


# ─── curvature: bottleneck edges are the least curved ────────────────────────

def cross_within(kappa):
    cross = [v for (i, j), v in kappa.items() if (i in BLOCK_A) != (j in BLOCK_A)]
    within = [v for (i, j), v in kappa.items() if (i in BLOCK_A) == (j in BLOCK_A)]
    return cross, within


def test_orc_marks_cross_block_bottlenecks():
    kappa = ollivier_ricci_curvature(block_graph())
    cross, within = cross_within(kappa)
    assert cross, "sparsified graph must keep at least one bridge (MST union)"
    assert max(cross) < min(within), \
        "every cross-block edge must be less curved than every within-block edge"


def test_forman_marks_cross_block_bottlenecks():
    kappa = forman_curvature(block_graph())
    cross, within = cross_within(kappa)
    assert np.mean(cross) < np.mean(within)


def test_curvature_sign_stability_bounds():
    stab = curvature_sign_stability(block_data(n=300), n_boot=5)
    assert stab and all(0.0 <= v <= 1.0 for v in stab.values())


# ─── vtree learners recover the block split ──────────────────────────────────

@pytest.mark.parametrize("method", ["orc", "forman", "spectral"])
def test_learner_recovers_block_split(method):
    vt = learned_vtree(block_data(), method=method)
    assert sorted(vtree_leaves(vt)) == list(range(6))
    top = {frozenset(vt.left.scope), frozenset(vt.right.scope)}
    assert top == {frozenset(BLOCK_A), frozenset(BLOCK_B)}


def test_ricci_flow_variant_also_recovers_split():
    vt = curvature_vtree(block_data(), curvature="ollivier", flow_iters=2)
    top = {frozenset(vt.left.scope), frozenset(vt.right.scope)}
    assert top == {frozenset(BLOCK_A), frozenset(BLOCK_B)}


def test_single_and_two_feature_edge_cases():
    X = block_data()[:, :2]
    vt = curvature_vtree(X)
    assert sorted(vtree_leaves(vt)) == [0, 1]
    vt = spectral_vtree(block_data()[:, :1])
    assert vtree_leaves(vt) == [0]


def test_unknown_method_raises():
    with pytest.raises(KeyError, match="Unknown vtree method"):
        learned_vtree(block_data(), method="hyperbolic")
    assert set(VTREE_METHODS) == {"chow_liu", "spectral", "orc", "forman", "random"}


# ─── the four properties hold for every learned structure ────────────────────

@pytest.mark.parametrize("method", ["orc", "forman", "spectral"])
def test_circuit_on_learned_vtree_is_valid(method):
    X = block_data(n=300)
    pc = DensityPC(learned_vtree(X, method=method), n_sum_components=2)
    validate_circuit(pc.root)  # smoothness + decomposability + structured dec.
    pc.fit_leaves(X)
    lp = pc.log_prob(X[:16])
    assert lp.shape == (16,) and torch.isfinite(lp).all()


# ─── matched-budget structure quality (the point of the exercise) ────────────

def test_orc_beats_random_on_block_data():
    """At identical capacity, the curvature structure must fit block data at
    least as well as a random vtree (the control arm of the ablation)."""
    from src.directions import fit_pc_nll

    X = block_data(n=800, seed=1)
    tr, te = X[:600], X[600:]
    nll = {}
    for method in ("orc", "random"):
        torch.manual_seed(0)
        pc = DensityPC(learned_vtree(tr, method=method, seed=0),
                       n_sum_components=3)
        pc.fit_leaves(tr)
        fit_pc_nll(pc, tr, epochs=60, lr=0.05)
        with torch.no_grad():
            nll[method] = float(-pc.log_prob(te).mean())
    assert nll["orc"] < nll["random"], \
        f"orc {nll['orc']:.3f} should beat random {nll['random']:.3f}"
