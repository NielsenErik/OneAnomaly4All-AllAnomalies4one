"""
Tests for the learned (Chow-Liu / HCLT-style) vtree.

The expressiveness claim is tested directly: on data with block-dependence
structure, the learned vtree must (i) keep dependent features deeper together
than independent ones, and (ii) give a better fit (lower NLL) than a random
vtree with identical capacity.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.directions import LatentPCDetector, RoutedRawPC, default_mlp_encoder, fit_pc_nll
from src.experiment import MultimodalPipeline
from src.probabilistic_circuits import (
    DensityPC,
    GaussianLeaf,
    chow_liu_vtree,
    lca_depth_matrix,
    learned_vtree,
    mutual_information_matrix,
    random_balanced_vtree,
    validate_circuit,
    vtree_leaves,
)
from tests.test_datasets_pipeline import synthetic_tabular

torch.manual_seed(0)


def block_data(n=600, seed=0):
    """6 features: {0,1,2} strongly coupled, {3,4,5} strongly coupled,
    blocks independent of each other."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 1, generator=g)
    b = torch.randn(n, 1, generator=g)
    eps = 0.3 * torch.randn(n, 6, generator=g)
    X = torch.cat([a, a, a, b, b, b], dim=1) + eps
    return X


# ─── Mutual information matrix ────────────────────────────────────────────────

def test_mi_matrix_recovers_dependence():
    X = block_data()
    M = mutual_information_matrix(X)
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T)
    assert M[0, 1] > M[0, 3], "within-block MI must exceed cross-block MI"
    assert M[3, 4] > M[2, 5]


def test_mi_matrix_discrete():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, 500)
    X = np.stack([z, z ^ (rng.random(500) < 0.05), rng.integers(0, 2, 500)], axis=1)
    M = mutual_information_matrix(X, kind="discrete")
    assert M[0, 1] > M[0, 2]


# ─── Chow-Liu vtree structure ─────────────────────────────────────────────────

def test_chow_liu_vtree_covers_all_features():
    X = block_data()
    vt = chow_liu_vtree(X)
    assert sorted(vtree_leaves(vt)) == list(range(6))


def test_chow_liu_vtree_single_feature():
    vt = chow_liu_vtree(torch.randn(50, 1))
    assert vtree_leaves(vt) == [0]


def test_chow_liu_vtree_groups_dependent_features():
    """Dependent features must share deeper LCAs than independent ones —
    the structural signature of the HCLT idea."""
    X = block_data()
    vt = chow_liu_vtree(X)
    L = lca_depth_matrix(vt)
    within = np.mean([L[(i, j)] for i in [0, 1, 2] for j in [0, 1, 2] if i != j]
                     + [L[(i, j)] for i in [3, 4, 5] for j in [3, 4, 5] if i != j])
    cross = np.mean([L[(i, j)] for i in [0, 1, 2] for j in [3, 4, 5]])
    assert within > cross, f"within-block LCA depth {within:.2f} ≤ cross {cross:.2f}"


def test_chow_liu_pc_validates():
    X = block_data()
    pc = DensityPC(chow_liu_vtree(X), n_sum_components=3, leaf_factory=GaussianLeaf)
    pc.validate()


def test_chow_liu_beats_random_vtree_on_structured_data():
    """Same capacity, same training: the learned structure must fit the
    block-dependent data better (lower held-out NLL)."""
    X = block_data(n=800, seed=1)
    X_train, X_val = X[:600], X[600:]

    nlls = {}
    for name, vt in [("chow_liu", chow_liu_vtree(X_train)),
                     ("random", random_balanced_vtree(list(range(6)), seed=0))]:
        pc = DensityPC(vt, n_sum_components=4, leaf_factory=GaussianLeaf)
        pc.fit_leaves(X_train)
        fit_pc_nll(pc, X_train, epochs=80, lr=0.05)
        with torch.no_grad():
            nlls[name] = float(-pc.log_prob(X_val).mean())

    assert nlls["chow_liu"] < nlls["random"], (
        f"learned vtree NLL {nlls['chow_liu']:.3f} not better than "
        f"random {nlls['random']:.3f}"
    )


def test_learned_vtree_dispatcher():
    X = block_data()
    assert sorted(vtree_leaves(learned_vtree(X, "chow_liu"))) == list(range(6))
    assert sorted(vtree_leaves(learned_vtree(X, "random"))) == list(range(6))
    with pytest.raises(KeyError, match="Unknown vtree method"):
        learned_vtree(X, "magic")


# ─── Integration: directions ─────────────────────────────────────────────────

def test_latent_detector_chow_liu():
    d, latent = 12, 6
    g = torch.Generator().manual_seed(0)
    X_in = torch.randn(300, d, generator=g)
    X_out = torch.randn(100, d, generator=g) * 0.5 + 5.0
    det = LatentPCDetector(default_mlp_encoder(d, latent, seed=0), latent_dim=latent,
                           n_sum_components=2, leaf_factory=GaussianLeaf,
                           vtree_method="chow_liu")
    assert det.pc is None                       # structure not built yet
    with pytest.raises(RuntimeError, match="call fit"):
        det.score(X_in)
    det.fit(X_in, epochs=40)
    det.validate()
    with torch.no_grad():
        assert float(det.score(X_out).mean()) > float(det.score(X_in).mean())


def test_routed_pc_chow_liu_shared_structure():
    d = 6
    Xa = block_data(seed=2)
    Xb = block_data(seed=3) + 1.0
    det = RoutedRawPC({"a": d, "b": d}, n_sum_components=2,
                      leaf_factory=GaussianLeaf, vtree_method="chow_liu")
    assert len(det.pcs) == 0                    # lazy until fit
    det.fit({"a": Xa, "b": Xb}, epochs=30)
    det.validate()
    # learned once from pooled data, shared across modalities
    assert det.pcs["a"].vtree is det.pcs["b"].vtree
    L = lca_depth_matrix(det.pcs["a"].vtree)
    assert L[(0, 1)] >= L[(0, 3)]


def test_routed_pc_add_modality_chow_liu_new_dim():
    det = RoutedRawPC({"a": 6}, n_sum_components=2,
                      leaf_factory=GaussianLeaf, vtree_method="chow_liu")
    det.fit({"a": block_data(seed=4)}, epochs=20)
    # new modality with a different dimension → fresh learned structure
    X_new = torch.randn(300, 4)
    X_new[:, 1] = X_new[:, 0] + 0.1 * torch.randn(300)
    det.add_modality("b", X_new, epochs=20)
    L = lca_depth_matrix(det.pcs["b"].vtree)
    assert L[(0, 1)] >= L[(0, 3)]


# ─── Integration: pipeline ────────────────────────────────────────────────────

def test_pipeline_chow_liu_default():
    src = synthetic_tabular(d=10, seed=20, name="src")
    new = synthetic_tabular(d=14, seed=21, name="new")
    pipe = MultimodalPipeline(latent_dim=8, n_sum_components=2,
                              leaf_factory=GaussianLeaf)   # chow_liu is the default
    assert pipe.pc is None
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.evaluate(src, "zero_shot")
    pipe.fit([src], epochs=30)
    pipe.validate()
    assert pipe.evaluate(new, "zero_shot")["auroc"] > 0.9


def test_pipeline_random_vtree_still_available():
    src = synthetic_tabular(d=8, seed=22, name="src")
    pipe = MultimodalPipeline(latent_dim=6, n_sum_components=2,
                              leaf_factory=GaussianLeaf, vtree_method="random")
    assert pipe.pc is not None                  # built eagerly
    pipe.fit([src], epochs=20)
    assert pipe.evaluate(src, "zero_shot")["auroc"] > 0.8
