"""
Tests for the two directions (src/directions.py) and the shared
anomaly-detection interface, including a side-by-side comparison on the same
synthetic multimodal task.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.directions import (
    LatentPCDetector,
    RoutedRawPC,
    auroc,
    build_consensus_routed_pc,
    default_mlp_encoder,
    evaluate_detector,
    fit_pc_contrastive,
    fit_pc_nll,
    fit_pc_unsupervised,
    nll_contrastive_loss,
)
from src.probabilistic_circuits import DensityPC, GaussianLeaf, random_balanced_vtree

torch.manual_seed(0)


def make_data(d=8, n=200, shift=6.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    X_in = torch.randn(n, d, generator=g)
    X_out = torch.randn(n, d, generator=g) * 0.5 + shift
    return X_in, X_out


# ─── auroc ────────────────────────────────────────────────────────────────────

def test_auroc_perfect_and_chance():
    assert auroc(torch.zeros(50), torch.ones(50)) == 1.0
    assert abs(auroc(torch.arange(100.0), torch.arange(100.0)) - 0.5) < 1e-9


# ─── fit_pc_nll ───────────────────────────────────────────────────────────────

def test_fit_pc_nll_decreases_loss():
    X = torch.randn(256, 4) * 2.0 + 1.0
    pc = DensityPC(random_balanced_vtree(list(range(4)), seed=0),
                   n_sum_components=2, leaf_factory=GaussianLeaf)
    pc.fit_leaves(X)
    hist = fit_pc_nll(pc, X, epochs=30, lr=0.05)
    assert hist[-1] < hist[0]


# ─── Training loops: unsupervised (primary) and supervised contrastive ───────

def test_unsupervised_alias_is_nll_loop():
    assert fit_pc_unsupervised is fit_pc_nll


def test_contrastive_loss_zero_when_separated():
    """Margin term vanishes once negatives sit ≥ margin below positives."""
    lp_pos = torch.full((8,), -1.0)
    lp_neg = torch.full((8,), -10.0)
    loss = nll_contrastive_loss(lp_pos, lp_neg, alpha=0.0, margin=1.0)
    assert float(loss) == 0.0


def test_contrastive_loss_penalizes_overlap():
    lp = torch.zeros(8)
    loss = nll_contrastive_loss(lp, lp, alpha=0.0, margin=2.0)
    assert abs(float(loss) - 2.0) < 1e-6


def test_fit_pc_contrastive_separates_densities():
    """After Eq.-4 training, negatives must score below positives by ~margin."""
    X_pos = torch.randn(200, 4)
    X_neg = torch.randn(200, 4) * 0.5 + 3.0  # close enough to overlap initially
    pc = DensityPC(random_balanced_vtree(list(range(4)), seed=0),
                   n_sum_components=2, leaf_factory=GaussianLeaf)
    pc.fit_leaves(X_pos)
    hist = fit_pc_contrastive(pc, X_pos, X_neg, epochs=60, lr=0.05,
                              alpha=0.5, margin=1.0)
    assert hist[-1] < hist[0]
    with torch.no_grad():
        gap = float(pc.log_prob(X_pos).mean() - pc.log_prob(X_neg).mean())
    assert gap > 1.0
    assert auroc(-pc.log_prob(X_pos), -pc.log_prob(X_neg)) > 0.95


def test_contrastive_beats_unsupervised_on_near_anomalies():
    """
    With anomalies overlapping the normal manifold, the supervised margin
    should separate them at least as well as NLL-only training.
    """
    g = torch.Generator().manual_seed(0)
    X_pos = torch.randn(300, 4, generator=g)
    X_neg = torch.randn(300, 4, generator=g) + 1.5  # heavy overlap
    vt = random_balanced_vtree(list(range(4)), seed=1)

    pc_u = DensityPC(vt, n_sum_components=2, leaf_factory=GaussianLeaf)
    pc_u.fit_leaves(X_pos)
    fit_pc_unsupervised(pc_u, X_pos, epochs=60, lr=0.05)

    pc_c = DensityPC(vt, n_sum_components=2, leaf_factory=GaussianLeaf)
    pc_c.fit_leaves(X_pos)
    fit_pc_contrastive(pc_c, X_pos, X_neg, epochs=60, lr=0.05, alpha=0.5, margin=1.0)

    with torch.no_grad():
        a_u = auroc(-pc_u.log_prob(X_pos), -pc_u.log_prob(X_neg))
        a_c = auroc(-pc_c.log_prob(X_pos), -pc_c.log_prob(X_neg))
    assert a_c >= a_u - 0.02, f"contrastive {a_c:.3f} vs unsupervised {a_u:.3f}"


def test_latent_detector_fit_contrastive_end_to_end():
    """θ and fϕ trained jointly (paper setting): encoder weights must move."""
    d, latent = 12, 4
    X_pos, X_neg = make_data(d=d, shift=2.0, seed=11)
    det = LatentPCDetector(default_mlp_encoder(d, latent, seed=7), latent_dim=latent,
                           n_sum_components=2, leaf_factory=GaussianLeaf)
    before = [p.clone() for p in det.encoders["default"].parameters()]
    det.fit_contrastive(X_pos, X_neg, epochs=40, lr=0.02, alpha=0.5, margin=1.0)
    moved = any(not torch.allclose(a, b) for a, b in
                zip(before, det.encoders["default"].parameters()))
    assert moved
    assert evaluate_detector(det, X_pos, X_neg) > 0.9


def test_latent_detector_fit_contrastive_missing_negatives_raises():
    det = LatentPCDetector(default_mlp_encoder(4, 2, seed=0), latent_dim=2)
    with pytest.raises(KeyError, match="Negatives missing"):
        det.fit_contrastive({"a": torch.randn(10, 4)}, {})


def test_routed_pc_fit_contrastive_with_partial_negatives():
    """Modality 'a' gets the supervised objective, 'b' falls back to NLL."""
    d = 6
    Xa_pos, Xa_neg = make_data(d=d, shift=2.5, seed=12)
    Xb_pos, Xb_out = make_data(d=d, shift=-5.0, seed=13)
    det = RoutedRawPC({"a": d, "b": d}, n_sum_components=2, leaf_factory=GaussianLeaf)
    det.fit_contrastive({"a": Xa_pos, "b": Xb_pos}, {"a": Xa_neg},
                        epochs=50, lr=0.05, margin=1.0)
    det.validate()
    assert evaluate_detector(det, Xa_pos, Xa_neg, modality="a") > 0.9
    assert evaluate_detector(det, Xb_pos, Xb_out, modality="b") > 0.9


# ─── Direction 1: LatentPCDetector ────────────────────────────────────────────

def test_latent_detector_single_modality():
    d, latent = 16, 8
    X_in, X_out = make_data(d=d, seed=1)
    det = LatentPCDetector(default_mlp_encoder(d, latent, seed=0), latent_dim=latent,
                           n_sum_components=2, leaf_factory=GaussianLeaf)
    det.fit(X_in, epochs=60, lr=0.05)
    det.validate()
    assert evaluate_detector(det, X_in, X_out) > 0.9


def test_latent_detector_multimodal_shared_pc():
    """Two modalities of different raw dim share one latent PC."""
    latent = 6
    Xa_in, Xa_out = make_data(d=12, seed=2)
    Xb_in, Xb_out = make_data(d=20, seed=3)
    det = LatentPCDetector(
        {"tabular": default_mlp_encoder(12, latent, seed=1),
         "signal": default_mlp_encoder(20, latent, seed=2)},
        latent_dim=latent, n_sum_components=2, leaf_factory=GaussianLeaf,
    )
    det.fit({"tabular": Xa_in, "signal": Xb_in}, epochs=60, lr=0.05)
    assert evaluate_detector(det, Xa_in, Xa_out, modality="tabular") > 0.8
    assert evaluate_detector(det, Xb_in, Xb_out, modality="signal") > 0.8


def test_latent_detector_sos_mode():
    d, latent = 10, 4
    X_in, X_out = make_data(d=d, seed=4)
    det = LatentPCDetector(default_mlp_encoder(d, latent, seed=3), latent_dim=latent,
                           n_sum_components=2, use_sos=True)
    det.fit(X_in, epochs=40, lr=0.03)
    det.validate()
    assert evaluate_detector(det, X_in, X_out) > 0.8


# ─── Direction 2: RoutedRawPC ─────────────────────────────────────────────────

def test_routed_pc_separates_ood():
    d = 8
    Xa_in, Xa_out = make_data(d=d, seed=5)
    Xb_in, Xb_out = make_data(d=d, shift=-6.0, seed=6)
    det = RoutedRawPC({"a": d, "b": d}, n_sum_components=2, leaf_factory=GaussianLeaf)
    det.fit({"a": Xa_in, "b": Xb_in}, epochs=60, lr=0.05)
    det.validate()
    assert evaluate_detector(det, Xa_in, Xa_out, modality="a") > 0.9
    assert evaluate_detector(det, Xb_in, Xb_out, modality="b") > 0.9


def test_routed_pc_shared_structure_object():
    """With equal dims, all sub-circuits must share one vtree object."""
    det = RoutedRawPC({"a": 8, "b": 8}, shared_structure=True)
    assert det.pcs["a"].vtree is det.pcs["b"].vtree


def test_routed_pc_unknown_modality_mixture():
    d = 6
    Xa_in, Xa_out = make_data(d=d, seed=7)
    det = RoutedRawPC({"a": d, "b": d}, n_sum_components=2, leaf_factory=GaussianLeaf)
    det.fit({"a": Xa_in, "b": Xa_in + 1.0}, epochs=40, lr=0.05)
    # scoring without a modality must still separate in-dist from OOD
    assert auroc(det.score(Xa_in), det.score(Xa_out)) > 0.9


def test_routed_pc_dimension_mismatch_raises():
    det = RoutedRawPC({"a": 4, "b": 4})
    with pytest.raises(ValueError, match="No sub-circuit"):
        det.log_prob(torch.randn(3, 9))


def test_routed_pc_different_dims_no_shared_structure():
    det = RoutedRawPC({"a": 4, "b": 10})
    det.validate()
    assert det.pcs["a"].vtree is not det.pcs["b"].vtree


# ─── Consensus-structure pipeline ────────────────────────────────────────────

def test_build_consensus_routed_pc():
    d = 6
    Xa_in, Xa_out = make_data(d=d, seed=8)
    Xb_in, _ = make_data(d=d, shift=-4.0, seed=9)
    det = build_consensus_routed_pc({"a": Xa_in, "b": Xb_in},
                                    n_sum_components=2,
                                    leaf_factory=GaussianLeaf,
                                    pretrain_epochs=10, epochs=40)
    det.validate()
    # consensus structure is shared across modalities
    assert det.pcs["a"].vtree is det.pcs["b"].vtree
    assert evaluate_detector(det, Xa_in, Xa_out, modality="a") > 0.9


def test_consensus_requires_aligned_dims():
    with pytest.raises(ValueError, match="aligned"):
        build_consensus_routed_pc({"a": torch.randn(20, 4), "b": torch.randn(20, 6)})


# ─── Side-by-side comparison (the whole point of the shared interface) ───────

def test_both_directions_comparable_on_same_task():
    d = 8
    X_in, X_out = make_data(d=d, seed=10)
    d1 = LatentPCDetector(default_mlp_encoder(d, 4, seed=5), latent_dim=4,
                          n_sum_components=2, leaf_factory=GaussianLeaf)
    d1.fit(X_in, epochs=40, lr=0.05)
    d2 = RoutedRawPC({"default": d}, n_sum_components=2, leaf_factory=GaussianLeaf)
    d2.fit({"default": X_in}, epochs=40, lr=0.05)
    a1 = evaluate_detector(d1, X_in, X_out)
    a2 = evaluate_detector(d2, X_in, X_out, modality="default")
    assert a1 > 0.7 and a2 > 0.7
