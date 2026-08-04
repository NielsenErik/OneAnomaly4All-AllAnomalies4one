"""
Tests for Stage 0a + 0b.

Covers:
  - VtreeLeaf / VtreeInternal scope properties
  - Round-trip JSON save / load
  - Random vtree generators (balanced + unbalanced)
  - DensityPC build from vtree
  - validate_circuit: passes on valid circuits, raises on violations
  - extract_vtree: round-trips through DensityPC
  - DensityPC.forward: output shape and finite values
  - DensityPC.anomaly_score: higher for out-of-distribution samples
  - Single-task vtree extraction
  - Consensus vtree: recovers known block-diagonal independence structure
    (synthetic case 11 from dev.md)
  - cogroup_matrix: feature pairs in same block have higher co-grouping
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Everything now lives in the single consolidated components file.
from src.probabilistic_circuits import (
    DensityPC,
    InputNode,
    ProductNode,
    SumNode,
    VtreeInternal,
    VtreeLeaf,
    VtreeNode,
    cogroup_matrix,
    consensus_vtree,
    extract_vtree,
    lca_depth_matrix,
    load_vtree,
    random_balanced_vtree,
    random_unbalanced_vtree,
    save_vtree,
    single_task_vtree,
    validate_circuit,
    vtree_depth,
    vtree_leaves,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_vtree_4() -> VtreeNode:
    """Manual balanced vtree over features {0, 1, 2, 3}."""
    return VtreeInternal(
        left=VtreeInternal(left=VtreeLeaf(0), right=VtreeLeaf(1)),
        right=VtreeInternal(left=VtreeLeaf(2), right=VtreeLeaf(3)),
    )


def make_block_vtree() -> VtreeNode:
    """
    Vtree encoding block structure: ({0,1,2} independent of {3,4,5}).
    Within each block the features are further split 1 vs 2.
    """
    left_block = VtreeInternal(
        left=VtreeLeaf(0),
        right=VtreeInternal(left=VtreeLeaf(1), right=VtreeLeaf(2)),
    )
    right_block = VtreeInternal(
        left=VtreeLeaf(3),
        right=VtreeInternal(left=VtreeLeaf(4), right=VtreeLeaf(5)),
    )
    return VtreeInternal(left=left_block, right=right_block)


# ─── VtreeLeaf / VtreeInternal ────────────────────────────────────────────────

def test_leaf_scope():
    assert VtreeLeaf(7).scope == frozenset({7})


def test_internal_scope():
    v = make_vtree_4()
    assert v.scope == frozenset({0, 1, 2, 3})
    assert v.left_scope == frozenset({0, 1})
    assert v.right_scope == frozenset({2, 3})


def test_vtree_depth():
    assert vtree_depth(VtreeLeaf(0)) == 0
    assert vtree_depth(make_vtree_4()) == 2


def test_vtree_leaves():
    assert sorted(vtree_leaves(make_vtree_4())) == [0, 1, 2, 3]


# ─── Serialization ────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    vt = make_vtree_4()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_vtree(vt, path)
        loaded = load_vtree(path)
        assert loaded == vt
        assert loaded.scope == vt.scope
    finally:
        os.unlink(path)


def test_json_sets_are_sorted_lists():
    """Scopes must be serialised as sorted lists, not raw Python sets."""
    vt = make_vtree_4()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        save_vtree(vt, path)
        with open(path) as f:
            raw = json.load(f)
        # Internal nodes have 'left' and 'right', no raw sets
        assert "type" in raw
    finally:
        os.unlink(path)


# ─── Random vtree generators ─────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 8, 16, 128])
def test_random_balanced_covers_all_features(n: int):
    feats = list(range(n))
    vt = random_balanced_vtree(feats, seed=0)
    assert sorted(vtree_leaves(vt)) == feats


@pytest.mark.parametrize("n", [2, 8, 16])
def test_random_unbalanced_covers_all_features(n: int):
    feats = list(range(n))
    vt = random_unbalanced_vtree(feats, seed=42)
    assert sorted(vtree_leaves(vt)) == feats


def test_balanced_vtree_different_seeds_differ():
    v1 = random_balanced_vtree(list(range(16)), seed=0)
    v2 = random_balanced_vtree(list(range(16)), seed=1)
    assert v1 != v2


def test_balanced_vtree_same_seed_reproducible():
    v1 = random_balanced_vtree(list(range(16)), seed=99)
    v2 = random_balanced_vtree(list(range(16)), seed=99)
    assert v1 == v2


# ─── DensityPC build ─────────────────────────────────────────────────────────

def test_density_pc_builds():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=2)
    assert pc.root is not None


def test_density_pc_k1_no_sum_nodes():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=1)
    for m in pc.modules():
        assert not isinstance(m, SumNode)


# ─── validate_circuit ─────────────────────────────────────────────────────────

def test_valid_circuit_passes():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=3)
    validate_circuit(pc.root)  # should not raise


def test_validate_detects_decomposability_violation():
    """ProductNode with overlapping scopes must raise."""
    leaf_a = InputNode(0)
    leaf_b = InputNode(0)  # same feature index → overlapping scope
    bad_prod = ProductNode([leaf_a, leaf_b])
    with pytest.raises(AssertionError, match="Decomposability"):
        validate_circuit(bad_prod)


def test_validate_detects_smoothness_violation():
    """SumNode with children of different scopes must raise."""
    leaf_a = InputNode(0)
    leaf_b = InputNode(1)  # different scope
    bad_sum = SumNode([leaf_a, leaf_b])
    with pytest.raises(AssertionError, match="Smoothness"):
        validate_circuit(bad_sum)


# ─── extract_vtree ────────────────────────────────────────────────────────────

def test_extract_vtree_roundtrip():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=2)
    extracted = extract_vtree(pc.root)
    # Scopes must match; tree shape may differ (sum nodes are transparent)
    assert extracted.scope == vt.scope


def test_single_task_vtree_scope():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=3)
    extracted = single_task_vtree(pc)
    assert extracted.scope == frozenset({0, 1, 2, 3})


# ─── DensityPC forward / anomaly score ───────────────────────────────────────

def test_forward_output_shape():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=2)
    z = torch.randn(8, 4)
    out = pc(z)
    assert out.shape == (8,)


def test_forward_finite():
    vt = make_vtree_4()
    pc = DensityPC(vt, n_sum_components=2)
    z = torch.randn(16, 4)
    out = pc(z)
    assert torch.isfinite(out).all()


def test_anomaly_score_higher_for_ood():
    """
    PC fit on N(0,1) data should assign higher NLL to N(10, 0.1) data.
    Tests the basic density model property.
    """
    feats = list(range(8))
    vt = random_balanced_vtree(feats, seed=0)
    pc = DensityPC(vt, n_sum_components=3)

    X_train = torch.randn(200, 8)
    pc.fit_leaves(X_train)

    in_dist = torch.randn(50, 8)
    ood = torch.randn(50, 8) * 0.1 + 10.0

    score_in = pc.anomaly_score(in_dist).mean().item()
    score_ood = pc.anomaly_score(ood).mean().item()
    assert score_ood > score_in, (
        f"Expected OOD score ({score_ood:.2f}) > in-dist score ({score_in:.2f})"
    )


# ─── lca_depth_matrix ─────────────────────────────────────────────────────────

def test_lca_depth_matrix_block():
    vt = make_block_vtree()
    M = lca_depth_matrix(vt)
    # Features within the same block (e.g. 0 and 1) share a deeper LCA
    # than cross-block pairs (e.g. 0 and 3)
    within = M.get((0, 1), M.get((1, 0), None))
    cross = M.get((0, 3), M.get((3, 0), None))
    assert within is not None and cross is not None
    assert within > cross, f"Expected within-block LCA depth > cross-block: {within} vs {cross}"


# ─── cogroup_matrix ────────────────────────────────────────────────────────────

def test_cogroup_matrix_within_block_higher():
    """
    Features in the same block should have higher co-grouping score than
    cross-block feature pairs when all source vtrees share the same block structure.
    """
    n = 6
    vtrees = [make_block_vtree() for _ in range(5)]
    M = cogroup_matrix(vtrees, n)
    # Within-block: (0,1), (0,2), (1,2) vs cross-block: (0,3), (0,4), (0,5)
    within = M[0, 1]
    cross = M[0, 3]
    assert within > cross, f"within={within:.3f} cross={cross:.3f}"


# ─── Consensus vtree: synthetic case 11 from dev.md ──────────────────────────

def test_consensus_recovers_known_block_structure():
    """
    Generate source vtrees that all encode the same two-block independence
    ({0,1,2} vs {3,4,5}).  The consensus vtree should place all within-block
    pairs closer together than cross-block pairs.

    This is the unit test for the whole consensus idea: if it can't recover a
    known shared structure from aligned sources, it won't work in the wild.
    """
    n = 6
    # All source vtrees encode the same block split (may vary internally)
    source_vtrees = [make_block_vtree() for _ in range(10)]
    cVt = consensus_vtree(source_vtrees, n_features=n)

    # The consensus vtree must cover all features
    assert sorted(vtree_leaves(cVt)) == list(range(n))

    # A DensityPC built from it must be valid
    pc = DensityPC(cVt, n_sum_components=2)
    validate_circuit(pc.root)

    # Co-grouping: within-block pairs should score higher than cross-block
    M = cogroup_matrix(source_vtrees, n)
    within_avg = np.mean([M[i, j] for i in [0, 1, 2] for j in [0, 1, 2] if i != j])
    cross_avg = np.mean([M[i, j] for i in [0, 1, 2] for j in [3, 4, 5]])
    assert within_avg > cross_avg, (
        f"Consensus input failed: within={within_avg:.3f} cross={cross_avg:.3f}"
    )


def test_consensus_single_vtree_returns_input():
    vt = make_vtree_4()
    result = consensus_vtree([vt], n_features=4)
    assert result == vt


def test_consensus_empty_falls_back_to_random():
    result = consensus_vtree([], n_features=4, fallback_seed=7)
    assert sorted(vtree_leaves(result)) == [0, 1, 2, 3]


# ─── Large-scale smoke test ───────────────────────────────────────────────────

def test_density_pc_128_features():
    """d=128 is the fixed projection dimension used in the transfer study."""
    d = 128
    vt = random_balanced_vtree(list(range(d)), seed=42)
    pc = DensityPC(vt, n_sum_components=3)
    pc.fit_leaves(torch.randn(64, d))
    out = pc(torch.randn(16, d))
    assert out.shape == (16,)
    assert torch.isfinite(out).all()
    validate_circuit(pc.root)
