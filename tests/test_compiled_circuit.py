"""
Tests for CompiledCircuit — the layer-parallel evaluator.

The contract is narrow and absolute: it must return what the recursive
evaluator returns, on every query the recursion supports, for every circuit it
claims to support.  A faster evaluator that is wrong anywhere is worthless,
and the failure is silent (training loss still falls), so the equality checks
here are the load-bearing part — the speed is not tested, it is measured in
poc/time_series/bench_device.py.

Edge cases are chosen to break a naive layered implementation:
  * ragged arities in one layer      -> padding must not leak probability mass
  * single-child sum and product     -> degenerate reductions
  * −inf child values                -> the max-shift/GEMM path must not NaN
  * shared children (a real DAG)     -> a node must be evaluated once, not once
                                        per path, and its slot reused
  * marginalisation                  -> leaf log_integral, not leaf density
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probabilistic_circuits import (
    CategoricalLeaf,
    CompiledCircuit,
    GaussianLeaf,
    GaussianMixtureLeaf,
    ProductNode,
    RegionGraphPC,
    SumNode,
    eval_log_marginal,
    eval_log_prob,
    log_partition,
    random_balanced_vtree,
    time_channel_vtree,
)

ATOL = 1e-4


def _agree(root, comp, x, marginalized=()):
    ref = eval_log_marginal(root, x, marginalized)
    got = comp.log_prob(x, marginalized=list(marginalized) or None)
    assert torch.allclose(ref, got, atol=ATOL, rtol=1e-4), \
        f"max |Δ| = {float((ref - got).abs().max()):.3e}"


def _region_pc(window=6, channels=5, K=4, seed=0):
    d = window * channels
    torch.manual_seed(seed)
    vt = time_channel_vtree(window, channels, mode="time")
    pc = RegionGraphPC(vt, n_sum_components=K, leaf_factory=GaussianLeaf,
                       weight_jitter=0.5, seed=seed)
    pc.validate()
    return pc, d


# ── the core equivalence ────────────────────────────────────────────────────

def test_density_matches_the_recursive_reference():
    pc, d = _region_pc()
    comp = CompiledCircuit(pc.root)
    _agree(pc.root, comp, torch.randn(32, d))


def test_marginals_match_for_every_subset_size():
    pc, d = _region_pc()
    comp = CompiledCircuit(pc.root)
    x = torch.randn(16, d)
    for step in (2, 3, 7, 11):
        _agree(pc.root, comp, x, marginalized=list(range(0, d, step)))


def test_marginalising_everything_gives_log_partition_zero():
    pc, d = _region_pc()
    comp = CompiledCircuit(pc.root)
    assert abs(float(comp.log_partition(d))) < 1e-4
    assert abs(float(log_partition(pc.root))) < 1e-4


def test_gate_raises_when_the_answers_disagree():
    """The gate must actually fail, not just exist."""
    pc, d = _region_pc()
    comp = CompiledCircuit(pc.root)
    with torch.no_grad():                       # corrupt the compiled copy only
        getattr(comp, "GaussianLeaf_mu").add_(3.0)
    with pytest.raises(RuntimeError, match="disagrees"):
        comp.assert_matches_reference(pc.root, torch.randn(8, d))


# ── structural edge cases ───────────────────────────────────────────────────

def test_ragged_arity_in_one_layer_does_not_leak_mass():
    """Two sums at the same depth with 2 and 4 children: the padded columns
    must contribute exactly nothing."""
    leaves = [GaussianLeaf(i) for i in range(4)]
    s_small = SumNode([leaves[0], leaves[1]])
    s_big = SumNode([GaussianLeaf(0), GaussianLeaf(0), GaussianLeaf(0), leaves[1]])
    root = SumNode([ProductNode([s_small, GaussianLeaf(2)]),
                    ProductNode([s_big, GaussianLeaf(3)])])
    comp = CompiledCircuit(root)
    x = torch.randn(16, 4)
    _agree(root, comp, x)
    plans = [p for p in comp.plans if p.kind == "sum"]
    assert any(p.arity == 4 for p in plans), "the ragged layer was not built"


def test_single_child_nodes():
    root = SumNode([ProductNode([SumNode([GaussianLeaf(0)]), GaussianLeaf(1)])])
    comp = CompiledCircuit(root)
    _agree(root, comp, torch.randn(8, 2))


def test_minus_inf_children_do_not_produce_nan():
    """A categorical leaf can return −inf.  Both the logsumexp path and the
    GEMM (max-shift) path must survive it."""
    torch.manual_seed(0)
    lf = CategoricalLeaf(0, n_categories=3)
    with torch.no_grad():
        lf.logits.copy_(torch.tensor([0.0, -float("inf"), -float("inf")]))
    other = CategoricalLeaf(0, n_categories=3)
    root = SumNode([ProductNode([lf, GaussianLeaf(1)]),
                    ProductNode([other, GaussianLeaf(1)])])
    comp = CompiledCircuit(root)
    x = torch.stack([torch.tensor([1.0, 0.3]), torch.tensor([0.0, -0.2])])
    got = comp.log_prob(x)
    assert torch.isfinite(got).all() or (got == -float("inf")).any()
    assert not torch.isnan(got).any()
    _agree(root, comp, x)


def test_dag_sharing_is_evaluated_once():
    """One shared subcircuit under two parents: the compiled schedule must give
    it ONE slot, which is the whole point of the DAG layout."""
    shared = SumNode([GaussianLeaf(0), GaussianLeaf(0)])
    root = SumNode([ProductNode([shared, GaussianLeaf(1)]),
                    ProductNode([shared, GaussianLeaf(2)])])
    comp = CompiledCircuit(root)
    assert comp.n_nodes == 8, comp.n_nodes      # 5 leaves + shared + 2 prods... + root
    _agree(root, comp, torch.randn(8, 3))


def test_mixture_and_categorical_leaves_are_supported():
    root = SumNode([
        ProductNode([GaussianMixtureLeaf(0, n_components=3), CategoricalLeaf(1, 4)]),
        ProductNode([GaussianMixtureLeaf(0, n_components=3), CategoricalLeaf(1, 4)]),
    ])
    comp = CompiledCircuit(root)
    x = torch.stack([torch.randn(16) * 0.5, torch.randint(0, 4, (16,)).float()], dim=1)
    _agree(root, comp, x)


def test_deep_and_wide_circuit():
    """The real shape: many layers, a wide widest layer, learned structure."""
    d, K = 64, 5
    torch.manual_seed(1)
    vt = random_balanced_vtree(list(range(d)), seed=1)
    pc = RegionGraphPC(vt, n_sum_components=K, leaf_factory=GaussianLeaf,
                       weight_jitter=0.5, seed=1)
    comp = CompiledCircuit(pc.root)
    rep = comp.schedule_report()
    assert rep["depth"] > 4 and rep["max_layer_width"] > 10
    assert rep["layers"] < rep["nodes"] / 10, "layering bought nothing"
    _agree(pc.root, comp, torch.randn(24, d))
    _agree(pc.root, comp, torch.randn(24, d), marginalized=[0, 5, 9, 30])


# ── training path ───────────────────────────────────────────────────────────

def test_write_back_makes_the_dag_agree_after_training():
    """Train on the packed parameters, push them home, and the DAG — which owns
    validate()/MPE/serialisation — must reproduce the trained density."""
    pc, d = _region_pc()
    x = torch.randn(64, d)
    comp = CompiledCircuit(pc.root)
    opt = torch.optim.Adam(comp.parameters(), lr=0.05)
    for _ in range(5):
        loss = -comp.log_prob(x).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    trained = comp.log_prob(x)
    assert float(loss) < float(-eval_log_prob(pc.root, x).mean()), "no learning"
    comp.write_back()
    assert torch.allclose(eval_log_prob(pc.root, x), trained, atol=ATOL, rtol=1e-4)
    pc.validate()                       # structure untouched by compilation


def test_compile_switches_the_query_path_and_can_be_switched_back():
    pc, d = _region_pc()
    x = torch.randn(16, d)
    ref = pc.log_prob(x).clone()
    pc.compile_(x_probe=x)
    assert pc.compiled is not None
    assert torch.allclose(pc.log_prob(x), ref, atol=ATOL, rtol=1e-4)
    assert torch.allclose(pc.log_marginal(x, [1, 2]),
                          eval_log_marginal(pc.root, x, [1, 2]), atol=ATOL, rtol=1e-4)
    pc.use_recursive()
    assert pc.compiled is None
    assert torch.allclose(pc.log_prob(x), ref, atol=ATOL, rtol=1e-4)


def test_unsupported_nodes_are_refused_not_silently_wrong():
    """SOS/signed circuits must raise, so the caller falls back rather than
    getting a fast wrong answer."""
    from src.probabilistic_circuits import SquaredPC

    vt = time_channel_vtree(4, 3, mode="time")
    sq = SquaredPC(vt, n_sum_components=2, leaf_factory=GaussianLeaf,
                   seed=0, region_graph=True)
    with pytest.raises((NotImplementedError, AttributeError, TypeError)):
        CompiledCircuit(sq.root)
