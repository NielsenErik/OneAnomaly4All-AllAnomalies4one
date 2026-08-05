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


def test_dense_gemm_layer_keeps_minus_inf_exact():
    """
    The GEMM path computes m + log(exp(g−m) @ W).  When every child of a sum
    node is −inf the matmul is exactly 0, and log 0 must stay −inf: an ε floor
    there would report ≈ −103 for an event of probability zero, which is a
    wrong answer that still looks plausible in a loss curve.  This is the
    regression test for exactly that bug.
    """
    dead = [CategoricalLeaf(0, n_categories=2) for _ in range(3)]
    for lf in dead:                       # every category impossible at v = 1
        with torch.no_grad():
            lf.logits.copy_(torch.tensor([0.0, -float("inf")]))
    # 4 sum nodes over the SAME 3 children -> a dense (GEMM) layer
    prods = [ProductNode([lf, GaussianLeaf(1)]) for lf in dead]
    sums = [SumNode(list(prods)) for _ in range(4)]
    root = ProductNode([SumNode(sums)])
    comp = CompiledCircuit(root)
    assert any(p.dense for p in comp.plans), "no dense layer was built"
    x = torch.tensor([[1.0, 0.5]])        # category 1 has probability zero
    got = comp.log_prob(x)
    ref = eval_log_prob(root, x)
    assert torch.isneginf(ref).all(), "the reference should be −inf here"
    assert torch.isneginf(got).all(), f"GEMM path returned {float(got)} not −inf"


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


# ── leaf initialisation guardrail ───────────────────────────────────────────

def test_leaf_sigma_never_collapses_on_a_plateau_feature():
    """
    A feature that is constant on more than half its samples has MAD exactly 0
    — routine for sensors read under discrete operating conditions (C-MAPSS
    FD002/FD004: 20 of 420 features).  Without a spread-relative floor the leaf
    gets σ ≈ 1.5e-6, ordinary points land at |z| ≈ 7e5, log f ≈ −2.3e11, and
    the first gradient step takes the whole circuit to NaN.  This is the
    regression test for that: every seed of two real subsets used to fail.
    """
    n = 1000
    col = torch.zeros(n)
    col[: n // 10] = torch.linspace(-3.0, 3.0, n // 10)   # 90% identical
    X = torch.stack([col, torch.randn(n)], dim=1)
    assert float(torch.median((col - col.median()).abs())) == 0.0, "not a plateau"

    lf = GaussianLeaf(0)
    lf.fit(X)
    assert float(lf.sigma) >= 0.01 * float(col.std()) * 0.99, float(lf.sigma)
    lp = lf.log_density(X[:, 0])
    assert torch.isfinite(lp).all()
    assert float(lp.min()) > -1e6, f"log density {float(lp.min()):.3e} will explode"


def test_a_plateau_feature_trains_without_nan():
    """End to end: the circuit that used to go NaN now trains."""
    torch.manual_seed(0)
    n, w, c = 512, 4, 6
    X = torch.randn(n, w * c)
    X[:, ::5] = 0.0                       # plateau features, MAD = 0
    X[: n // 20, ::5] = torch.randn(n // 20, X[:, ::5].shape[1]) * 2
    vt = time_channel_vtree(w, c, mode="time")
    pc = RegionGraphPC(vt, n_sum_components=4, leaf_factory=GaussianLeaf,
                       weight_jitter=0.5, seed=0)
    pc.fit_leaves(X)
    comp = CompiledCircuit(pc.root)
    opt = torch.optim.Adam(comp.parameters(), lr=0.05)
    for _ in range(10):
        loss = -comp.log_prob(X).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss), "NaN is back"
    assert float(loss) < 1e6


# ── box (interval) queries — the censored likelihood ────────────────────────

def _survival_pc(window=5, channels=4, n_bins=10, K=3, seed=0):
    """A (window, tau) circuit shaped like SurvivalPC's: Gaussians on the
    sensors, a Categorical on tau, tau coupled deep."""
    from poc.time_series.circuits import (attach_variable, build_window_vtree,
                                          mixed_leaf_factory)

    torch.manual_seed(seed)
    d = window * channels
    base = build_window_vtree("chain", window, channels, X=torch.randn(64, d))
    vt = attach_variable(base, d, where="deep")
    pc = RegionGraphPC(vt, n_sum_components=K,
                       leaf_factory=mixed_leaf_factory(d, n_bins, 1),
                       weight_jitter=0.5, seed=seed)
    pc.validate()
    return pc, d, n_bins


def _z(n, d, n_bins):
    return torch.cat([torch.randn(n, d),
                      torch.randint(0, n_bins, (n, 1)).float()], dim=1)


@pytest.mark.parametrize("lo,hi", [(0.0, float("inf")), (3.0, float("inf")),
                                   (2.0, 7.0), (-float("inf"), 5.0)])
def test_box_query_matches_the_reference_scalar_endpoints(lo, hi):
    pc, d, nb = _survival_pc()
    comp = CompiledCircuit(pc.root)
    z = _z(24, d, nb)
    ref = eval_log_marginal(pc.root, z, (), boxes={d: (lo, hi)})
    got = comp.log_box(z, {d: (lo, hi)})
    assert torch.allclose(ref, got, atol=ATOL, rtol=1e-4), \
        f"max |Δ| = {float((ref - got).abs().max()):.3e}"


def test_box_query_with_per_sample_endpoints():
    """Every censored unit has its OWN censoring time; a batch of them must be
    one pass, not one pass per distinct threshold."""
    pc, d, nb = _survival_pc()
    comp = CompiledCircuit(pc.root)
    z = _z(32, d, nb)
    lo = torch.randint(0, nb, (32,)).float()
    ref = eval_log_marginal(pc.root, z, (), boxes={d: (lo, float("inf"))})
    got = comp.log_box(z, {d: (lo, float("inf"))})
    assert torch.allclose(ref, got, atol=ATOL, rtol=1e-4)


def test_box_and_marginalisation_compose():
    """Survival under dead sensors: box on tau, marginalise the dead channels."""
    pc, d, nb = _survival_pc()
    comp = CompiledCircuit(pc.root)
    z = _z(16, d, nb)
    lo = torch.randint(0, nb, (16,)).float()
    marg = [0, 3, 7]
    ref = eval_log_marginal(pc.root, z, marg, boxes={d: (lo, float("inf"))})
    got = comp.log_box(z, {d: (lo, float("inf"))}, marginalized=marg)
    assert torch.allclose(ref, got, atol=ATOL, rtol=1e-4)


def test_full_range_box_equals_marginalising_that_variable():
    """S(0) = P(tau >= 0) = 1, so the box over ALL bins must equal the density
    with tau marginalised out.  An identity, so it must hold exactly."""
    pc, d, nb = _survival_pc()
    comp = CompiledCircuit(pc.root)
    z = _z(16, d, nb)
    box = comp.log_box(z, {d: (0.0, float("inf"))})
    marg = comp.log_prob(z, marginalized=[d])
    assert torch.allclose(box, marg, atol=1e-5, rtol=1e-5)


def test_gaussian_and_mixture_leaves_support_boxes():
    """Box support must not be Categorical-only — the tau leaf is categorical
    today, but a continuous tau is a one-line config change."""
    for leaf in (GaussianLeaf, lambda i: GaussianMixtureLeaf(i, n_components=3)):
        root = SumNode([ProductNode([leaf(0), leaf(1)]),
                        ProductNode([leaf(0), leaf(1)])])
        comp = CompiledCircuit(root)
        x = torch.randn(12, 2)
        lo = torch.rand(12) - 2.0
        for endpoints in ((-1.0, 1.5), (lo, float("inf"))):
            ref = eval_log_marginal(root, x, (), boxes={0: endpoints})
            got = comp.log_box(x, {0: endpoints})
            assert torch.allclose(ref, got, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("n_components", [1, 3])
def test_compiled_matches_recursion_with_fitted_sigma_floors(n_components):
    """
    The per-feature σ floor is part of the density, so the compiled path must
    pack it — not recompute σ from log_sigma with the default epsilon.

    Every other test here builds leaves and never calls `fit`, which leaves
    every floor at its 1e-5 default and makes the two paths agree by accident.
    Fitting is what separates them: on a near-constant feature the floor
    becomes ~1e-3, and a compiled path that ignores it disagrees with the
    recursion by tens of nats — which is exactly how this was found, as a
    fast-path refusal mid-benchmark rather than as a wrong number.
    """
    torch.manual_seed(0)
    n, w, c = 256, 4, 5
    X = torch.randn(n, w * c)
    X[:, ::4] = 1.0                       # near-constant -> floor binds
    X[: n // 25, ::4] += 0.5

    leaf = (GaussianLeaf if n_components == 1
            else (lambda i: GaussianMixtureLeaf(i, n_components=n_components)))
    vt = time_channel_vtree(w, c, mode="time")
    pc = RegionGraphPC(vt, n_sum_components=3, leaf_factory=leaf,
                       weight_jitter=0.5, seed=0)
    pc.fit_leaves(X)

    floors = [float(m.sigma_floor) for m in pc.root.modules()
              if hasattr(m, "sigma_floor")]
    assert max(floors) > 1e-4, "test is vacuous unless some floor actually binds"

    comp = CompiledCircuit(pc.root)
    x = X[:32]
    assert torch.allclose(eval_log_prob(pc.root, x), comp.log_prob(x),
                          atol=ATOL, rtol=1e-4)
    assert torch.allclose(eval_log_marginal(pc.root, x, (0, 1)),
                          comp.log_prob(x, marginalized=[0, 1]),
                          atol=ATOL, rtol=1e-4)
    lo = torch.full((32,), -0.5)
    ref = eval_log_marginal(pc.root, x, (), boxes={0: (lo, float("inf"))})
    assert torch.allclose(ref, comp.log_box(x, {0: (lo, float("inf"))}),
                          atol=ATOL, rtol=1e-4)
