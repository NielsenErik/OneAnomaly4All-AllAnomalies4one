"""
Tests for the region-graph DAG layout, interval/box queries and temporal vtrees.

The load-bearing claims are: (1) the DAG is the SAME model as the tree, only
smaller — all four circuit properties still hold and the density is still
exactly normalised; (2) size grows as O(d·K²) rather than O(d·K^depth);
(3) box queries are exact, which is what makes survival/censored likelihoods
exact; (4) the K units of a region do not collapse into one another.
"""
import math

import pytest
import torch

from src.probabilistic_circuits import (
    CategoricalLeaf,
    DensityPC,
    GaussianLeaf,
    GaussianMixtureLeaf,
    RegionGraphPC,
    SquaredPC,
    circuit_size,
    eval_log_marginal,
    random_balanced_vtree,
    time_channel_vtree,
    validate_circuit,
    validate_determinism,
    validate_structured_decomposability,
    vtree_leaves,
)


def _vt(d, seed=0):
    return random_balanced_vtree(list(range(d)), seed=seed)


# ─── properties ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("d,K", [(8, 3), (16, 4), (31, 2)])
def test_all_four_properties_hold(d, K):
    pc = RegionGraphPC(_vt(d), n_sum_components=K, leaf_factory=GaussianLeaf)
    validate_circuit(pc.root)                       # smoothness + decomposability
    validate_structured_decomposability(pc.root, pc.vtree)
    pc.validate()


def test_k1_is_deterministic():
    """K=1 has no sum nodes at all, so determinism holds trivially."""
    pc = RegionGraphPC(_vt(8), n_sum_components=1, leaf_factory=GaussianLeaf)
    validate_determinism(pc.root, torch.randn(16, 8))


def test_mixture_is_not_deterministic():
    """Full-support mixtures must FAIL the determinism check — by design."""
    pc = RegionGraphPC(_vt(8), n_sum_components=3, leaf_factory=GaussianLeaf)
    with pytest.raises(AssertionError, match="Determinism"):
        validate_determinism(pc.root, torch.randn(16, 8))


# ─── exactness ──────────────────────────────────────────────────────────────

def test_partition_is_one():
    pc = RegionGraphPC(_vt(12), n_sum_components=4, leaf_factory=GaussianLeaf)
    pc.fit_leaves(torch.randn(64, 12))
    assert abs(float(pc.log_partition().detach())) < 1e-5


def test_marginal_dominates_joint_and_is_consistent():
    d = 10
    pc = RegionGraphPC(_vt(d), n_sum_components=3, leaf_factory=GaussianLeaf)
    X = torch.randn(32, d)
    pc.fit_leaves(X)
    joint = pc.log_prob(X)
    marg = pc.log_marginal(X, [0, 3, 7])
    assert torch.all(marg >= joint - 1e-4)          # integrating out adds mass
    # marginalizing everything = log 1 = 0
    allm = pc.log_marginal(X, list(range(d)))
    assert torch.allclose(allm, torch.zeros_like(allm), atol=1e-4)


def test_box_over_full_support_equals_marginal():
    d = 8
    pc = RegionGraphPC(_vt(d), n_sum_components=3, leaf_factory=GaussianLeaf)
    X = torch.randn(24, d)
    pc.fit_leaves(X)
    inf = float("inf")
    boxes = {i: (-inf, inf) for i in [1, 4]}
    assert torch.allclose(pc.log_box(X, boxes), pc.log_marginal(X, [1, 4]), atol=1e-5)


def test_box_is_additive_over_a_split():
    """P(x_i < a) + P(x_i >= a) must equal the marginal with x_i integrated out."""
    d = 6
    pc = RegionGraphPC(_vt(d), n_sum_components=3, leaf_factory=GaussianLeaf)
    X = torch.randn(24, d)
    pc.fit_leaves(X)
    inf, a = float("inf"), 0.3
    lo = pc.log_box(X, {2: (-inf, a)})
    hi = pc.log_box(X, {2: (a, inf)})
    total = torch.logsumexp(torch.stack([lo, hi]), dim=0)
    assert torch.allclose(total, pc.log_marginal(X, [2]), atol=1e-4)


def test_gaussian_leaf_interval_matches_quadrature():
    leaf = GaussianLeaf(0, mu_init=0.4, sigma_init=0.8)
    grid = torch.linspace(-1.0, 1.5, 20001)
    dens = leaf.log_density(grid).exp()
    numeric = torch.trapz(dens, grid)
    exact = leaf.log_interval(-1.0, 1.5).exp()
    assert abs(float(exact) - float(numeric)) < 1e-4


def test_gaussian_mixture_leaf_interval_matches_quadrature():
    leaf = GaussianMixtureLeaf(0, n_components=3)
    grid = torch.linspace(-2.0, 0.7, 20001)
    numeric = torch.trapz(leaf.log_density(grid).exp(), grid)
    exact = leaf.log_interval(-2.0, 0.7).exp()
    assert abs(float(exact) - float(numeric)) < 1e-4


def test_far_tail_interval_does_not_underflow():
    """The reason for the two-sided anchoring: a naive Phi(hi)-Phi(lo) is 0 here."""
    leaf = GaussianLeaf(0, mu_init=0.0, sigma_init=1.0)
    v = float(leaf.log_interval(8.0, 9.0))
    assert math.isfinite(v) and v < -30          # tiny but representable in log space


def test_categorical_interval_is_inclusive_suffix_sum():
    leaf = CategoricalLeaf(0, n_categories=5)
    with torch.no_grad():
        leaf.logits.copy_(torch.tensor([0.1, 0.5, -0.2, 0.3, 0.0]))
    p = torch.softmax(leaf.logits, 0)
    assert abs(float(leaf.log_interval(2, 4).exp()) - float(p[2:].sum())) < 1e-6
    inf = float("inf")
    assert abs(float(leaf.log_interval(-inf, inf).exp()) - 1.0) < 1e-6


def test_per_sample_interval_endpoints():
    """Per-unit censoring times must work as one vectorised call."""
    leaf = CategoricalLeaf(0, n_categories=6)
    lo = torch.tensor([0.0, 2.0, 5.0])
    out = leaf.log_interval(lo, float("inf"))
    assert out.shape == (3,)
    ref = torch.stack([leaf.log_interval(float(v), float("inf")) for v in lo])
    assert torch.allclose(out, ref, atol=1e-6)


# ─── the size claim ─────────────────────────────────────────────────────────

def test_dag_is_linear_in_d_and_far_smaller_than_tree():
    K = 3
    sizes = [circuit_size(RegionGraphPC(_vt(d), K, GaussianLeaf).root)["leaf"]
             for d in (8, 16, 32)]
    assert sizes == [8 * K, 16 * K, 32 * K]        # exactly d*K, no depth term
    tree = circuit_size(DensityPC(_vt(16), K, GaussianLeaf).root)["leaf"]
    assert tree > 20 * sizes[1]                    # tree blows up on the same vtree


def test_products_are_shared_not_duplicated():
    """K parents mixing one product list is what makes the layout a DAG."""
    d, K = 8, 3
    sz = circuit_size(RegionGraphPC(_vt(d), K, GaussianLeaf).root)
    # (d-1) internal regions, each with exactly K*K products built once
    assert sz["product"] == (d - 1) * K * K


# ─── symmetry breaking (the bug that made p(tau|x) constant) ────────────────

def _region_unit_outputs(pc, d, x):
    """Evaluate every unit of the root's first child region on the same input."""
    child = pc.region_graph.partitions[0][0]
    region = pc._regions[id(child)]
    marg = [i for i in range(d) if i not in child.scope]
    return region, [eval_log_marginal(u, x, marg) for u in region]


def test_region_units_do_not_stay_identical():
    """
    With uniform sum weights the K units of a region are the SAME function —
    a uniform mixture of a shared product list does not depend on which unit
    you ask — so they receive identical gradients forever and the circuit
    collapses to an independence model.  weight_jitter must prevent that.

    Note both tests fit the leaves first: with default identical leaves every
    product is the same function too, and then no weighting can tell the units
    apart.  Leaf jitter and weight jitter are both required, which is exactly
    the pair of bugs this guards.
    """
    d = 6
    x, X = torch.randn(8, d), torch.randn(64, d)
    pc = RegionGraphPC(_vt(d), n_sum_components=4, leaf_factory=GaussianLeaf,
                       weight_jitter=0.5, seed=0)
    pc.fit_leaves(X)
    region, outs = _region_unit_outputs(pc, d, x)
    assert len(region) >= 2
    assert not torch.allclose(outs[0], outs[1], atol=1e-6)


def test_weight_jitter_zero_reproduces_the_collapse():
    """Guard-rail: the failure mode is real, so it must stay reproducible."""
    d = 6
    x, X = torch.randn(8, d), torch.randn(64, d)
    pc = RegionGraphPC(_vt(d), n_sum_components=4, leaf_factory=GaussianLeaf,
                       weight_jitter=0.0)
    pc.fit_leaves(X)                       # leaves differ, weights do not
    _, outs = _region_unit_outputs(pc, d, x)
    assert torch.allclose(outs[0], outs[1], atol=1e-6)


# ─── temporal vtrees ────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["time", "channel"])
def test_temporal_vtree_covers_the_window(mode):
    w, C = 5, 4
    vt = time_channel_vtree(w, C, mode=mode)
    assert sorted(vtree_leaves(vt)) == list(range(w * C))
    pc = RegionGraphPC(vt, n_sum_components=2, leaf_factory=GaussianLeaf)
    pc.validate()


def test_channel_groups_partition_check():
    with pytest.raises(ValueError, match="partition"):
        time_channel_vtree(4, 4, mode="channel", channel_groups=[[0, 1], [2]])


def test_grouped_vtree_keeps_group_members_together():
    w, C = 3, 4
    vt = time_channel_vtree(w, C, mode="channel", channel_groups=[[0, 1], [2, 3]])
    # the root split must separate {ch 0,1} from {ch 2,3}
    left = set(vtree_leaves(vt.left))
    assert left == {t * C + c for t in range(w) for c in (0, 1)}


# ─── SOS on the DAG layout ──────────────────────────────────────────────────

def test_squared_pc_region_graph_is_normalised_and_valid():
    d, K = 8, 2
    pc = SquaredPC(_vt(d), n_sum_components=K, leaf_factory=GaussianLeaf,
                   region_graph=True)
    pc.validate()
    X = torch.randn(16, d)
    pc.fit_leaves(X)
    lp = pc.log_prob(X)
    assert lp.shape == (16,) and torch.isfinite(lp).all()
    # marginalizing every variable of a normalised density gives log 1 = 0
    allm = pc.log_marginal(X, list(range(d)))
    assert torch.allclose(allm, torch.zeros_like(allm), atol=1e-3)


def test_squared_pc_region_graph_is_smaller_than_tree():
    d, K = 12, 2
    dag = circuit_size(SquaredPC(_vt(d), K, GaussianLeaf, region_graph=True).root)
    tree = circuit_size(SquaredPC(_vt(d), K, GaussianLeaf, region_graph=False).root)
    assert dag["leaf"] < tree["leaf"]


# ─── region graphs: n-ary, multi-partition, chain ───────────────────────────

from src.probabilistic_circuits import (          # noqa: E402
    RegionNode,
    chain_region_graph,
    curvature_region_graph,
    delta_window_transform,
    is_structured_decomposable_rg,
    learned_region_graph,
    region_graph_arity,
    region_graph_from_vtree,
    region_nodes,
)


def _dep_data(n=300, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    for a, b in [(0, 1), (0, 2), (4, 5), (8, 9), (8, 10)]:
        X[:, b] = X[:, a] * 0.9 + 0.15 * torch.randn(n, generator=g)
    return X


def test_vtree_lifts_to_equivalent_region_graph():
    vt = _vt(8)
    rg = region_graph_from_vtree(vt)
    assert is_structured_decomposable_rg(rg)
    assert region_graph_arity(rg) == 2
    assert rg.scope == frozenset(range(8))


@pytest.mark.parametrize("method", ["orc", "forman", "spectral"])
def test_learned_region_graph_is_valid_and_exact(method):
    X = _dep_data()
    rg = learned_region_graph(X, method=method, max_arity=4)
    assert is_structured_decomposable_rg(rg)      # n_partitions=1
    pc = RegionGraphPC(rg, n_sum_components=3, leaf_factory=GaussianLeaf)
    pc.validate()
    pc.fit_leaves(X)
    assert abs(float(pc.log_partition().detach())) < 1e-4
    assert pc.is_structured_decomposable


def test_multi_partition_gives_up_structured_decomposability_only():
    """Exact density/marginals must survive; SOS must refuse."""
    X = _dep_data()
    rg = curvature_region_graph(X, curvature="forman", n_partitions=3, max_arity=4)
    if is_structured_decomposable_rg(rg):
        pytest.skip("this data yielded a single partition per region")
    pc = RegionGraphPC(rg, n_sum_components=3, leaf_factory=GaussianLeaf)
    pc.validate()                                  # smoothness + decomposability
    pc.fit_leaves(X)
    assert abs(float(pc.log_partition().detach())) < 1e-4      # still normalised
    lp = pc.log_prob(X[:16])
    assert torch.all(pc.log_marginal(X[:16], [0, 1]) >= lp - 1e-4)
    with pytest.raises(AssertionError, match="multiple partitions"):
        pc.validate(require_structured=True)
    with pytest.raises(ValueError, match="structured-decomposable"):
        SquaredPC(rg, n_sum_components=2, leaf_factory=GaussianLeaf)


def test_sos_accepts_nary_single_partition_region_graph():
    X = _dep_data()
    rg = learned_region_graph(X, method="spectral", max_arity=4)
    sq = SquaredPC(rg, n_sum_components=2, leaf_factory=GaussianLeaf)
    sq.validate()
    sq.fit_leaves(X)
    allm = sq.log_marginal(X[:8], list(range(X.shape[1])))
    assert torch.allclose(allm, torch.zeros_like(allm), atol=1e-3)


def test_chain_region_graph_shape_and_exactness():
    w, C = 6, 3
    rg = chain_region_graph(w, C)
    assert rg.scope == frozenset(range(w * C))
    assert is_structured_decomposable_rg(rg)
    pc = RegionGraphPC(rg, n_sum_components=4, leaf_factory=GaussianLeaf)
    pc.validate()
    X = torch.randn(32, w * C)
    pc.fit_leaves(X)
    assert abs(float(pc.log_partition().detach())) < 1e-4


def test_chain_is_order_sensitive_unlike_balanced_vtree():
    """
    The point of the chain: permuting timesteps must change the density.
    Built on an AR(1) signal so there IS temporal structure to detect.
    """
    w, C, n = 8, 2, 400
    g = torch.Generator().manual_seed(0)
    seq = torch.zeros(n, w, C)
    for t in range(1, w):
        seq[:, t] = 0.9 * seq[:, t - 1] + 0.44 * torch.randn(n, C, generator=g)
    X = seq.reshape(n, -1)
    Xd = delta_window_transform(X, w, C)
    pc = RegionGraphPC(chain_region_graph(w, C), n_sum_components=4,
                       leaf_factory=GaussianLeaf, seed=0)
    pc.fit_leaves(Xd)
    opt = torch.optim.Adam(pc.parameters(), lr=0.05)
    for _ in range(120):
        loss = -pc.log_prob(Xd).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    perm = seq.clone()
    idx = torch.randperm(w, generator=g)
    perm[:, :, 0] = seq[:, idx, 0]
    Pd = delta_window_transform(perm.reshape(n, -1), w, C)
    with torch.no_grad():
        assert float(pc.log_prob(Xd).mean()) > float(pc.log_prob(Pd).mean()) + 0.1


def test_delta_transform_is_invertible_and_unit_jacobian():
    w, C = 5, 3
    X = torch.randn(16, w * C)
    D = delta_window_transform(X, w, C)
    back = D.reshape(-1, w, C).cumsum(1).reshape(len(X), -1)
    assert torch.allclose(back, X, atol=1e-5)
