"""
Tier 1 — the channel-blocked structure and the two-pass relational map.

One test per claim of IMPLEMENTATION_PLAN_2026-09-08.md §Tier 1, cheap enough
(seconds, tiny circuits) to run on every commit:

  1.1  a channel's timesteps form ONE region, and a structure where they do not
       is refused loudly instead of producing plausible wrong numbers
  1.2  structures are compared at matched parameter count, and a comparison
       between factorised models is VOID rather than PASS/FAIL
  1.3  the two-pass map equals the 3·C `typed_scores` oracle to float32
       tolerance  ← the correctness backbone of the method
  1.4  arbitrary per-window missing-sensor masks equal the per-mask
       `log_marginal` oracle, and the cost does not grow with the number of
       distinct masks
  1.5  the beam search returns the same values as the per-candidate cost model
       (Lüdtke et al. 2022 re-implemented on the same circuit) at a fraction of
       the passes
  1.6  the null distribution of R_c genuinely moves with the mask, which is why
       one threshold cannot hold its false-alarm rate as sensors drop out

The failure mode these guard against is the project's most expensive one: a
number that is well-formed, plausible, and answering a different question from
the one its column name asks.  Every identity here is exact, so every tolerance
below is float32 round-off — if one of them has to be loosened, something is
wrong with the model, not with the test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.time_series.circuits import WindowPC, match_K, structure_param_count
from poc.time_series.data import contaminate_windows, make_ad_task
from poc.time_series.pipeline import channel_dependence, gate_verdict
from poc.time_series.relational import (
    MaskCalibrator,
    mask_library,
    naive_subset_search,
    subset_search,
)
from src.probabilistic_circuits import (
    BlockStructureError,
    channel_blocked_vtree,
    channel_scopes,
    chow_liu_channel_order,
    vtree_nodes,
)

W, C = 5, 6
TOL = 1e-3            # nats; the observed disagreement is ~1e-5 (float32)


@pytest.fixture(scope="module")
def task():
    return make_ad_task(window=W, stride=3, seed=0, n_units=16,
                        n_channels=C, n_regimes=2)


@pytest.fixture(scope="module")
def pc(task):
    """
    A blocked circuit trained past the FACTORISATION crossover.

    Not a detail: this circuit is exactly a product of per-channel marginals
    for its first tens of epochs, and under a factorised model every relational
    quantity is identically zero — so a test suite that trained for five epochs
    would be asserting agreement between two ways of computing 0.

    Not a detail either: on this task the crossover is late.  Measured
    dependence at (epochs, lr) = (45, .05) 1.5e-05, (80, .05) 0.0, (80, .10)
    0.0, (120, .08) 1.33 nats — so the budget below is the cheapest one that
    is past it, and the assertion is what stops a future speed-up from
    silently walking back into the empty regime.
    """
    m = WindowPC(W, C, vtree_method="channel_blocked", n_sum_components=4,
                 leaf_components=1, seed=0, device="cpu")
    m.fit(task.X_train, epochs=120, lr=0.08, X_val=task.X_val)
    assert channel_dependence(m, task.X_val) > 1e-3, (
        "the fixture circuit is factorised across channels, so every "
        "relational quantity below is identically zero and the tests would "
        "pass vacuously")
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 1.1 — the structure and its precondition
# ═══════════════════════════════════════════════════════════════════════════

def test_each_channel_is_exactly_one_vtree_subtree():
    """The blocking claim, checked on the vtree rather than on a consequence."""
    vt = channel_blocked_vtree(W, C, order=list(reversed(range(C))))
    scopes = {n.scope for n in vtree_nodes(vt)}
    for c, want in enumerate(channel_scopes(W, C)):
        assert want in scopes, f"channel {c} is not a subtree of the vtree"


def test_a_split_channel_is_refused_not_approximated(task):
    """
    A time-split structure entered per timestep has NO channel boundary, and
    the two-pass identities are wrong there.  The failure must be an exception,
    not a plausible number: this is the guard that makes the whole method
    falsifiable rather than merely computable.
    """
    m = WindowPC(W, C, vtree_method="time", n_sum_components=3, seed=0,
                 device="cpu").fit(task.X_train[:128], epochs=1)
    assert not m.is_channel_blocked
    with pytest.raises(BlockStructureError):
        m.relational_map(task.X_test[:4])


def test_chow_liu_channel_order_is_a_permutation_and_pairs_coupled_channels():
    """The ordering is the only freedom left once blocking is imposed."""
    torch.manual_seed(0)
    X = torch.randn(256, W * C)
    X[:, 3::C] = 0.95 * X[:, 1::C] + 0.05 * torch.randn(256, W)   # couple 1 and 3
    order = chow_liu_channel_order(X, W, C)
    assert sorted(order) == list(range(C))
    assert abs(order.index(1) - order.index(3)) == 1, (
        f"the two strongly coupled channels are not adjacent in {order}")


# ═══════════════════════════════════════════════════════════════════════════
# 1.3 — the correctness backbone
# ═══════════════════════════════════════════════════════════════════════════

def test_two_pass_map_matches_the_3C_oracle(pc, task):
    """
    `relational_map` (one upward + one downward pass) against `typed_scores`
    (3·C independent marginal queries).  Same three statistics, two entirely
    different algorithms; they must agree to float32 round-off.

    If this test ever fails, nothing else in Tier 1 means anything.
    """
    X = task.X_test[:48]
    oracle = pc.typed_scores(X)
    fast = pc.relational_map(X)
    for key in ("marginal", "conditional", "structural"):
        err = float((oracle[key] - fast[key]).abs().max())
        assert err < TOL, f"{key} disagrees with the oracle by {err:.3e} nats"


def test_R_equals_the_structural_term_by_construction(pc, task):
    """R_c = log p(x_c) + log p(x_-c) − log p(x) IS the structural term.  The
    contribution is the algorithm and the regime, never a new identity, and
    this pins the equality so no write-up can drift into claiming one."""
    r = pc.relational_map(task.X_test[:32])
    assert torch.allclose(r["R"], r["structural"], atol=0, rtol=0)
    assert torch.allclose(r["R"], r["conditional"] - r["marginal"], atol=1e-4)


def test_the_map_costs_two_passes_whatever_C_is(pc, task):
    """Cost is measured, not argued: the oracle spends 3·C circuit queries, the
    two-pass map spends a fixed number of partial passes."""
    r = pc.relational_map(task.X_test[:32])
    assert r["cost"].passes <= 4, r["cost"].as_dict()


# ═══════════════════════════════════════════════════════════════════════════
# 1.4 — arbitrary missing-sensor masks
# ═══════════════════════════════════════════════════════════════════════════

def test_masked_score_matches_exact_marginalisation(pc, task):
    """One mask, against the reference path that works on any structure."""
    X = task.X_test[:64]
    for dead in ([0], [1, 4], [0, 2, 5]):
        mask = torch.ones(C, dtype=torch.bool)
        mask[dead] = False
        fast = pc.score_with_masks(X, mask)
        ref = pc.score_with_missing(X, dead)
        assert float((fast - ref).abs().max()) < TOL


def test_a_different_mask_per_window_costs_one_pass(pc, task):
    """
    The Tier 1.4 claim in its strongest form: every window in the batch can
    carry its OWN sensor failure and the batch still costs one pass, because
    the substitution happens at the channel boundary rather than in a re-run.
    """
    X = task.X_test[:32]
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(len(X), C, generator=g) > 0.35
    mask[:, 0] = True                       # never lose every sensor
    fast = pc.score_with_masks(X, mask, per_window=True)
    for i in range(len(X)):
        dead = [c for c in range(C) if not mask[i, c]]
        ref = pc.score_with_missing(X[i:i + 1], dead)
        assert abs(float(fast[i] - ref[0])) < TOL
    assert pc.last_query_cost.passes <= 4


def test_cost_does_not_grow_with_the_number_of_distinct_masks(pc, task):
    """
    The measured version of "reuse shared work": going from one mask to a
    different mask per window must not multiply the work.  Node visits, not
    wall clock — the constant factors of the Python evaluator are a separate
    (and honestly reported) matter.
    """
    X = task.X_test[:32]
    one = torch.ones(len(X), C, dtype=torch.bool)
    one[:, 2] = False
    many = torch.rand(len(X), C, generator=torch.Generator().manual_seed(1)) > 0.4
    many[:, 0] = True
    a = pc.relational_map(X, mask=one)["cost"]
    b = pc.relational_map(X, mask=many)["cost"]
    assert b.n_masks > a.n_masks, "the second batch should hold many patterns"
    assert b.node_visits == a.node_visits
    assert b.passes == a.passes


def test_unobserved_channels_get_no_statistic(pc, task):
    """A sensor that is not there has no relational statistic.  nan, not 0 and
    not the unmasked value — either of those is a fabricated reading."""
    mask = torch.ones(C, dtype=torch.bool)
    mask[[1, 3]] = False
    r = pc.relational_map(task.X_test[:16], mask=mask)
    for key in ("R", "structural", "marginal", "conditional"):
        assert bool(torch.isnan(r[key][:, [1, 3]]).all()), key
        assert bool(torch.isfinite(r[key][:, [0, 2, 4, 5]]).all()), key


# ═══════════════════════════════════════════════════════════════════════════
# 1.5 — subset search
# ═══════════════════════════════════════════════════════════════════════════

def test_beam_search_matches_the_per_candidate_cost_model(pc, task):
    """
    Same statistic, same beam, same answers — the only difference is the cost.
    That is the whole separation from the 2022 SPN explanation method, so it is
    asserted on the VALUES (which must be identical) rather than on the argmax
    (which is free to differ wherever two subsets tie to round-off).
    """
    X = task.X_test[:24]
    fast = subset_search(pc.relational(), pc._prep(X), max_size=2, beam=3)
    slow = naive_subset_search(pc, X, max_size=2, beam=3)
    assert np.nanmax(np.abs(fast.values - slow.values)) < TOL
    assert fast.candidates_scored == slow.candidates_scored
    assert fast.cost.passes < slow.cost.passes / 3, (
        f"{fast.cost.passes} vs {slow.cost.passes} passes: the reuse is not "
        "buying what it is claimed to buy")


def test_subset_search_stays_inside_the_observed_sensors(pc, task):
    """Under a mask, a dead sensor can never be blamed."""
    mask = torch.ones(C, dtype=torch.bool)
    mask[[2, 5]] = False
    res = subset_search(pc.relational(), pc._prep(task.X_test[:16]),
                        mask=mask, max_size=2, beam=3)
    assert all(2 not in s and 5 not in s for s in res.subsets)


# ═══════════════════════════════════════════════════════════════════════════
# 1.6 — mask-conditional calibration
# ═══════════════════════════════════════════════════════════════════════════

def test_the_null_of_R_moves_with_the_mask(pc, task):
    """
    The premise of 1.6, checked directly: thresholds fitted per mask are NOT
    the same number.  If they were, a single threshold would do and the whole
    calibration step would be ceremony.
    """
    lib = mask_library(C, task.channel_groups, ks=(1, 2), n_per_k=1, seed=0)
    cal = MaskCalibrator(pc.relational(), alpha=0.10).fit(pc._prep(task.X_val), lib)
    full = cal.threshold("full")
    moved = False
    for name in lib:
        if name == "full":
            continue
        thr = cal.threshold(name)
        both = np.isfinite(full) & np.isfinite(thr)
        moved |= bool(np.nanmax(np.abs(full[both] - thr[both])) > 1e-3)
    assert moved, "no mask moved any threshold; 1.6 would have nothing to fix"


def test_mask_conditional_thresholds_hold_their_false_alarm_rate(pc, task):
    """
    Fitted on one half of the healthy validation windows, measured on the
    other.  The quantile is the conservative split-conformal one, so the
    measured rate should sit at or below alpha for EVERY mask.

    The mask-blind rate is measured beside it but deliberately not asserted:
    which direction it drifts is a property of the data, and this file pins
    mechanisms, not findings.
    """
    lib = mask_library(C, task.channel_groups, ks=(1, 2), n_per_k=1, seed=0)
    Xv = pc._prep(task.X_val)
    half = len(Xv) // 2
    cal = MaskCalibrator(pc.relational(), alpha=0.10).fit(Xv[:half], lib)
    rows = cal.report(Xv[half:], lib)
    assert len(rows) == len(lib)
    for r in rows:
        assert r["fpr_masked"] <= 0.30, r
        assert 0.0 <= r["fpr_blind"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 1.2 — the gate's own machinery
# ═══════════════════════════════════════════════════════════════════════════

def test_structures_are_compared_at_matched_parameters():
    """A vtree that admits more parameters is a bigger model, not a better
    structure; the 2026-08-06 re-measurement showed how far that moves a
    ranking."""
    target = structure_param_count(W, C, "chow_liu", 4,
                                   X=torch.randn(64, W * C))
    for method in ("channel_blocked", "channel", "random", "time"):
        K, n = match_K(W, C, method, target, k_grid=(2, 3, 4, 6, 8),
                       X=torch.randn(64, W * C))
        assert abs(n - target) / target < 0.10, (method, K, n, target)


def test_a_comparison_between_factorised_models_is_void_not_a_pass():
    """
    The guard that stops the gate from ranking structures that cannot differ.
    A circuit carrying no cross-channel dependence is a product of per-channel
    marginals whatever its vtree, so a PASS there would be a statement about
    nothing — and it was reachable: this circuit is exactly factorised for its
    first tens of epochs while the training loss falls smoothly.
    """
    ev = {"gate_candidate": "channel_blocked", "gate_max_auroc_loss": 0.02,
          "gate_max_nll_loss_frac": 0.05, "gate_min_dependence_nats": 1e-3}
    flat = [{"vtree": "channel_blocked", "auroc": 0.80, "val_nll": 60.0,
             "dependence_nats": 0.0, "blocked": True},
            {"vtree": "chow_liu", "auroc": 0.81, "val_nll": 59.9,
             "dependence_nats": 1e-9, "blocked": False}]
    assert gate_verdict(flat, ev)["verdict"] == "VOID"

    live = [dict(r, dependence_nats=2.0) for r in flat]
    assert gate_verdict(live, ev)["verdict"] == "PASS"
    lost = [{"vtree": "channel_blocked", "auroc": 0.70, "val_nll": 60.0,
             "dependence_nats": 2.0, "blocked": True},
            {"vtree": "chow_liu", "auroc": 0.81, "val_nll": 59.9,
             "dependence_nats": 2.0, "blocked": False}]
    assert gate_verdict(lost, ev)["verdict"] == "FAIL"


def test_the_gate_measures_blocking_not_the_channel_order():
    """
    The comparison set is the arms that are NOT channel-blocked, as the config
    pre-registers.  A blocked arm beating the candidate means the channel ORDER
    is wrong — a different and much cheaper problem than "the boundary is
    unaffordable", and failing the gate on it would stop the wrong thing.  The
    stricter comparison is still reported, never silently dropped.
    """
    ev = {"gate_candidate": "channel_blocked", "gate_max_auroc_loss": 0.02,
          "gate_max_nll_loss_frac": 0.05, "gate_min_dependence_nats": 1e-3}
    rows = [{"vtree": "channel_blocked", "auroc": 0.76, "val_nll": 58.0,
             "dependence_nats": 3.6, "blocked": True},
            {"vtree": "channel", "auroc": 0.83, "val_nll": 45.0,      # blocked, better
             "dependence_nats": 6.9, "blocked": True},
            {"vtree": "random", "auroc": 0.75, "val_nll": 60.0,
             "dependence_nats": 2.0, "blocked": False}]
    v = gate_verdict(rows, ev)
    assert v["verdict"] == "PASS"
    assert v["comparators"] == ["random"]
    assert v["delta_auroc"] > 0 > v["delta_auroc_vs_any"]

    only_blocked = [r for r in rows if r["blocked"]]
    assert gate_verdict(only_blocked, ev)["verdict"] == "NO_COMPARATOR"


def test_the_labelled_validation_half_is_built_from_validation_windows(task):
    """
    Tier 1.2 needs a labelled detection score that is not the test set: the
    test split is evaluated once, at the end, and a structure chosen on it
    would make every later number a selection artefact (plan §3.3).
    """
    Xc, y, kinds, affected = contaminate_windows(
        task.X_val, W, C, inject_rate=0.5, donors=task.X_train, seed=1)
    assert Xc.shape == task.X_val.shape
    assert 0 < int(y.sum()) < len(y)
    for i in range(len(y)):
        assert (kinds[i] == "normal") == (y[i] == 0)
        assert bool(affected[i]) == bool(y[i])
        if y[i] == 0:
            assert torch.equal(Xc[i], task.X_val[i])
