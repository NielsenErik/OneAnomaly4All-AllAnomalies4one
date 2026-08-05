"""
Experiment hygiene — the bug CLASS, not the bugs.

Six silent degeneracies have each produced a confident wrong answer here, and
each was fixed individually afterwards.  The generalisable lesson is already
written down in hand-off §3:

    an A/B flag must switch exactly one thing, and a test should assert that
    the "off" branch reproduces a recorded number.  Check the control, not
    just the treatment.

This file turns that lesson, and the four hazards still documented only in
prose, into executable checks.  Three of them FAIL on purpose and are marked
`xfail(strict=True)`: they are the open items from hand-off §A.5-A.7, and the
strict marker means the suite will tell you the day someone fixes one.

The four recurring shapes, each with a test below:

  1. one flag, two effects        (init vs runtime floor; leaf_components)
  2. a diagnostic written against a LITERAL instead of the object's own state
                                  (`@floor` counting an absolute 1e-3)
  3. a guardrail whose threshold is unrelated to the quantity it guards
                                  (E[tau|x] sd vs 1e-3*cap)
  4. two copies of the model, one of which silently wins
                                  (the compiled evaluator vs the DAG)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.time_series.circuits import DegenerateModelError, SurvivalPC, WindowPC
from poc.time_series.data import make_ad_task, make_rul_task
from src.probabilistic_circuits import (
    GaussianLeaf,
    GaussianMixtureLeaf,
    InputNode,
    RegionGraphPC,
    relative_sigma_floor,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shape 1 — one flag, two effects
#
# `--floor legacy` switched the init AND the runtime bound, so the control arm
# was a regime that had never existed and the whole capacity table was
# unreadable.  `leaf_components=1` switches the leaf CLASS, the init RULE and
# the component COUNT together, so "does a second component help?" is still
# unanswerable (§A.4).  Same shape, found twice, five weeks apart.
# ═══════════════════════════════════════════════════════════════════════════

def test_floor_flag_switches_the_runtime_bound_only():
    """Regression on the fixed case, kept here as the reference example of
    what a one-thing-at-a-time flag looks like when it is right."""
    vals = np.r_[np.full(400, 1.0),
                 np.random.default_rng(1).normal(1.0, 0.5, 112)]
    X = vals.reshape(-1, 1)
    seen = {}
    for relative in (True, False):
        GaussianLeaf.use_relative_floor = relative
        try:
            leaf = GaussianLeaf(0)
            leaf.fit(X)
            seen[relative] = (float(leaf.sigma), float(leaf.sigma_floor))
        finally:
            GaussianLeaf.use_relative_floor = True

    (s_rel, f_rel), (s_leg, f_leg) = seen[True], seen[False]
    assert f_rel != f_leg, "the flag must change the runtime floor"
    assert s_rel == pytest.approx(s_leg, rel=0.30), (
        f"the flag also moved the initial width ({s_leg:.3e} -> {s_rel:.3e}) "
        "by more than the documented 1.25x-at-the-floor asymmetry: it is "
        "switching two things again")


def test_mixture_at_1_makes_the_leaf_arms_comparable():
    """
    The §A.4 confound and its fix, both pinned.

    By DEFAULT c=1 and c=2 differ in the leaf class and the init rule as well
    as the component count — that is deliberate and must not change, because
    every recorded 1-component number was produced by `GaussianLeaf`.  What
    `mixture_at_1=True` adds is an arm where the ONLY difference is the count,
    which is the arm that can actually answer "does a second component help?"
    """
    default1 = WindowPC(4, 3, leaf_components=1)._leaf_factory()(0)
    default2 = WindowPC(4, 3, leaf_components=2)._leaf_factory()(0)
    assert type(default1) is not type(default2), (
        "the default arms are now the same class, which silently moves every "
        "recorded 1-component number")

    opt1 = WindowPC(4, 3, leaf_components=1, mixture_at_1=True)._leaf_factory()(0)
    assert type(opt1) is type(default2), (
        "mixture_at_1 does not produce the same leaf class as c=2, so the arm "
        "still cannot isolate the component count")
    assert opt1.n_components == 1


def test_single_component_mixture_centres_on_the_median():
    """np.linspace(0.1, 0.9, 1) is [0.1], not [0.5] — which would have put the
    n=1 mixture's only Gaussian on the 10th percentile and rigged the very arm
    added above.  Fixed 2026-08-05; this keeps it fixed."""
    vals = np.random.default_rng(0).normal(5.0, 2.0, 4096)
    leaf = GaussianMixtureLeaf(0, n_components=1)
    leaf.fit(vals.reshape(-1, 1))
    assert float(leaf.mus[0]) == pytest.approx(float(np.median(vals)), abs=0.2)


def test_relative_floor_flag_is_process_global():
    """
    Not a bug, a HAZARD, pinned so it is at least known: the flag is a CLASS
    attribute, so setting it anywhere changes every leaf fitted afterwards in
    the process — including inside another experiment in the same batch.  What
    saves the already-fitted circuits is that the floor is a BUFFER, copied at
    fit time; that property is what this test protects.
    """
    vals = np.random.default_rng(0).normal(0.0, 1.0, 512).reshape(-1, 1)
    GaussianLeaf.use_relative_floor = True
    try:
        early = GaussianLeaf(0)
        early.fit(vals)
        before = float(early.sigma_floor)

        GaussianLeaf.use_relative_floor = False        # another experiment
        late = GaussianLeaf(0)
        late.fit(vals)

        assert float(early.sigma_floor) == before, (
            "flipping the class flag retroactively changed an ALREADY FITTED "
            "leaf — every circuit alive in the process is affected and no "
            "batch that mixes the two modes is interpretable")
        assert float(late.sigma_floor) != before, "the flag stopped working"
    finally:
        GaussianLeaf.use_relative_floor = True


# ═══════════════════════════════════════════════════════════════════════════
# Shape 2 — a diagnostic written against a literal, not against the object
#
# `n_floor = (sigma < 1e-3).sum()` in bench_rul_leaves counts an ABSOLUTE
# bound.  Under --floor relative every leaf's floor is max(0.01*std, 1e-3),
# which is >= 1e-3 by construction, so the counter is structurally 0 and the
# zeros in the recorded table are not evidence of anything (§A.6).
# ═══════════════════════════════════════════════════════════════════════════

def leaves_at_their_own_floor(circuit, tol: float = 1.10) -> int:
    """The mode-independent replacement: count leaves pressed against THEIR
    OWN floor buffer.  Works under either flag, which is the whole point."""
    n = 0
    for m in circuit.modules():
        if isinstance(m, GaussianLeaf):
            n += int(float(m.sigma) <= tol * float(m.sigma_floor))
        elif isinstance(m, GaussianMixtureLeaf):
            n += int((m.sigmas <= tol * m.sigma_floor).sum())
    return n


def test_absolute_collapse_counter_is_blind_under_a_relative_floor():
    """
    Build the situation the bench reports on — leaves driven onto their floor
    in relative mode — and show the two counters disagree completely.  This is
    §A.6 demonstrated rather than asserted: the recorded `@floor = 0` column
    means "the counter cannot fire", not "nothing collapsed".
    """
    vals = np.random.default_rng(0).normal(0.0, 1.0, 512).reshape(-1, 1)
    GaussianLeaf.use_relative_floor = True
    try:
        leaf = GaussianLeaf(0)
        leaf.fit(vals)
        opt = torch.optim.Adam(leaf.parameters(), lr=0.5)
        for _ in range(800):                    # drive sigma down, hard
            opt.zero_grad()
            leaf.sigma.backward()
            opt.step()
    finally:
        GaussianLeaf.use_relative_floor = True

    sigma, floor = float(leaf.sigma), float(leaf.sigma_floor)
    absolute = int(sigma < 1e-3)                # what the bench counts
    relative = int(sigma <= 1.10 * floor)       # what it should count
    print(f"\n[floor counter] sigma {sigma:.3e}, own floor {floor:.3e} -> "
          f"absolute-1e-3 counter {absolute}, own-floor counter {relative}")
    assert relative == 1, "the leaf did not reach its floor; test is void"
    assert absolute == 0, (
        "the absolute counter fired, so this feature's relative floor is "
        "below 1e-3 and the demonstration needs a different column")


def test_own_floor_counter_works_in_both_modes():
    """The replacement must not have the defect it replaces."""
    task = make_ad_task(window=4, stride=3, seed=0, n_units=8, n_channels=4,
                        n_regimes=2)
    for relative in (True, False):
        GaussianLeaf.use_relative_floor = relative
        try:
            pc = WindowPC(task.window, task.n_channels, vtree_method="chain",
                          n_sum_components=3, seed=0, device="cpu")
            pc.fit(task.X_train, epochs=3)
            n = leaves_at_their_own_floor(pc.pc)
        finally:
            GaussianLeaf.use_relative_floor = True
        assert 0 <= n <= pc.d, f"counter returned {n} in relative={relative}"


# ═══════════════════════════════════════════════════════════════════════════
# Shape 3 — a guardrail whose threshold is unrelated to what it guards
#
# `SurvivalPC.predict` refuses a predictive whose sd is below 1e-3 * cap =
# 0.13 cycles.  The TARGET's own sd is ~35 cycles.  So a predictive that
# explains essentially nothing — sd 4 cycles, the measured `tau_where='root'`
# behaviour at K=8 — sails through and gets reported.
# ═══════════════════════════════════════════════════════════════════════════

def test_degeneracy_threshold_is_far_below_the_targets_own_spread():
    task = make_rul_task(window=6, stride=4, seed=0, n_units=24, n_channels=5,
                         n_regimes=2, n_bins=10, cap=100.0, censor_frac=0.3)
    target_sd = float(task.rul_test.std())
    threshold = 1e-3 * task.cap
    print(f"\n[guardrail] refuses below {threshold:.3f} cycles; the target's "
          f"own sd is {target_sd:.1f} cycles ({target_sd / threshold:.0f}x)")
    assert target_sd / threshold > 50, (
        "the guardrail threshold is now within striking distance of the "
        "target's spread, so this concern has been addressed")


def test_a_nearly_constant_predictive_passes_the_guardrail():
    """
    Concrete version of the same point.  A predictive with 5% of the target's
    spread is useless — R^2 near zero — and is accepted without complaint.
    The guardrail catches TOTAL collapse only; it is not a quality check, and
    nothing downstream is.
    """
    task = make_rul_task(window=6, stride=4, seed=0, n_units=24, n_channels=5,
                         n_regimes=2, n_bins=10, cap=100.0, censor_frac=0.3)
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method="chain", n_sum_components=4,
                    tau_where="deep", seed=0, device="cpu")
    pc.fit(task.X_train, task.tau_train, task.delta_train, epochs=3)

    n = len(task.X_test)
    mean = float(task.rul_test.mean())
    # a predictive that barely moves: uniform, tilted by +-0.05 between the
    # outermost bins, i.e. E[tau|x] swinging about +-4.5 of 100 cycles
    weak = torch.full((n, task.n_bins), 1.0 / task.n_bins)
    tilt = 0.05 * torch.linspace(-1.0, 1.0, n)
    weak[:, 0] += tilt
    weak[:, -1] -= tilt
    pc.log_pmf = lambda X, **kw: torch.log(weak[: len(X)])

    out = pc.predict(task.X_test)                     # no exception
    ratio = float(out["mean"].std()) / float(task.rul_test.std())
    print(f"\n[guardrail] accepted a predictive with {ratio:.3%} of the "
          f"target's spread (mean target {mean:.0f} cycles)")
    assert ratio < 0.10, "the fake predictive is not weak enough to make the point"


# ═══════════════════════════════════════════════════════════════════════════
# Shape 4 — two copies of the model, one of which silently wins
#
# After `fit`, RegionGraphPC.log_prob routes through the COMPILED evaluator,
# which holds its own parameter tensors.  `write_back()` syncs compiled -> DAG;
# there is no DAG -> compiled sync.  So anything that edits the DAG after a fit
# — a calibration pass, a pruning experiment, a leaf-surgery diagnostic — is
# silently ignored, with correct-looking results.  Same family as `.to()`
# being exponential: a convenience that quietly does the wrong thing.
# ═══════════════════════════════════════════════════════════════════════════

def test_dag_edits_do_not_reach_the_compiled_copy():
    task = make_ad_task(window=4, stride=3, seed=0, n_units=10, n_channels=4,
                        n_regimes=2)
    pc = WindowPC(task.window, task.n_channels, vtree_method="chain",
                  n_sum_components=3, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=5)
    assert pc.compiled is not None, "this build did not compile; test is void"

    before = float(pc.score(task.X_train[:64]).mean())
    with torch.no_grad():
        for m in pc.pc.modules():
            if isinstance(m, GaussianLeaf):
                m.log_sigma.fill_(6.0)            # sigma ~ 400: destroy the fit
    after = float(pc.score(task.X_train[:64]).mean())
    assert after == pytest.approx(before, rel=1e-6), (
        "the DAG edit reached the score, so the compiled copy is being kept "
        "in sync and this hazard is gone")

    pc.pc.use_recursive()                          # the documented escape
    assert float(pc.score(task.X_train[:64]).mean()) != pytest.approx(before,
                                                                      rel=1e-3)


def test_move_circuit_is_the_only_safe_device_move():
    """`.to()` on a region-graph circuit recurses over children() with no
    memoisation and is exponential in depth (0.2 s -> 400 s, silently, with
    correct results).  The project's answer is a convention in a gotcha list;
    this pins at least that the safe helper exists and is what the models use."""
    import inspect

    from src.probabilistic_circuits import move_circuit_
    src = inspect.getsource(WindowPC.fit) + inspect.getsource(SurvivalPC.fit)
    assert src.count("move_circuit_(self.pc") == 2, (
        "a fit path stopped using move_circuit_ to place its circuit")
    for bad in ("self.pc.to(", "self.pc = self.pc.to", "pc.to(self.device)"):
        assert bad not in src, f"a circuit is being moved with {bad!r} again"


def test_default_leaf_factory_floors_its_width():
    """
    §A.7, fixed 2026-08-05.  `InputNode` is the DEFAULT leaf_factory for
    RegionGraphPC/DensityPC, and until now it was the one leaf with no width
    floor — so every path outside the RUL task was exposed to the
    sigma ~ 1.5e-6 -> NaN failure that GaussianLeaf had already fixed.
    """
    # a median-constant column: exactly one whole sensor channel of every
    # C-MAPSS subset looks like this (30/450 features on FD001)
    vals = np.r_[np.full(400, 1.0),
                 np.random.default_rng(1).normal(1.0, 0.5, 112)]
    leaf = InputNode(0)
    leaf.fit(vals.reshape(-1, 1))
    sigma = float(leaf.sigma)
    assert sigma >= relative_sigma_floor(vals) * 0.99, (
        f"InputNode fitted sigma = {sigma:.3e} on a MAD-zero column; the "
        f"relative floor would be {relative_sigma_floor(vals):.3e}")


def test_zero_weight_jitter_is_refused():
    """Degeneracy #2, refused at construction rather than documented in a
    gotcha list.  The escape hatch (`allow_zero_jitter`) exists so a test can
    still build the degenerate circuit deliberately."""
    task = make_ad_task(window=4, stride=3, seed=0, n_units=8, n_channels=4,
                        n_regimes=2)
    with pytest.raises((ValueError, DegenerateModelError)):
        WindowPC(task.window, task.n_channels, vtree_method="chain",
                 n_sum_components=3, weight_jitter=0.0, seed=0,
                 device="cpu").fit(task.X_train, epochs=2)
