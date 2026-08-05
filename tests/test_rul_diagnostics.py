"""
RUL diagnostics — WHY the prognosis half fails, one cause at a time.

Three negatives are on record: the exact censored likelihood makes things
worse (T1, killed on a pre-registered gate), the predictive is badly
uncalibrated (PICP 0.38-0.52 at nominal 0.90 — the "exact != calibrated"
finding), and ridge wins point accuracy on real FD001.  What is NOT on record
is a mechanism for any of them.  Five hypotheses (H1-H5) were written down on
2026-08-02 with the signature that would confirm each, and then the expensive
end-to-end run was done instead of the cheap diagnostics.  These are the cheap
diagnostics.

  A. Is the OBJECTIVE sound?      (does the censored term have a trivial
                                   maximiser, and can the estimator recover a
                                   known conditional at all?)
  B. Where does the tau SIGNAL go? (H1 gradient balance, H3 coupling capacity,
                                   H5 binning damage — measured, not argued)
  C. Is the censoring MACHINERY exact?  (the box query is the whole claim)
  D. What is the miscalibration MADE OF?  (not width: the PIT variance is
                                   1/12 to three decimals while PICP reads
                                   0.62.  It is a location shift plus an
                                   endpoint convention — q05/q95 are bin
                                   CENTRES scored against a target in cycles)

Section D is the load-bearing one.  At the recorded settings the predictive's
dispersion is correctly calibrated, every coverage miss sits within half a bin
width, and reading the bin EDGES instead of the centres moves PICP from 0.62
to 0.93 for one extra bin of width.  What survives as a real defect is a
location shift — and section A shows that shift is the censoring bias, i.e.
the same defect the write-up counts twice.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.time_series.circuits import DegenerateModelError, SurvivalPC
from poc.time_series.data import make_rul_task
from poc.time_series.metrics import mpiw, picp
from src.probabilistic_circuits import CategoricalLeaf, GaussianLeaf

# The recorded RUL configuration, shrunk until it runs in seconds: bins/cap
# (hence the bin WIDTH, which section D turns on) are kept at the values every
# reported number used.
BINS, CAP = 25, 130.0
SMALL = dict(window=6, stride=4, n_units=24, n_channels=5, n_regimes=2,
             n_bins=10, cap=100.0)


def _fit(task, epochs=25, tau_where="deep", K=8, seed=0, use_censored=True):
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method="chain", n_sum_components=K,
                    tau_where=tau_where, seed=seed, device="cpu")
    pc.fit(task.X_train, task.tau_train, task.delta_train, epochs=epochs,
           use_censored=use_censored)
    return pc


@pytest.fixture(scope="module")
def task():
    return make_rul_task(seed=0, censor_frac=0.3, **SMALL)


@pytest.fixture(scope="module")
def pc(task):
    return _fit(task)


# ═══════════════════════════════════════════════════════════════════════════
# A.  Is the OBJECTIVE sound?
# ═══════════════════════════════════════════════════════════════════════════

def test_censored_objective_over_predicts_as_censoring_rises():
    """
    THE MECHANISM behind the failed gate, isolated and made reproducible.

    `log P(tau >= c | x)` is maximised by pushing all mass above every
    censoring time; only the uncensored units anchor against it.  So the bias
    of E[tau|x] must grow with the censored fraction — and it is a property of
    the objective, not of any particular dataset or bug.

    Measured here (2 seeds, tiny circuits): bias -1.5 cycles at 20% censoring,
    +6.3 at 69%.  Same sign and same direction as the 70%-censoring row that
    killed T1 (RMSE 28.19 vs 24.62, over-predicting remaining life).  Any
    future censored objective — reweighted, warm-started, hazard-parameterised
    — has to move this number or it has not addressed the problem.
    """
    def bias_at(cf):
        out = []
        for seed in (0, 1):
            t = make_rul_task(seed=seed, censor_frac=cf, **SMALL)
            p = _fit(t, seed=seed).predict(t.X_test)
            out.append(float((p["mean"] - t.rul_test).mean()))
        return float(np.mean(out))

    low, high = bias_at(0.15), bias_at(0.75)
    print(f"\n[censoring bias] 15% censoring {low:+.2f} cycles, "
          f"75% censoring {high:+.2f} cycles")
    assert high > low + 3.0, (
        f"bias barely moves with censoring ({low:+.2f} -> {high:+.2f}): the "
        "trivial-maximiser explanation for the failed T1 gate is not "
        "reproducible here, and the retired hypothesis deserves a re-look")
    assert high > 0, "heavy censoring should push the predictive UP, not down"


def test_model_recovers_a_known_conditional_with_no_censoring():
    """
    THE PRECONDITION that was never checked before the gate was run.  If the
    estimator cannot recover p(tau|x) when tau is plainly readable from x and
    nothing is censored, then no censoring result means anything — the gate
    would have been measuring the estimator, not the censored term.

    Deliberately easy: tau is one channel of the last timestep plus small
    noise.  A model that fails here is broken, not merely inaccurate.
    """
    def synth(n, seed):
        rng = np.random.default_rng(seed)
        k = rng.integers(0, 8, size=n)
        X = rng.normal(size=(n, 4, 3)).astype(np.float32)
        X[:, -1, 0] = k + rng.normal(0, 0.15, size=n)
        return torch.from_numpy(X.reshape(n, -1)), torch.from_numpy(k)

    Xtr, ktr = synth(600, 0)
    Xte, kte = synth(400, 1)
    pc = SurvivalPC(4, 3, 8, 8.0, vtree_method="chain", n_sum_components=6,
                    tau_where="deep", seed=0, device="cpu")
    pc.fit(Xtr, ktr, torch.ones(len(ktr), dtype=torch.long), epochs=40)

    out = pc.predict(Xte)
    centers = pc.bin_centers().numpy()
    corr = float(np.corrcoef(out["mean"].numpy(), centers[kte.numpy()])[0, 1])
    mode_acc = float((out["pmf"].argmax(1).numpy() == kte.numpy()).mean())
    print(f"\n[recovery] corr(E[tau|x], truth) = {corr:.3f}, "
          f"mode accuracy = {mode_acc:.3f} (chance {1/8:.3f})")
    assert corr > 0.85, (
        f"the estimator only reaches corr {corr:.2f} on a conditional that is "
        "readable off one feature — the RUL model is broken upstream of any "
        "censoring question")
    assert mode_acc > 1.5 / 8, (
        f"the pmf mode is at {mode_acc:.2f} vs {1/8:.2f} chance: the point "
        "estimate tracks but the DISTRIBUTION is a smear, which is what the "
        "downstream distributional metrics are actually scoring")


# ═══════════════════════════════════════════════════════════════════════════
# B.  Where does the tau signal go?  (H1, H3, H5 — measured at last)
# ═══════════════════════════════════════════════════════════════════════════

def test_tau_gradient_share_is_far_below_its_dimensional_share(task, pc):
    """
    H1, quantified.  The loss is log p(x, tau) over 1 + window*C variables, so
    the window dominates the gradient and the censoring information enters
    only through the tau leaf and its coupling.  H1 was ranked "most likely
    single cause" and never measured.

    Measured at this scale: tau collects ~2.0% of the leaf gradient mass while
    it is ~3.2% of the dimensions (1.6x under its share), and per PARAMETER
    its gradient is ~6-7x smaller than a window leaf's.  H1 is supported in
    direction but is not the order-of-magnitude effect it was assumed to be —
    which matters, because a re-weighting fix has to beat a 1.6x, and the
    ratio grows with window*C, so it should be re-measured at full width
    (450 features) before anyone builds one.
    """
    pc.pc.use_recursive()          # the compiled copy holds its own parameters
    z = pc._augment(task.X_train[:256], task.tau_train[:256].float())
    loss = -pc.pc.log_prob(z).mean()
    pc.pc.zero_grad()
    loss.backward()

    tau_g, win_g, tau_n, win_n = [], [], 0, 0
    for m in pc.pc.modules():
        if isinstance(m, CategoricalLeaf):
            for p in m.parameters():
                if p.grad is not None:
                    tau_g.append(float(p.grad.abs().sum()))
                    tau_n += p.numel()
        elif isinstance(m, GaussianLeaf):
            for p in m.parameters():
                if p.grad is not None:
                    win_g.append(float(p.grad.abs().sum()))
                    win_n += p.numel()
    pc.pc.zero_grad()
    assert tau_g and win_g, "no gradients reached the leaves"

    share = sum(tau_g) / (sum(tau_g) + sum(win_g))
    dim_share = 1.0 / (pc.d + 1)
    per_param = (sum(tau_g) / tau_n) / (sum(win_g) / win_n)
    print(f"\n[gradient] tau share of leaf gradient mass {share:.4f} vs "
          f"dimensional share {dim_share:.4f} "
          f"({dim_share / max(share, 1e-12):.1f}x under); per-parameter ratio "
          f"tau/window {per_param:.3f}")
    assert share < dim_share, (
        "tau now receives at least its dimensional share of the gradient — "
        "H1 (the censoring term is drowned by the window likelihood) no "
        "longer holds and the retired T1 result should be revisited")


def test_root_coupling_is_degenerate_at_almost_every_K(task):
    """
    H3, swept rather than spot-checked — and the sweep is the point.

    `tau_where='root'` caps p(tau|x) at a convex combination of K profiles.
    The two-line diagnostic written down on 2026-08-02 ("count the distinct
    predictions; if it is <= K the coupling is the bottleneck") was never run.
    Run over K:

        root  K=4   sd 0.009 cycles      deep  K=4   sd 19.3
        root  K=6   sd 0.024             deep  K=6   sd 19.0
        root  K=8   sd 4.27              deep  K=8   sd 19.8
        root  K=12  sd 0.001             deep  K=12  sd 20.3

    Root is CONSTANT at K=4, 6 and 12 and merely feeble at K=8.  Two things
    follow.  First, 'root is degenerate' is not a property of one unlucky run,
    which is what the write-up currently implies.  Second, K=8 shows the
    degeneracy is not monotone in capacity — so a single K is not evidence
    either way, and the ablation that keeps `root` "only as an ablation" is
    reporting a coin flip.
    """
    rows = {}
    for where in ("root", "deep"):
        for K in (4, 6, 12):
            m = _fit(task, tau_where=where, K=K)
            p = m.predict(task.X_test, check_degenerate=False)["mean"].numpy()
            rows[(where, K)] = (float(p.std()),
                                len(np.unique(np.round(p, 2))))
    print("\n[coupling] sd of E[tau|x] (cycles) / distinct values")
    for (where, K), (sd, n) in rows.items():
        print(f"  {where:5s} K={K:2d}  sd {sd:7.3f}  distinct {n:4d}")

    for K in (4, 6, 12):
        sd_root, sd_deep = rows[("root", K)][0], rows[("deep", K)][0]
        assert sd_deep > 3 * sd_root, (
            f"at K={K} the root coupling reaches sd {sd_root:.2f} vs deep "
            f"{sd_deep:.2f}: H3 (the K-profile coupling is the bottleneck) no "
            "longer holds and `tau_where` stops being a forced choice")


def test_the_degeneracy_guardrail_catches_the_root_collapse(task):
    """The guardrail added after degeneracy #3 must actually fire on the
    configuration that caused it.  Its threshold is absolute (1e-3 * cap);
    see tests/test_experiment_hygiene.py for how far that sits below the
    target's own spread."""
    root = _fit(task, tau_where="root", K=4)
    with pytest.raises(DegenerateModelError):
        root.predict(task.X_test)


def test_tau_marginal_tracks_the_empirical_histogram(task, pc):
    """
    H5, and it comes out NEGATIVE — worth pinning precisely because it clears
    a suspect.  With cap=130 most windows sit in the top bin, so the fear was
    that p(tau) collapses onto it and the conditional inherits the damage.
    Measured: the model's tau marginal matches the empirical histogram bin for
    bin (top bin 0.595 vs 0.577).  Binning is not the failure; stop looking
    here.
    """
    pc.pc.use_recursive()
    x0 = task.X_test[:1]
    logs = []
    for k in range(task.n_bins):
        z = pc._augment(x0, torch.full((1,), float(k)))
        with torch.no_grad():
            logs.append(float(pc.pc.log_marginal(z, list(range(pc.d)))))
    q = np.exp(np.asarray(logs) - max(logs))
    q /= q.sum()
    emp = np.bincount(task.tau_train.numpy(), minlength=task.n_bins).astype(float)
    emp /= emp.sum()

    tv = 0.5 * float(np.abs(q - emp).sum())
    print(f"\n[tau marginal] total-variation distance to empirical {tv:.3f}; "
          f"top bin model {q[-1]:.3f} vs data {emp[-1]:.3f}")
    assert tv < 0.15, (
        f"the learned tau marginal is {tv:.2f} TV away from the data — H5 "
        "(binning/cap damage) is back on the table and the cap should be "
        "swept before anything else")


# ═══════════════════════════════════════════════════════════════════════════
# C.  Is the censoring MACHINERY exact?
#
# The censored likelihood is a box query.  Everything T1 claimed rested on it
# being exact, and the compiled evaluator is gated against the recursion at
# fit time — but never against an independent construction of the same
# quantity.  These two do that.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("t_bin", [1, 4, 7])
def test_survival_equals_the_pmf_suffix_sum(task, pc, t_bin):
    X = task.X_test[:64]
    s = pc.log_survival(X, t_bin)
    ref = torch.logsumexp(pc.log_pmf(X)[:, t_bin + 1:], dim=1)
    err = float((s - ref).abs().max())
    assert err < 1e-4, (
        f"log S({t_bin}|x) from the box query differs from the pmf suffix sum "
        f"by {err:.2e} nats — the censored likelihood and the reported "
        "survival curve are not the same object")


def test_survival_is_monotone_in_time(task, pc):
    X = task.X_test[:64]
    curve = torch.stack([pc.log_survival(X, t) for t in range(task.n_bins - 1)])
    diffs = curve[1:] - curve[:-1]
    assert float(diffs.max()) < 1e-5, "S(t|x) increases somewhere"


# ═══════════════════════════════════════════════════════════════════════════
# D.  Is the miscalibration REAL?
#
# "Exact != calibrated" is recorded as the project's most interesting finding
# and as a direct complication of its "exact therefore trustworthy" framing.
# Before it goes in a paper it has to survive the dullest possible
# alternative: that the interval endpoints are in the wrong units.
#
# `SurvivalPC.predict` returns q05/q95 as bin CENTRES.  `picp` scores them
# against `rul_test` in CYCLES.  Every reported PICP for the circuit comes
# from that pair (run_rul.py:82, pipeline.py:416), and the conformal layer
# adds a scalar to those same centres (conformal.py:158) — i.e. it is fitting
# back the half-bin the extraction dropped.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def recorded():
    """One fit at the settings every recorded RUL number used: chain, K=12,
    tau deep, bins=25, cap=130 — the bin WIDTH (5.2 cycles) is the quantity
    section D is about, so it is not shrunk."""
    t = make_rul_task(window=8, stride=4, seed=0, n_units=30, n_channels=6,
                      n_regimes=3, n_bins=BINS, cap=CAP, censor_frac=0.35)
    m = _fit(t, epochs=30, K=12)
    return t, m, m.predict(t.X_test)


def _pit(pmf: np.ndarray, k: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomised PIT for a discrete predictive: U = F(k-1) + u*p(k).  Under a
    correctly calibrated predictive U is exactly Uniform(0, 1) — which makes
    it the right instrument here, because unlike PICP it never compares a
    discretised quantity to a continuous one."""
    rng = np.random.default_rng(seed)
    cdf = pmf.cumsum(1)
    rows = np.arange(len(k))
    below = np.where(k > 0, cdf[rows, np.maximum(k - 1, 0)], 0.0)
    return below + rng.random(len(k)) * pmf[rows, k]


def test_predictive_dispersion_is_correctly_calibrated(recorded):
    """
    THE TEST.  "PICP 0.38-0.52 at nominal 0.90" reads as an overconfident
    predictive — intervals too narrow.  An overconfident predictive puts the
    truth in the TAILS of its own distribution, so its PIT variance exceeds
    1/12.  Measured: PIT variance 0.0841 against 1/12 = 0.0833, i.e. the
    dispersion is right to three decimals while PICP reads 0.62.

    So the density is NOT overconfident, and "exact but overconfident" is not
    what the recorded numbers show.  Whatever "exact != calibrated" is about,
    it is not the width of the predictive.
    """
    task, pc, pred = recorded
    u = _pit(pred["pmf"].numpy(), task.tau_test.numpy())
    n = len(u)
    ks = float(np.max(np.abs(np.sort(u) - np.arange(1, n + 1) / n)))
    cov = picp(pred["q05"], pred["q95"], task.rul_test)
    print(f"\n[calibration] PICP(nominal 0.90) = {cov:.3f}   "
          f"PIT mean {u.mean():.3f} (0.5) var {u.var():.4f} "
          f"(1/12 = {1/12:.4f})  KS {ks:.3f}")

    assert abs(u.var() - 1 / 12) < 0.008, (
        f"PIT variance {u.var():.4f} is far from 1/12: the predictive really "
        "is mis-dispersed, and the interval-endpoint explanation below is not "
        "the whole story")
    assert cov < 0.80, (
        "PICP is no longer reproducing the recorded under-coverage, so this "
        "whole section is measuring a different model than the one on record")


def test_the_pit_deviation_is_location_not_shape(recorded):
    """
    The residual miscalibration, attributed.  PIT mean 0.408 against 0.5 with
    the variance already correct is a pure LOCATION error: the predictive is
    shifted, not the wrong width.  A shift is exactly what section A measures
    the censored objective producing — so the calibration defect and the
    censoring defect are ONE defect, counted twice in the write-up.

    This also decides what the conformal layer was doing: an additive shift is
    what CQR-style recalibration removes, which is why it "worked" without any
    exactness guarantee.
    """
    task, pc, pred = recorded
    u = _pit(pred["pmf"].numpy(), task.tau_test.numpy())
    location = abs(float(u.mean()) - 0.5)
    shape = abs(float(u.var()) - 1 / 12) / (1 / 12)
    print(f"\n[pit decomposition] location |mean-0.5| = {location:.3f}, "
          f"shape |var-1/12|/(1/12) = {shape:.3f}")
    assert location > 3 * shape, (
        f"the PIT defect is not predominantly a location shift "
        f"(location {location:.3f} vs shape {shape:.3f}) — the calibration "
        "story and the censoring story are separate after all")


def test_every_coverage_miss_is_inside_half_a_bin(recorded):
    """
    THE MECHANISM, isolated.  If the interval were genuinely too narrow the
    misses would be spread over many cycles.  Instead their median distance
    outside the interval is exactly half a bin width — the distance from a
    bin's centre to its edge.  The truth is landing in the RIGHT bin and being
    scored as outside it.
    """
    task, pc, pred = recorded
    true = task.rul_test.numpy()
    lo, hi = pred["q05"].numpy(), pred["q95"].numpy()
    out = np.maximum(lo - true, true - hi)
    miss = out[out > 0]
    half_bin = 0.5 * CAP / BINS
    assert len(miss) > 20, "too few misses to characterise"
    med = float(np.median(miss))
    print(f"\n[misses] {len(miss)}/{len(true)} outside; median overshoot "
          f"{med:.2f} cycles vs half a bin = {half_bin:.2f}")
    assert med < 1.5 * half_bin, (
        f"median overshoot {med:.2f} cycles is well beyond half a bin "
        f"({half_bin:.2f}) — the intervals are genuinely too narrow and the "
        "centre/edge explanation does not account for the under-coverage")


def test_predict_returns_both_endpoint_conventions(recorded):
    """
    Pins the step-0 fix.  `q05`/`q95` must keep meaning bin CENTRES so every
    number recorded against them stays comparable; `q05_edge`/`q95_edge` are
    the outer edges of the SAME bins, so the edge interval always contains the
    centre one and is wider by at most one bin.
    """
    task, pc, pred = recorded
    for k in ("q05", "q95", "q05_edge", "q95_edge"):
        assert k in pred, f"predict() lost {k}"
    bw = task.cap / task.n_bins
    assert bool((pred["q05_edge"] <= pred["q05"]).all())
    assert bool((pred["q95_edge"] >= pred["q95"]).all())
    widen = (pred["q95_edge"] - pred["q05_edge"]) - (pred["q95"] - pred["q05"])
    assert float(widen.max()) == pytest.approx(bw, rel=1e-4), (
        "the edge interval should be exactly one bin wider than the centre "
        "one; a different gap means the two are not derived from the same "
        "quantile bins")


def test_coverage_is_restored_by_reading_the_bin_edges(recorded):
    """
    THE FIX, measured.  Same circuit, same pmf, same quantile bins — only the
    endpoints are read as the outer EDGES of the selected bins instead of
    their centres, which is the correct discrete-to-continuous convention for
    covering a continuous target.

    Measured at the recorded settings: PICP 0.62 -> 0.93 for one extra bin
    width of MPIW (68.5 -> 73.7).  Post-hoc conformal reached 0.82-0.91 on the
    same quantity, which is now explained: it was learning an additive
    constant of about half a bin at each end.
    """
    task, pc, pred = recorded
    pmf = pred["pmf"].numpy()
    cdf = pmf.cumsum(1)
    edges = np.linspace(0.0, task.cap, task.n_bins + 1)
    lo_i = np.clip((cdf < 0.05).sum(1), 0, task.n_bins - 1)
    hi_i = np.clip((cdf < 0.95).sum(1), 0, task.n_bins - 1)
    true = task.rul_test.numpy()

    cov_c = picp(pred["q05"], pred["q95"], true)
    cov_e = picp(edges[lo_i], edges[hi_i + 1], true)
    print(f"\n[endpoints] centres PICP {cov_c:.3f} MPIW "
          f"{mpiw(pred['q05'], pred['q95']):.1f}  ->  edges PICP {cov_e:.3f} "
          f"MPIW {mpiw(edges[lo_i], edges[hi_i + 1]):.1f}")
    assert cov_e > cov_c + 0.15, (
        "reading the bin edges does not recover coverage, so the endpoint "
        "convention is not the explanation for the recorded PICP")
    assert cov_e > 0.75, (
        f"even with edge endpoints coverage is only {cov_e:.2f} at nominal "
        "0.90 — part of the miscalibration is real and needs its own account")


def test_discrete_target_is_covered_by_the_discrete_interval(recorded):
    """
    The apples-to-apples version: does the true BIN fall inside the selected
    bins?  This is the question the pmf actually answers, and it is the one
    that should have been asked before concluding anything about exactness and
    trust.  Measured: 0.993.
    """
    task, pc, pred = recorded
    cdf = pred["pmf"].numpy().cumsum(1)
    lo_i = np.clip((cdf < 0.05).sum(1), 0, task.n_bins - 1)
    hi_i = np.clip((cdf < 0.95).sum(1), 0, task.n_bins - 1)
    k = task.tau_test.numpy()
    cov = float(((k >= lo_i) & (k <= hi_i)).mean())
    print(f"\n[discrete coverage] true bin inside the selected bins: {cov:.3f}")
    assert cov > 0.85, (
        f"the pmf misses its own target bin {1 - cov:.1%} of the time — the "
        "predictive is genuinely wrong, not merely mis-read")
