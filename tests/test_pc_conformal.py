"""
E-value conformal layer — validity, and the ways it can be valid-but-worthless.

Two kinds of test here, and the second kind is the point.

  A. VALIDITY.  The guarantees are finite-sample and distribution-free, so they
     are testable directly rather than asymptotically: a valid e-value has mean
     <= 1, a conformal p-value is super-uniform, and a 1-alpha set covers at
     least 1-alpha of exchangeable draws.  These are checked on synthetic
     exchangeable data with no circuit involved, because the guarantee must not
     depend on the model being any good.

  B. THE FAILURE MODES THAT STILL LOOK CORRECT.  A conformal wrapper reports
     textbook coverage on top of a degenerate model, a leaked split, or a
     threshold set by one calibration point.  Those are the outcomes that get
     published, so each one has a test that asserts the layer REFUSES rather
     than returns:

       - degenerate conditional            -> DegenerateModelError
       - alpha below the resolution floor  -> ValueError
       - evaluator changed between cal/test-> RuntimeError
       - sum domination by one cal unit    -> measured (sum_concentration)

The censored-bound tests are the ones that matter most for the claim this file
exists to support: using right-censored units via the exact box query must be
CONSERVATIVE relative to dropping them, never anti-conservative.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.time_series.circuits import DegenerateModelError
from poc.time_series.pc_conformal import (
    ConformalSurvivalBound,
    ConformalTestMartingale,
    CoveragePolicy,
    EvalueICAD,
    EvalueMerge,
    EvaluePredictionSet,
    alpha_floor,
    assert_conditional_varies,
    e_prediction_set,
    e_to_alarm,
    assert_non_overlapping,
    hpd_scores,
    martingale_lead_times,
    naive_product_alarm_rate,
    online_conformal_pvalues,
    p_prediction_set,
    p_to_e,
    reduce_by_unit,
    select_lambda_for_target_size,
    split_units_three,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fakes — the layer is duck-typed on SurvivalPC, so validity is testable
# without fitting a circuit (17 s suites, not 17 min)
# ═══════════════════════════════════════════════════════════════════════════

class FakeSurvivalPC:
    """Supplies exactly the four members the wrapper touches."""

    def __init__(self, log_pmf: np.ndarray, cap: float = 130.0):
        self._log_pmf = np.asarray(log_pmf, dtype=float)
        self.n_bins = self._log_pmf.shape[1]
        self.cap = cap
        self.compiled = None
        self.evaluator = "recursive"

    def log_pmf(self, X):
        idx = np.asarray(X).reshape(len(X), -1)[:, 0].astype(int)
        return torch.from_numpy(self._log_pmf[idx])

    def bin_edges(self):
        return torch.linspace(0, self.cap, self.n_bins + 1)

    def bin_centers(self):
        e = self.bin_edges()
        return 0.5 * (e[:-1] + e[1:])


def peaked_log_pmf(n: int, n_bins: int, seed: int = 0,
                   sharpness: float = 3.0) -> np.ndarray:
    """Conditionals that genuinely differ across samples (varying mode AND width)."""
    rng = np.random.default_rng(seed)
    ks = np.arange(n_bins)
    modes = rng.integers(0, n_bins, size=n)
    widths = rng.uniform(0.8, 3.0, size=n)
    logits = -sharpness * ((ks[None, :] - modes[:, None]) / widths[:, None]) ** 2
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def sample_from(log_pmf: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = np.exp(log_pmf)
    P = P / P.sum(axis=1, keepdims=True)
    return np.array([rng.choice(len(p), p=p) for p in P])


def index_batch(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).reshape(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# A. Validity
# ═══════════════════════════════════════════════════════════════════════════

def test_p_to_e_has_expectation_exactly_one():
    """The defining property, checked by QUADRATURE rather than sampling — the
    identity is exact, and for small kappa no sample mean can demonstrate it
    (see the variance test below)."""
    # LOG-spaced: f(p) = kappa*p^(kappa-1) is singular at 0, so a uniform grid
    # puts ~200 units of spurious mass in the first cell. The exact truncated
    # integral on [eps, 1] is 1 - eps^kappa, which is what is checked.
    eps = 1e-9
    grid = np.logspace(np.log10(eps), 0.0, 400_001)
    for kappa in (0.2, 0.5, 0.8):
        got = np.trapezoid(p_to_e(grid, kappa), grid)
        assert abs(got - (1.0 - eps ** kappa)) < 1e-6


def test_p_to_e_concentrates_only_above_kappa_one_half():
    """E[f^2] = kappa^2 * int p^(2k-2) diverges for kappa <= 1/2, so the sample
    mean does not concentrate there. Validity is untouched — but `mean_e` in
    `diagnose` is noisy at small kappa, and this is why."""
    p = np.random.default_rng(0).uniform(0, 1, 200_000)
    assert abs(p_to_e(p, 0.8).mean() - 1.0) < 0.01        # finite variance
    assert abs(p_to_e(p, 0.6).mean() - 1.0) < 0.03
    heavy = p_to_e(p, 0.2)
    assert heavy.max() > 1e3 * heavy.mean()               # the heavy tail itself


def test_p_to_e_rejects_kappa_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            p_to_e(np.array([0.5]), bad)


def test_conformal_p_values_are_super_uniform():
    """
    P(p <= a) <= a for exchangeable data — the finite-sample guarantee.

    Averaged over CALIBRATION DRAWS, because that is what conformal promises.
    Conditional on one frozen calibration set the rate is Beta-distributed
    around a with sd ~ sqrt(a(1-a)/n) — at n=300 and a=0.05 that is 0.013, so a
    single-split test at tolerance 0.02 rejects a correct implementation
    roughly a third of the time.  Marginal validity is the claim; test it.
    """
    rng = np.random.default_rng(1)
    rates = {a: [] for a in (0.01, 0.05, 0.10, 0.25)}
    for _ in range(200):
        pool = rng.normal(size=500)
        icad = EvalueICAD().calibrate(pool[:300])
        p = icad.p_value(pool[300:])
        for a in rates:
            rates[a].append(np.mean(p <= a))
    for a, obs in rates.items():
        assert np.mean(obs) <= a + 0.01


def test_icad_alarm_rate_is_controlled_on_exchangeable_healthy_data():
    """Laxhammar's own diagnostic: the empirical alarm rate sits at or below
    nominal over many random orderings of the calibration/test split."""
    rng = np.random.default_rng(2)
    rates = []
    for rep in range(60):
        pool = rng.normal(size=400)
        icad = EvalueICAD().calibrate(pool[:200])
        rates.append(np.mean(icad.alarm(pool[200:], alpha=0.10)))
    assert np.mean(rates) <= 0.10 + 0.02
    assert max(rates) <= 0.35            # no ordering blows the rate up


def test_icad_diagnose_reports_mean_e_below_one_on_healthy_data():
    rng = np.random.default_rng(3)
    pool = rng.normal(size=800)
    icad = EvalueICAD().calibrate(pool[:400])
    d = icad.diagnose(pool[400:], alpha=0.10)
    assert d["mean_e"] <= 1.0 + 0.15
    assert d["exchangeability_ok"] == 1.0
    assert d["saturation"] < 0.5


def test_icad_detects_a_real_shift():
    """Validity is worthless without power: a shifted population must alarm."""
    rng = np.random.default_rng(4)
    icad = EvalueICAD().calibrate(rng.normal(size=400))
    anomalous = rng.normal(loc=4.0, size=200)
    assert np.mean(icad.alarm(anomalous, alpha=0.10)) > 0.8


def test_merged_e_value_stays_valid_under_perfect_dependence():
    """The merge must hold with NO independence assumption — so test it at the
    hardest case, two perfectly dependent statistics."""
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 100_000)
    e = p_to_e(p, 0.5)
    merged = EvalueMerge()(e, e)                 # identical, maximally dependent
    assert abs(merged.mean() - 1.0) < 0.05
    assert np.allclose(merged, e)

    mixed = EvalueMerge()(e, p_to_e(rng.uniform(0, 1, 100_000), 0.5))
    assert mixed.mean() <= 1.05


def test_merge_rejects_misaligned_and_invalid_weights():
    e = np.ones(10)
    with pytest.raises(ValueError):
        EvalueMerge()(e, np.ones(9))
    with pytest.raises(ValueError):
        EvalueMerge(weights=[0.7, 0.7])(e, e)


def test_hpd_scores_are_at_least_one_and_exactly_one_at_the_mode():
    L = peaked_log_pmf(50, 12, seed=6)
    S = hpd_scores(L)
    assert S.min() >= 1.0 - 1e-9
    assert np.allclose(S[np.arange(len(S)), L.argmax(axis=1)], 1.0)


def test_e_prediction_set_covers_at_least_one_minus_alpha():
    """The headline guarantee, on exchangeable draws from a known conditional."""
    n_bins = 15
    L = peaked_log_pmf(1200, n_bins, seed=7)
    y = sample_from(L, seed=8)
    S = hpd_scores(L)
    s_all = S[np.arange(len(y)), y]
    cal, test = slice(0, 400), slice(400, None)
    for alpha in (0.05, 0.10, 0.20):
        keep = e_prediction_set(s_all[cal], S[test], alpha)
        cov = keep[np.arange(keep.shape[0]), y[test]].mean()
        assert cov >= 1.0 - alpha


def test_prediction_sets_are_adaptive_when_the_conditional_is():
    """Sets must be NARROWER where the circuit is confident — the property that
    justifies a density model as the base of this wrapper at all."""
    n_bins = 20
    L = peaked_log_pmf(1500, n_bins, seed=9)
    y = sample_from(L, seed=10)
    S = hpd_scores(L)
    s_cal = S[np.arange(400), y[:400]]
    keep = e_prediction_set(s_cal, S[400:], 0.10)
    sizes = keep.sum(axis=1)
    entropy = -(np.exp(L[400:]) * L[400:]).sum(axis=1)
    lo, hi = entropy < np.quantile(entropy, 0.25), entropy > np.quantile(entropy, 0.75)
    assert sizes[lo].mean() < sizes[hi].mean()


def test_e_sets_are_wider_than_p_sets_and_both_are_valid():
    """The trade the write-up must state: e-sets pay width for post-hoc
    validity. If the e-set were ever SMALLER, the implementation is wrong."""
    n_bins = 15
    L = peaked_log_pmf(1200, n_bins, seed=30)
    y = sample_from(L, seed=31)
    S = hpd_scores(L)
    s_cal = S[np.arange(400), y[:400]]
    for alpha in (0.10, 0.20):
        keep_e = e_prediction_set(s_cal, S[400:], alpha)
        keep_p = p_prediction_set(s_cal, S[400:], alpha)
        cov_e = keep_e[np.arange(keep_e.shape[0]), y[400:]].mean()
        cov_p = keep_p[np.arange(keep_p.shape[0]), y[400:]].mean()
        assert cov_e >= 1 - alpha and cov_p >= 1 - alpha - 0.03
        assert keep_e.sum(axis=1).mean() >= keep_p.sum(axis=1).mean()


def test_report_exposes_the_p_fixed_baseline():
    n_bins = 12
    L = peaked_log_pmf(600, n_bins, seed=32)
    y = sample_from(L, seed=33)
    pc = FakeSurvivalPC(L)
    X = index_batch(600)
    eps = EvaluePredictionSet(pc, alpha=0.10).calibrate(X[:300], y[:300])
    rep = eps.report(X[300:], y[300:])
    assert rep["e_width_premium"] >= -1e-9          # e is never tighter
    assert rep["p_fixed_coverage"] >= 0.85


def test_unit_reduction_choice_changes_the_guarantee_and_is_recorded():
    """`max` gives a unit-level guarantee and near-vacuous sets; `random` gives
    window-level coverage. Mixing them silently in one table would be a
    misreport, so the choice is recorded."""
    n_bins = 12
    L = peaked_log_pmf(600, n_bins, seed=34)
    y = sample_from(L, seed=35)
    units = np.repeat(np.arange(60), 10)
    pc = FakeSurvivalPC(L)
    X = index_batch(600)

    wide = EvaluePredictionSet(pc, alpha=0.20, reduce="max")
    wide.calibrate(X, y, unit_ids=units)
    tight = EvaluePredictionSet(pc, alpha=0.20, reduce="random")
    tight.calibrate(X, y, unit_ids=units)

    assert wide.diagnostics["reduce"] == "max"
    assert tight.diagnostics["reduce"] == "random"
    assert wide.n_cal == tight.n_cal == 60
    assert wide.s_cal.mean() > tight.s_cal.mean()      # max is the worst window
    assert wide.report(X, y)["mean_size_bins"] >= tight.report(X, y)["mean_size_bins"]


def test_censored_calibration_rows_are_dropped_from_a_two_sided_set():
    """For a censored window `tau` is the CENSORING BOUND, not the label.
    Scoring it as the label puts a wrong value in the calibration quantile —
    a correctness bug the `delta_cal` argument exists to prevent."""
    n_bins = 12
    L = peaked_log_pmf(400, n_bins, seed=38)
    y = sample_from(L, seed=39)
    delta = np.ones(400, dtype=int)
    delta[:100] = 0                                  # 25% right-censored
    pc = FakeSurvivalPC(L)
    X = index_batch(400)

    naive = EvaluePredictionSet(pc, alpha=0.20).calibrate(X, y)
    fixed = EvaluePredictionSet(pc, alpha=0.20).calibrate(X, y, delta_cal=delta)
    assert naive.n_cal == 400 and fixed.n_cal == 300
    assert fixed.diagnostics["frac_censored_dropped"] == pytest.approx(0.25)
    assert fixed.diagnostics["censoring_handled"] == 1.0
    assert naive.diagnostics["censoring_handled"] == 0.0


def test_all_censored_calibration_points_at_the_one_sided_bound():
    """The set cannot be calibrated at all; the error must say what can."""
    L = peaked_log_pmf(200, 10, seed=40)
    pc = FakeSurvivalPC(L)
    eps = EvaluePredictionSet(pc, alpha=0.20)
    with pytest.raises(ValueError, match="ConformalSurvivalBound"):
        eps.calibrate(index_batch(200), sample_from(L, seed=41),
                      delta_cal=np.zeros(200, dtype=int))


def test_unknown_reduction_is_refused():
    L = peaked_log_pmf(50, 8, seed=36)
    pc = FakeSurvivalPC(L)
    eps = EvaluePredictionSet(pc, reduce="mean")
    with pytest.raises(KeyError, match="unknown reduce"):
        eps.calibrate(index_batch(50), sample_from(L, seed=37),
                      unit_ids=np.repeat(np.arange(10), 5))


# ═══════════════════════════════════════════════════════════════════════════
# B. The failure modes that still look correct
# ═══════════════════════════════════════════════════════════════════════════

def test_degenerate_conditional_is_refused_not_calibrated():
    """A conditional that ignores x yields identical, perfectly covered sets.
    The layer must refuse: this is the `leaf_components=1` cell."""
    flat = np.tile(np.log(np.full(10, 0.1)), (200, 1))
    with pytest.raises(DegenerateModelError, match="degenerate conditional"):
        assert_conditional_varies(flat)

    pc = FakeSurvivalPC(flat)
    eps = EvaluePredictionSet(pc, alpha=0.10)
    with pytest.raises(DegenerateModelError):
        eps.calibrate(index_batch(200), np.zeros(200, dtype=int))


def test_a_barely_varying_conditional_still_trips_the_gate():
    rng = np.random.default_rng(11)
    L = np.log(np.full((300, 10), 0.1)) + 1e-6 * rng.normal(size=(300, 10))
    L -= np.log(np.exp(L).sum(axis=1, keepdims=True))
    with pytest.raises(DegenerateModelError):
        assert_conditional_varies(L)


def test_healthy_conditional_passes_the_gate():
    assert assert_conditional_varies(peaked_log_pmf(200, 12, seed=12)) > 1e-3


def test_alpha_below_the_resolution_floor_is_refused():
    """n=20 calibration units cannot express alpha=0.01; returning "everything"
    would be true and useless, so it raises instead."""
    L = peaked_log_pmf(60, 10, seed=13)
    S = hpd_scores(L)
    s_cal = S[np.arange(20), sample_from(L[:20], seed=14)]
    assert alpha_floor(20) == pytest.approx(1.0 / 21.0)
    with pytest.raises(ValueError, match="resolution floor"):
        e_prediction_set(s_cal, S[20:], 0.01)
    e_prediction_set(s_cal, S[20:], 0.20)          # above the floor: fine


def test_icad_alarm_refuses_an_unsupported_alpha():
    icad = EvalueICAD().calibrate(np.random.default_rng(15).normal(size=30))
    with pytest.raises(ValueError, match="resolution floor"):
        icad.alarm(np.zeros(5), alpha=0.001)


def test_changing_the_evaluator_between_calibration_and_scoring_is_refused():
    """Rank-based scores are exactly where a 1e-4 evaluator wobble flips a tie."""
    class Model:
        compiled = None
        evaluator = "recursive"

    m = Model()
    icad = EvalueICAD().calibrate(np.random.default_rng(16).normal(size=50), model=m)
    m.compiled = object()                      # silent fallback -> layered
    with pytest.raises(RuntimeError, match="evaluator changed"):
        icad.p_value(np.zeros(5), model=m)


def test_sum_domination_by_one_calibration_unit_is_measured():
    """One confidently-wrong calibration unit setting every threshold is the
    Task A scale trap reappearing in Task B. It must be visible."""
    n_bins = 12
    L = peaked_log_pmf(200, n_bins, seed=17, sharpness=3.0)
    # implant one unit whose TRUE bin is far from its confident mode
    L[0] = np.log(np.full(n_bins, 1e-12))
    L[0, 0] = np.log(1.0 - 1e-11)
    L[0] -= np.log(np.exp(L[0]).sum())
    y = sample_from(L, seed=18)
    y[0] = n_bins - 1                                # confidently wrong

    pc = FakeSurvivalPC(L)
    eps = EvaluePredictionSet(pc, alpha=0.10, log_s_max=60.0)
    eps.calibrate(index_batch(200), y)
    assert eps.diagnostics["sum_concentration"] > 0.9   # one point IS the sum

    tight = EvaluePredictionSet(pc, alpha=0.10, log_s_max=5.0)
    tight.calibrate(index_batch(200), y)
    assert tight.diagnostics["sum_concentration"] < eps.diagnostics["sum_concentration"]


def test_split_units_three_never_shares_a_unit():
    units = np.repeat(np.arange(30), 7)
    fit, cal, test = split_units_three(units, seed=19)
    assert (fit | cal | test).all() and not (fit & cal).any()
    assert not (cal & test).any() and not (fit & test).any()
    for a, b in ((fit, cal), (cal, test), (fit, test)):
        assert not set(units[a]) & set(units[b])


def test_split_units_three_refuses_too_few_units():
    with pytest.raises(ValueError, match="at least 3 units"):
        split_units_three(np.repeat(np.arange(2), 5))


def test_reduce_by_unit_max_is_monotone_in_time():
    """The per-unit max makes the sequential score non-decreasing (Laxhammar
    §4.1 requirement 3) — a mean would let an alarm un-fire."""
    scores = np.array([1.0, 5.0, 2.0, 0.5])
    units = np.array([0, 0, 0, 0])
    s, u = reduce_by_unit(scores, units, "max")
    assert s[0] == 5.0 and u[0] == 0
    running = np.maximum.accumulate(scores)
    assert (np.diff(running) >= 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# C. Coverage policy
# ═══════════════════════════════════════════════════════════════════════════

def test_policy_with_lambda_zero_reproduces_the_fixed_alpha_baseline():
    """The comparison must be nested, or a 'gain' is unattributable."""
    L = peaked_log_pmf(300, 12, seed=20)
    pol = CoveragePolicy(alpha=0.10, lam=0.0)
    pol._fit_scaler(L)
    assert np.allclose(pol.alpha_for(L), 0.10)


def test_policy_keeps_mean_alpha_at_target_and_never_uses_test_neighbours():
    L = peaked_log_pmf(400, 12, seed=21)
    pol = CoveragePolicy(alpha=0.10, lam=1.0)
    z = pol._fit_scaler(L)
    pol.tilt_mean = float(np.exp(1.0 * z).mean())
    a = pol.alpha_for(L)
    assert abs(a.mean() - 0.10) < 0.02
    # frozen normaliser: scoring a SUBSET must not change any alpha, or the
    # level would depend on which other test points happened to be in the batch
    assert np.allclose(pol.alpha_for(L[:50]), a[:50])


def test_policy_training_selects_a_lambda_and_reports_the_nested_baseline():
    n_bins = 15
    L = peaked_log_pmf(600, n_bins, seed=22)
    y = sample_from(L, seed=23)
    S = hpd_scores(L)
    s_cal = S[np.arange(len(y)), y]
    pol = CoveragePolicy(alpha=0.10).fit(L, y, s_cal)
    d = pol.diagnostics
    assert np.isfinite(d["baseline_mean_size"])
    assert d["loo_coverage"] >= 0.85
    # whichever lam wins, it cannot be worse than the nested lam=0 baseline
    assert d["loo_mean_size"] <= d["baseline_mean_size"] + 1e-9


def test_lambda_selection_hits_a_target_set_size():
    n_bins = 20
    L = peaked_log_pmf(500, n_bins, seed=24)
    y = sample_from(L, seed=25)
    s_cal = hpd_scores(L)[np.arange(len(y)), y]
    pol = CoveragePolicy(alpha=0.10)
    pol._fit_scaler(L)
    for target in (3.0, 6.0):
        select_lambda_for_target_size(pol, L, s_cal, target)
        assert abs(pol.diagnostics["achieved_size"] - target) < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# C2. Conformal test martingale — anytime validity
# ═══════════════════════════════════════════════════════════════════════════

def test_online_conformal_pvalues_are_iid_uniform_under_exchangeability():
    """Exact uniformity (not just super-uniformity) is what makes the product a
    martingale. Checked on the marginal AND on the lag-1 dependence."""
    rng = np.random.default_rng(50)
    allp = np.concatenate([
        online_conformal_pvalues(rng.normal(size=300), n_warm=50, seed=s)
        for s in range(40)])
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(np.mean(allp <= q) - q) < 0.02
    assert abs(np.corrcoef(allp[:-1], allp[1:])[0, 1]) < 0.05


def test_warm_start_does_not_break_uniformity():
    rng = np.random.default_rng(51)
    p = np.concatenate([
        online_conformal_pvalues(rng.normal(size=200), n_warm=n, seed=s)
        for s in range(30) for n in (0, 100)])
    assert abs(p.mean() - 0.5) < 0.02


def test_martingale_never_alarms_faster_than_ville_allows():
    """THE gate. P(sup_T M_T >= 1/alpha) <= alpha on an exchangeable sequence,
    for both betting rules. If this fails the monitor is worthless — an
    anytime-valid alarm whose rate is not controlled is just a threshold."""
    rng = np.random.default_rng(52)
    for betting in ("mixture", "power"):
        for alpha in (0.05, 0.20):
            fired = []
            for rep in range(400):
                ctm = ConformalTestMartingale(betting=betting, alpha=alpha,
                                              eps=0.5, seed=rep)
                res = ctm.run(rng.normal(size=200), warm_scores=rng.normal(size=100))
                fired.append(res["alarm_at"] >= 0)
            assert np.mean(fired) <= alpha + 0.03, (betting, alpha, np.mean(fired))


def test_martingale_detects_a_changepoint():
    """Validity without power is useless: a distribution shift must be caught."""
    rng = np.random.default_rng(53)
    fired, leads = [], []
    for rep in range(60):
        healthy = rng.normal(size=60)
        shifted = rng.normal(loc=3.0, size=60)
        ctm = ConformalTestMartingale(alpha=0.05, seed=rep)
        res = ctm.run(np.concatenate([healthy, shifted]),
                      warm_scores=rng.normal(size=100))
        fired.append(res["alarm_at"] >= 0)
        if res["alarm_at"] >= 0:
            leads.append(res["alarm_at"])
    assert np.mean(fired) > 0.9
    assert np.median(leads) >= 55          # fires after the change, not before


def test_naive_products_apparent_validity_is_an_artefact_of_kappa():
    """
    The failure this construction exists to avoid — and it is a TRAP, not a
    blow-up.

    At the default kappa=0.5 the naive product respects the Ville bound by
    accident: E[log f(P)] = log k + 1 - k = -0.19 per step, so it goes bankrupt
    before the shared-calibration-set bias can push it anywhere. Weaken that
    drift (kappa -> 0.8, which is squarely inside the range §6 tells you to
    ablate) and it alarms ~3x more often than alpha permits on data that is
    exchangeable by construction.

    Pinned because the tempting version of this test — run it at the default,
    see 2%, conclude the shortcut is fine — is exactly the mistake.
    """
    rng = np.random.default_rng(54)
    rates = {}
    for kappa in (0.5, 0.8):
        fired = [naive_product_alarm_rate(rng.normal(size=50),
                                          rng.normal(size=1500),
                                          alpha=0.05, kappa=kappa)["ever_alarmed"]
                 for _ in range(300)]
        rates[kappa] = float(np.mean(fired))

    assert rates[0.5] <= 0.08                     # accidentally fine
    assert rates[0.8] > 0.09                      # genuinely invalid
    assert rates[0.8] > 2.5 * rates[0.5]

    # the proper construction holds at BOTH, because it does not depend on the
    # betting function for its validity
    for betting, eps in (("power", 0.5), ("power", 0.8)):
        proper = [ConformalTestMartingale(betting=betting, eps=eps, alpha=0.05,
                                          seed=r).run(rng.normal(size=300),
                                                      warm_scores=rng.normal(size=50)
                                                      )["alarm_at"] >= 0
                  for r in range(300)]
        assert np.mean(proper) <= 0.08, (betting, eps, np.mean(proper))


def test_naive_product_is_not_rescued_by_being_powerful():
    """It detects a strong changepoint just as well, so 'it works in practice'
    is not an argument — the two differ in guarantee, not in power."""
    rng = np.random.default_rng(57)
    seq = lambda: np.concatenate([rng.normal(size=60), rng.normal(loc=3.0, size=60)])
    naive = [naive_product_alarm_rate(rng.normal(size=200), seq(),
                                      alpha=0.05)["ever_alarmed"] for _ in range(100)]
    proper = [ConformalTestMartingale(alpha=0.05, seed=r).run(
        seq(), warm_scores=rng.normal(size=200))["alarm_at"] >= 0 for r in range(100)]
    assert np.mean(naive) > 0.9 and np.mean(proper) > 0.9


def test_mixture_grid_coarseness_cannot_break_validity():
    """A convex combination of martingales is a martingale, so grid size is a
    power knob only. Pinned because 'refine the grid' looks like a fix and
    would mask a real validity bug."""
    rng = np.random.default_rng(55)
    for n_grid in (3, 51):
        fired = [ConformalTestMartingale(n_grid=n_grid, alpha=0.05, seed=r).run(
            rng.normal(size=150), warm_scores=rng.normal(size=100))["alarm_at"] >= 0
            for r in range(300)]
        assert np.mean(fired) <= 0.08


def test_overlapping_windows_are_refused_for_sequential_monitoring():
    """Overlap makes the exchangeability null false before any degradation, and
    the martingale then alarms on the overlap (57% on healthy data at stride 1
    against nominal 5%). No correction repairs it, so it is refused."""
    assert_non_overlapping(window=6, stride=6)
    assert_non_overlapping(window=6, stride=8)
    for stride in (1, 2, 5):
        with pytest.raises(ValueError, match="NON-OVERLAPPING"):
            assert_non_overlapping(window=6, stride=stride)

    rng = np.random.default_rng(58)
    with pytest.raises(ValueError, match="NON-OVERLAPPING"):
        martingale_lead_times(rng.normal(size=60), np.repeat(np.arange(6), 10),
                              np.tile(np.linspace(10, 1, 10), 6),
                              window=6, stride=2)


def test_autocorrelated_scores_break_the_martingale_null():
    """The mechanism behind the guard: an AR(1) sequence with NO changepoint
    still alarms far above alpha, because it is not exchangeable."""
    rng = np.random.default_rng(59)
    def ar1(n, rho=0.9):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + rng.normal() * np.sqrt(1 - rho ** 2)
        return x
    fired_ar = [ConformalTestMartingale(alpha=0.05, seed=r).run(ar1(150))["alarm_at"] >= 0
                for r in range(200)]
    fired_iid = [ConformalTestMartingale(alpha=0.05, seed=r).run(rng.normal(size=150))["alarm_at"] >= 0
                 for r in range(200)]
    assert np.mean(fired_iid) <= 0.08          # exchangeable: Ville holds
    assert np.mean(fired_ar) > 0.15            # autocorrelated: null is false


def test_lead_times_are_reported_not_a_detection_rate():
    rng = np.random.default_rng(56)
    n_units, per = 12, 40
    scores, units, rul = [], [], []
    for v in range(n_units):
        healthy = rng.normal(size=per // 2)
        degrading = rng.normal(loc=0.0, size=per // 2) + np.linspace(0, 4, per // 2)
        scores.append(np.concatenate([healthy, degrading]))
        units.append(np.full(per, v))
        rul.append(np.linspace(per, 1, per))
    out = martingale_lead_times(np.concatenate(scores), np.concatenate(units),
                                np.concatenate(rul), warm_scores=rng.normal(size=100),
                                alpha=0.05)
    assert out["frac_fired"] > 0.8
    assert np.isfinite(out["median_lead_time"])
    assert 0 < out["median_lead_time"] < per      # warning before failure
    assert out["lead_time"][~out["fired"]].size == (~out["fired"]).sum()


# ═══════════════════════════════════════════════════════════════════════════
# D. The censored lower bound — the piece that needs the circuit
# ═══════════════════════════════════════════════════════════════════════════

def _survival_setup(n: int = 800, n_bins: int = 20, cap: float = 130.0,
                    seed: int = 26, censor_frac: float = 0.4):
    L = peaked_log_pmf(n, n_bins, seed=seed)
    bins = sample_from(L, seed=seed + 1)
    edges = np.linspace(0, cap, n_bins + 1)
    rng = np.random.default_rng(seed + 2)
    # true RUL uniform within its bin
    rul = edges[bins] + rng.uniform(0, 1, n) * (edges[1] - edges[0])
    delta = (rng.uniform(0, 1, n) > censor_frac).astype(int)
    observed = np.where(delta == 1, rul, rul * rng.uniform(0.2, 0.9, n))
    return FakeSurvivalPC(L, cap), rul, observed, delta


def test_survival_bound_covers_at_nominal_rate():
    pc, rul, observed, delta = _survival_setup()
    n = len(rul)
    cb = ConformalSurvivalBound(pc, alpha=0.10, use_censored=False)
    cb.calibrate(index_batch(n)[:400], observed[:400], delta[:400])
    rep = cb.report(index_batch(n)[400:], rul[400:])
    assert rep["coverage"] >= 0.90 - 0.04


def test_using_censored_units_is_conservative_never_anti_conservative():
    """The whole claim of the piece: an upper bound on an unobservable score
    can only RAISE q_hat, hence only LOWER the bound, hence only increase
    coverage. If this ever reverses, the bound is invalid."""
    pc, rul, observed, delta = _survival_setup(censor_frac=0.5)
    n = len(rul)
    Xc, Xt = index_batch(n)[:400], index_batch(n)[400:]

    dropped = ConformalSurvivalBound(pc, alpha=0.10, use_censored=False)
    dropped.calibrate(Xc, observed[:400], delta[:400])
    kept = ConformalSurvivalBound(pc, alpha=0.10, use_censored=True)
    kept.calibrate(Xc, observed[:400], delta[:400])

    assert kept.q_hat >= dropped.q_hat - 1e-9
    assert (kept.predict(Xt) <= dropped.predict(Xt) + 1e-9).all()
    assert kept.report(Xt, rul[400:])["coverage"] >= 0.90 - 0.04
    assert kept.n_censored_used > 0


def test_censored_units_are_actually_used_not_silently_dropped():
    pc, rul, observed, delta = _survival_setup(censor_frac=0.6)
    n = len(rul)
    cb = ConformalSurvivalBound(pc, alpha=0.10, use_censored=True)
    cb.calibrate(index_batch(n)[:400], observed[:400], delta[:400])
    assert cb.n_cal == 400
    assert cb.diagnostics["n_censored_used"] == int((delta[:400] == 0).sum())


def test_all_censored_calibration_set_explains_itself():
    pc, rul, observed, delta = _survival_setup(n=200, censor_frac=1.0)
    cb = ConformalSurvivalBound(pc, alpha=0.10, use_censored=False)
    with pytest.raises(ValueError, match="use_censored=True"):
        cb.calibrate(index_batch(200), observed, np.zeros(200, dtype=int))


def test_survival_bound_reads_bin_edges_not_centres():
    """§B.2: a bound quoted at the centre concedes half a bin for a purely
    representational reason. The raw bound must land on an edge."""
    pc, rul, observed, delta = _survival_setup(n=100)
    cb = ConformalSurvivalBound(pc, alpha=0.10)
    raw = cb.raw_bound(index_batch(100))
    edges = np.linspace(0, pc.cap, pc.n_bins + 1)
    assert np.isin(np.round(raw, 6), np.round(edges, 6)).all()


# ═══════════════════════════════════════════════════════════════════════════
# E. End to end, on the fake circuit
# ═══════════════════════════════════════════════════════════════════════════

def test_end_to_end_sets_and_merge():
    n, n_bins = 900, 15
    L = peaked_log_pmf(n, n_bins, seed=27)
    y = sample_from(L, seed=28)
    pc = FakeSurvivalPC(L)
    X = index_batch(n)
    units = np.repeat(np.arange(n // 10), 10)

    eps = EvaluePredictionSet(pc, alpha=0.10).calibrate(X[:400], y[:400])
    rep = eps.report(X[400:], y[400:])
    assert rep["coverage"] >= 0.90 - 0.04
    assert 0 < rep["mean_size_bins"] < n_bins

    # the merge: an anomaly e-value and a RUL e-value on the same units
    rng = np.random.default_rng(29)
    icad = EvalueICAD().calibrate(rng.normal(size=200))
    e_anom = icad.e_value(rng.normal(size=100))
    e_rul = p_to_e(rng.uniform(0, 1, 100), 0.5)
    merged = EvalueMerge()(e_anom, e_rul)
    assert merged.shape == (100,)
    assert merged.mean() <= 1.3
    assert e_to_alarm(merged, 0.05).dtype == bool
