"""
E-value conformal layer on top of the exact PC.

Companion to `poc/EVALUES_Brainstorm.md`.  The design is that document's; what
differs, and why, is recorded here next to the code rather than in a changelog.

WHAT THIS IS.  A distribution-free wrapper that turns the circuit's exact
densities into *calibrated* statements — an anomaly alarm with a controlled
false-alarm rate, and a RUL prediction set with marginal coverage — using
e-values rather than p-values, so the level may be chosen after seeing the data
(post-hoc validity) and so the two statistics can be merged.

WHAT THIS IS NOT.  It does not make the circuit more accurate.  Nothing here
retrains, fine-tunes or even reads the circuit's parameters; the wrapper is
read-only with respect to the model (§7 of the brainstorm).  If AUROC is 0.85
before, it is 0.85 after.  Conformal buys guarantees, not sharpness, and this
module's diagnostics exist to stop the guarantee from being mistaken for one.

THE ONE THING THAT MAKES THIS WORTH DOING HERE.  Two of the pieces below use
something no baseline in this PoC can supply:

  * `EvalueMerge` combines the anomaly statistic and the RUL statistic into a
    single unit-health e-value that stays valid under ARBITRARY dependence
    (Vovk & Wang 2021).  They are obviously dependent — same circuit, same
    density — and no independence assumption is needed.
  * `ConformalSurvivalBound` calibrates a lower bound on remaining life while
    USING the right-censored units rather than dropping them, because the
    circuit computes log P(tau > c | x) exactly via `log_box`.  That is the
    conformal analogue of what `SurvivalPC.fit` already does in training, and
    it is the one place exactness and conformal genuinely need each other.

READ THIS BEFORE TRUSTING A NUMBER OUT OF THIS FILE
---------------------------------------------------
A valid guarantee about a useless prediction is the worst outcome available
here, because it looks correct: coverage comes out at 1-alpha, no code path
complains, and the underlying conditional may be constant.  Two measurements on
record make that a live risk rather than a hypothetical:

  1. `logs/ts/capacity_sweep` — with `leaf_components=1` and the chain
     structure, detection AUROC is flat (0.7502-0.7528) across K=2..16, i.e.
     across a 62x parameter range.  `leaf_components=1` is the DEFAULT of both
     `WindowPC` and `SurvivalPC`.  A conditional from that cell may not depend
     on x in any useful way, and every prediction set built on it would be
     identical and perfectly valid.
  2. `hands_off.md` §B.2 — the "exact != calibrated" finding was largely a
     bin-CENTRE-vs-EDGE unit mismatch, and split conformal was buying "exactly
     the half-bin and nothing else".  So the calibration gap this layer is
     nominally motivated by is mostly already closed; what remains genuinely
     unavailable elsewhere is post-hoc validity, merging, and the censored
     bound.

`assert_conditional_varies` is therefore called by every set-valued routine
here, not offered as an optional check.  It is the cheapest possible guard
against publishing a valid statement about nothing.

EXCHANGEABILITY
---------------
Sliding windows from one engine are near-duplicates, so calibrating on windows
would report a rate that evaporates on a new engine.  Units are the
exchangeable objects.  `unit_ids` is therefore required wherever a guarantee is
claimed, exactly as in `poc/time_series/conformal.py`, and reducing to one
score per unit by `max` has a second benefit: it makes the sequential score
monotone non-decreasing in time, which is requirement 3 of Laxhammar & Falkman
section 4.1 and what makes a sequential alarm well-calibrated in the first
place.

The induced resolution floor is not a tuning knob.  With n calibration UNITS no
level below 1/(n+1) is expressible; FD001's ~100 train units on a 50/50 split
give n ~ 50, hence alpha >= ~0.02.  `alpha_floor()` reports it and the set
routines refuse below it rather than silently returning everything.

THREE SPLITS, NOT TWO
---------------------
    D_fit    the PC is fitted here, and ONLY here
    D_cal    calibration scores; the circuit must never have seen these rows
    D_test   deployment

The coverage policy MAY be trained on D_cal — that is what post-hoc validity
licenses and the whole reason for e-values.  The circuit may not.  Fitting or
fine-tuning the PC on D_cal destroys exchangeability of the calibration scores
and no amount of e-value machinery repairs it.  `split_units_three` builds the
split; nothing in this module can write to the circuit, which is the structural
enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .circuits import DegenerateModelError
from .conformal import _np, conformal_quantile

__all__ = [
    "split_units_three", "reduce_by_unit", "alpha_floor",
    "p_to_e", "e_to_alarm", "EvalueICAD", "EvalueMerge",
    "hpd_scores", "e_prediction_set", "p_prediction_set", "EvaluePredictionSet",
    "CoveragePolicy", "select_lambda_for_target_size",
    "ConformalSurvivalBound", "censored_log_survival",
    "online_conformal_pvalues", "ConformalTestMartingale",
    "naive_product_alarm_rate", "martingale_lead_times",
    "assert_non_overlapping",
    "assert_conditional_varies", "evaluator_fingerprint",
]


# ═══════════════════════════════════════════════════════════════════════════
# 0. Splits, exchangeability units, and the resolution floor
# ═══════════════════════════════════════════════════════════════════════════

def split_units_three(
    unit_ids,
    fit_frac: float = 0.5,
    cal_frac: float = 0.25,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boolean window masks (fit, cal, test) that never share a UNIT.

    The three-way split is the structural decision of the whole design (§1):
    two splits are enough for plain conformal, three are required the moment
    anything is TRAINED on the calibration set — which the coverage policy is.
    Splitting by unit rather than by window is what makes the calibration
    scores exchangeable at all.
    """
    u = _np(unit_ids).astype(int)
    uniq = np.unique(u)
    if len(uniq) < 3:
        raise ValueError(
            f"three-way unit split needs at least 3 units, got {len(uniq)}. "
            "Calibration guarantees here are per-UNIT; a window split would "
            "report coverage that does not hold on a new engine.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_fit = max(1, int(round(len(uniq) * fit_frac)))
    n_cal = max(1, int(round(len(uniq) * cal_frac)))
    n_fit = min(n_fit, len(uniq) - 2)
    n_cal = min(n_cal, len(uniq) - n_fit - 1)
    fit_u = set(uniq[perm[:n_fit]].tolist())
    cal_u = set(uniq[perm[n_fit:n_fit + n_cal]].tolist())
    in_fit = np.array([int(v) in fit_u for v in u])
    in_cal = np.array([int(v) in cal_u for v in u])
    return in_fit, in_cal, ~(in_fit | in_cal)


def reduce_by_unit(scores, unit_ids, how: str = "max"
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    One score per unit, returned with the unit ids in the same order.

    `how="max"` is the default for a reason beyond convenience: taking the
    running maximum over a unit's windows makes the preliminary score monotone
    non-decreasing in time, which is requirement 3 of Laxhammar section 4.1.
    A mean would violate it (a healthy stretch late in life can pull the score
    back down, so an alarm could un-fire).
    """
    s, u = _np(scores), _np(unit_ids).astype(int)
    uniq = np.unique(u)
    agg = {"max": np.max, "mean": np.mean, "median": np.median}[how]
    return np.array([agg(s[u == v]) for v in uniq]), uniq


def alpha_floor(n_cal: int) -> float:
    """
    Smallest expressible miscoverage level, 1/(n+1).

    A data-collection constraint, not a tuning knob (§4.1 / Laxhammar §3.6).
    Report it in any table rather than quoting an alpha the calibration set
    cannot support.
    """
    return 1.0 / (int(n_cal) + 1.0)


def evaluator_fingerprint(model) -> str:
    """
    Which evaluator produced a score: "layered" (CompiledCircuit) or
    "recursive" (the per-node reference).

    Calibration and test scores MUST come from the same one.  They agree to
    ~1e-4 relative by `assert_matches_reference`, which is irrelevant for the
    density ratios in Task B and decidedly not irrelevant for Task A, which is
    RANK-based: a 1e-4 wobble is exactly what flips a tie.  `_compile_or_fallback`
    can silently drop to the recursion (unsupported leaf type, failed gate), so
    the evaluator is recorded at calibration time and checked at scoring time
    rather than assumed.
    """
    compiled = getattr(model, "compiled", None)
    if compiled is not None:
        return "layered"
    return str(getattr(model, "evaluator", "recursive"))


def _check_fingerprint(recorded: Optional[str], model) -> None:
    if recorded is None or model is None:
        return
    now = evaluator_fingerprint(model)
    if now != recorded:
        raise RuntimeError(
            f"evaluator changed between calibration ({recorded}) and scoring "
            f"({now}). Calibration and test scores must come from the same "
            "function or exchangeability of the scores is broken — the "
            "rank-based alarm is the sensitive one. Pin it (pc.use_recursive() "
            "for both, or neither) and re-calibrate.")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Gates
# ═══════════════════════════════════════════════════════════════════════════

def assert_conditional_varies(log_pmf, min_tv: float = 1e-3) -> float:
    """
    Refuse a conditional p(tau | x) that is the same for every input.

    Returns the mean total-variation distance from the batch-average pmf, and
    raises `DegenerateModelError` below `min_tv`.  TV is in [0, 1], so the
    threshold is scale-free — unlike a nat threshold, which moves with window
    length and bin count and would be the same literal-instead-of-state mistake
    that made the `@floor` diagnostic unable to fire.

    This is gate 2 of §5, and it is called by every set-valued routine here
    rather than being optional.  Its failure mode is the expensive one: a
    degenerate conditional produces IDENTICAL prediction sets for every unit,
    with textbook-correct marginal coverage, and every adaptivity claim built
    on it is vacuous while every number looks fine.  On the default
    `leaf_components=1` config this check is expected to be tight.
    """
    P = np.exp(_np(log_pmf))
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
    if len(P) < 2:
        return float("inf")
    tv = 0.5 * np.abs(P - P.mean(axis=0, keepdims=True)).sum(axis=1)
    mean_tv = float(tv.mean())
    if not np.isfinite(mean_tv) or mean_tv < min_tv:
        raise DegenerateModelError(
            f"degenerate conditional: p(tau|x) sits {mean_tv:.3e} in mean total "
            f"variation from the batch average over {len(P)} inputs, i.e. it "
            "barely depends on x. Prediction sets built on this would be "
            "identical for every unit AND correctly covered — valid and "
            "worthless. Check leaf_components (1 is a known-degenerate cell "
            "with the chain structure), tau_where ('deep', not 'root') and "
            "weight_jitter (> 0) before calibrating.")
    return mean_tv


# ═══════════════════════════════════════════════════════════════════════════
# 2. p-values, e-values, merging
# ═══════════════════════════════════════════════════════════════════════════

def p_to_e(p, kappa: float = 0.5) -> np.ndarray:
    """
    Calibrate a p-value into an e-value: f_kappa(p) = kappa * p^(kappa - 1),
    kappa in (0, 1).

    Valid because E[f(P)] = 1 exactly for uniform P and <= 1 for the
    conservative (super-uniform) conformal p-value, since f is decreasing.

    kappa trades sensitivity for flatness: the largest attainable e-value is
    kappa * (n+1)^(1-kappa), so at kappa=0.5 it is 0.5*sqrt(n+1).  Smaller
    kappa reacts harder to very small p (rare, severe anomalies); larger kappa
    is flatter.  Ablate it; the default is not a measured choice.

    A CONSTRAINT ON THAT ABLATION, found while testing this function.  The mean
    is exactly 1 for every kappa in (0,1), but the VARIANCE is finite only for
    kappa > 1/2:

        E[f(P)^2] = kappa^2 * int_0^1 p^(2*kappa - 2) dp  <  inf   iff  kappa > 1/2

    Below 1/2 the e-value is heavy-tailed enough that its sample mean does not
    concentrate — an empirical average over 200k draws still misses 1 by ~5% at
    kappa=0.2.  Validity is unaffected (it is a statement about the expectation,
    which is exact), but three things that look like bugs are not: the
    `diagnose` gate `mean_e <= 1` gets noisy, merged e-values inherit the heavy
    tail, and any measured "power" at small kappa is dominated by rare huge
    values.  So kappa < 0.5 is a deliberate choice to be defended, not a free
    sensitivity knob, and `mean_e` should be read with more slack there.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must lie in (0, 1), got {kappa}")
    pv = np.clip(_np(p), 1e-300, 1.0)
    return kappa * pv ** (kappa - 1.0)


def e_to_alarm(e, alpha: float = 0.05) -> np.ndarray:
    """
    Alarm where E >= 1/alpha.  Markov's inequality gives P(E >= 1/alpha) <=
    alpha for any valid e-value, so the false-alarm rate is controlled with no
    distributional assumption and — this is the point of e-values — alpha may
    be chosen after seeing the data.
    """
    return _np(e) >= 1.0 / float(alpha)


@dataclass
class EvalueICAD:
    """
    Task A — the anomaly alarm.  Inductive conformal anomaly detection with an
    e-value cap on top (Laxhammar & Falkman 2015 + post-hoc validity).

    WHY RANKS AND NOT THE SOFT-RANK E-VARIABLE.  The soft-rank statistic
    E = S_test / [(1/(n+1))(sum_i S_i + S_test)] needs S >= 0 and is invariant
    to S -> cS but NOT to S -> S + b, so an NLL cannot simply be shifted
    positive.  The shift-free choice S = exp(-log p) then puts the whole
    dynamic range of the density into a sum of exponentials, and this circuit's
    log p spans hundreds of nats — so sum_i S_i ~ max_i S_i and the detector
    degenerates into "compare against the single worst calibration window".
    Ranks are invariant to ANY monotone transform of the score, so the scale
    problem vanishes entirely, and p -> e keeps post-hoc validity.

    Scores are POSITIVELY oriented: larger = more anomalous (i.e. NLL, which is
    what `WindowPC.score` and `SurvivalPC.anomaly_score` already return).
    """

    kappa: float = 0.5
    how: str = "max"                       # per-unit reduction
    cal_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_cal: int = 0
    per_unit: bool = True
    evaluator: Optional[str] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # ── calibration ──────────────────────────────────────────────────────

    def calibrate(self, scores, unit_ids=None, model=None) -> "EvalueICAD":
        """
        Record the calibration scores.  `unit_ids` is required for a per-unit
        guarantee; passing None is allowed but records `per_unit=False` so the
        weaker claim is visible in the diagnostics instead of implied.
        """
        if unit_ids is None:
            s = _np(scores)
            self.per_unit = False
        else:
            s, _ = reduce_by_unit(scores, unit_ids, self.how)
            self.per_unit = True
        if len(s) == 0:
            raise ValueError("empty calibration set")
        self.cal_scores = np.sort(s)
        self.n_cal = len(s)
        self.evaluator = evaluator_fingerprint(model) if model is not None else None
        self.diagnostics = {
            "n_cal": float(self.n_cal),
            "alpha_floor": alpha_floor(self.n_cal),
            "per_unit": float(self.per_unit),
            "max_e": self.kappa * (self.n_cal + 1.0) ** (1.0 - self.kappa),
        }
        return self

    # ── scoring ──────────────────────────────────────────────────────────

    def p_value(self, scores, unit_ids=None, model=None) -> np.ndarray:
        """
        Conformal p-value p = (1 + #{i : S_i >= S_test}) / (n + 1).

        The `1 +` is not a smoothing fudge: it is what makes the p-value valid
        (super-uniform) at finite n rather than asymptotically.
        """
        _check_fingerprint(self.evaluator, model)
        if self.n_cal == 0:
            raise RuntimeError("calibrate() first")
        s = (reduce_by_unit(scores, unit_ids, self.how)[0]
             if unit_ids is not None else _np(scores))
        # cal_scores is sorted ascending; #{S_i >= t} = n - searchsorted(t, left)
        ge = self.n_cal - np.searchsorted(self.cal_scores, s, side="left")
        return (1.0 + ge) / (self.n_cal + 1.0)

    def e_value(self, scores, unit_ids=None, model=None) -> np.ndarray:
        return p_to_e(self.p_value(scores, unit_ids, model), self.kappa)

    def alarm(self, scores, alpha: float = 0.05, unit_ids=None,
              model=None) -> np.ndarray:
        if alpha < alpha_floor(self.n_cal):
            raise ValueError(
                f"alpha={alpha} is below the resolution floor "
                f"{alpha_floor(self.n_cal):.4f} set by n_cal={self.n_cal} "
                "calibration units. The data cannot express this level; "
                "report the floor rather than a level it does not support.")
        return e_to_alarm(self.e_value(scores, unit_ids, model), alpha)

    # ── the gate ─────────────────────────────────────────────────────────

    def diagnose(self, healthy_scores, unit_ids=None, alpha: float = 0.05,
                 model=None) -> Dict[str, float]:
        """
        Gate 1 of §5, on HELD-OUT HEALTHY data: a valid e-value has mean <= 1.

        If `mean_e` comes out materially above 1 on data that should be
        exchangeable with the calibration set, then either exchangeability is
        broken (windows treated as units, a leaked unit across the split) or
        the score is oriented the wrong way.  Everything downstream is void
        until it passes, so this runs before anything is reported.

        `saturation` is the fraction of test points achieving the maximum
        attainable e-value.  Near 1 means the statistic has hit its ceiling and
        the ranking carries no further information — the failure mode the
        soft-rank variant would have had, kept as a tripwire in case anyone
        switches this back.
        """
        e = self.e_value(healthy_scores, unit_ids, model)
        max_e = self.kappa * (self.n_cal + 1.0) ** (1.0 - self.kappa)
        out = {
            "mean_e": float(e.mean()),
            "max_e": float(max_e),
            "saturation": float(np.mean(e >= max_e - 1e-9)),
            "empirical_alarm_rate": float(np.mean(e_to_alarm(e, alpha))),
            "nominal_alpha": float(alpha),
            "n_cal": float(self.n_cal),
            "alpha_floor": alpha_floor(self.n_cal),
            "per_unit": float(self.per_unit),
        }
        out["exchangeability_ok"] = float(out["mean_e"] <= 1.0 + 3.0 / np.sqrt(max(len(e), 1)))
        return out


@dataclass
class EvalueMerge:
    """
    Merge several e-values into one, valid under ARBITRARY dependence.

    The weighted arithmetic mean of e-values is an e-value with no independence
    assumption whatsoever (Vovk & Wang 2021).  That is what makes this usable
    here: the anomaly statistic and the RUL statistic come from the SAME
    circuit and the same density, so they are obviously dependent, and any
    p-value combination rule (Fisher, Stouffer) would be invalid.

    This is the piece of the design that actually exploits the model's
    structure — one density answering two questions, merged into a single unit
    health number without an independence lie.

    Weights are a POWER choice, not a validity choice: any fixed non-negative
    weights summing to 1 are valid.  Equal weights are a placeholder.
    """

    weights: Optional[Sequence[float]] = None

    def __call__(self, *e_values) -> np.ndarray:
        arrs = [_np(e).reshape(-1) for e in e_values]
        if not arrs:
            raise ValueError("nothing to merge")
        n = len(arrs[0])
        if any(len(a) != n for a in arrs):
            raise ValueError(
                f"e-value arrays must align sample-for-sample, got lengths "
                f"{[len(a) for a in arrs]}. Merge at the UNIT level if the two "
                "statistics were reduced differently.")
        w = (np.full(len(arrs), 1.0 / len(arrs)) if self.weights is None
             else np.asarray(self.weights, dtype=float))
        if w.min() < 0 or not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must be non-negative and sum to 1")
        return np.tensordot(w, np.stack(arrs, axis=0), axes=(0, 0))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Task B — RUL prediction sets
# ═══════════════════════════════════════════════════════════════════════════

def hpd_scores(log_pmf, log_s_max: float = 30.0) -> np.ndarray:
    """
    Highest-predictive-density nonconformity scores over the RUL bins:

        S(x, k) = exp( log p(tau=k*|x) - log p(tau=k|x) ),   k* = argmax_k

    shape (N, n_bins), with S >= 1 and S(x, k*) = 1 by construction.

    WHY THIS SCORE.  S_min = 1 > 0 is assumption (i) of the paper's Theorem
    2.6, satisfied here for free — most scores do not satisfy it.  And
    thresholding S is thresholding the density, so the resulting set is a
    genuine superlevel set of p(tau|x): narrow where the circuit is confident,
    wide where it is not.  That local adaptivity is the strongest argument for
    a PC as the base model of this wrapper, and it is precisely what the
    paper's own regression testbed (Appendix C) cannot produce.  Because tau is
    already discretised into bins with a CategoricalLeaf, we are literally in
    their classification case and sidestep that degeneracy.

    ON `log_s_max` — this clip is load-bearing, not a numerical nicety.
    S enters the set threshold through sum_i S_i, so a single CONFIDENTLY WRONG
    calibration unit contributes exp(large) and sets the threshold for every
    prediction set on its own.  That is the same sum-domination failure that
    rules out the soft-rank statistic in Task A, reappearing in a different
    place, and Theorem 2.6's finite S_max assumption is exactly about it.
    `EvaluePredictionSet` reports `sum_concentration = max_i S_i / sum_i S_i`
    so the condition is measured rather than assumed; float32 exp also
    overflows near 88 nats, which is well inside this circuit's range.
    """
    L = _np(log_pmf)
    L = L - L.max(axis=1, keepdims=True)          # log p(k|x) - log p(k*|x) <= 0
    return np.exp(np.clip(-L, 0.0, log_s_max))


def e_prediction_set(
    s_cal: np.ndarray,
    s_test: np.ndarray,
    alpha,
    ensure_nonempty: bool = False,
) -> np.ndarray:
    """
    The e-value prediction set

        C(x) = { k : S(x,k) < sum_i S_i / ((n+1)*alpha - 1) }

    which is exactly "include k unless its e-value E = S / [(sum_i S_i + S)/(n+1)]
    exceeds 1/alpha".  `alpha` may be a scalar or a per-sample array (N,) — the
    latter is what makes an adaptive coverage policy legal at all, and it is
    legal only because e-values are post-hoc valid.

    Requires (n+1)*alpha > 1, i.e. alpha above the resolution floor; below it
    the threshold is negative or infinite and the set is everything, which is a
    true but useless statement.  Refused rather than returned.

    An EMPTY set is a legitimate output (the model finds every bin implausible)
    and is informative; `ensure_nonempty` forces the modal bin in for consumers
    that need a non-degenerate interval.
    """
    n = len(s_cal)
    a = np.asarray(alpha, dtype=float).reshape(-1)
    if np.any(a <= alpha_floor(n)):
        raise ValueError(
            f"alpha must exceed the resolution floor {alpha_floor(n):.4f} for "
            f"n_cal={n}; got min alpha {a.min():.4f}. Below the floor every set "
            "is the whole bin range. Collect more calibration UNITS or report "
            "the floor.")
    denom = (n + 1.0) * a - 1.0
    thresh = s_cal.sum() / denom                      # (N,) or (1,)
    keep = s_test < thresh.reshape(-1, 1)
    if ensure_nonempty:
        keep[np.arange(len(s_test)), s_test.argmin(axis=1)] = True
    return keep


def p_prediction_set(s_cal: np.ndarray, s_test: np.ndarray, alpha: float
                     ) -> np.ndarray:
    """
    The plain split-conformal ("p-fixed") set: keep k iff S(x,k) <= Q_{1-alpha},
    the (1-alpha)(1+1/n) empirical quantile of the calibration scores.

    THIS IS THE BASELINE THAT MUST APPEAR IN EVERY TABLE (§5 step 5).  It has
    the same marginal coverage guarantee, it is uniformly SMALLER than the
    e-value set, and it has no post-hoc validity — the level must be fixed
    before looking at the data, and no coverage policy may be trained.

    That is the entire trade, and it is a trade, not a free lunch: the e-set
    pays width for the right to choose alpha afterwards and to merge
    statistics.  Quoting an e-set's coverage without this column next to it
    makes a conservative bound look like a modelling result.  `EvaluePredictionSet.report`
    therefore computes both by default.
    """
    q = conformal_quantile(np.asarray(s_cal, dtype=float), float(alpha))
    return s_test <= q


@dataclass
class EvaluePredictionSet:
    """
    Task B — calibrated RUL prediction sets over the tau bins, wrapped around a
    fitted `SurvivalPC`.

    The circuit is not modified: `pc.predict` still returns the exact pmf and
    every other query still works.  The set is added alongside.
    """

    pc: object                                   # SurvivalPC (duck-typed)
    alpha: float = 0.10
    log_s_max: float = 30.0
    reduce: str = "random"                       # per-unit reduction; see below
    seed: int = 0
    s_cal: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_cal: int = 0
    evaluator: Optional[str] = None
    policy: Optional["CoveragePolicy"] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # ── calibration ──────────────────────────────────────────────────────

    def calibrate(self, X_cal: torch.Tensor, tau_cal, unit_ids=None,
                  delta_cal=None) -> "EvaluePredictionSet":
        """
        Calibration scores S_i = S(x_i, tau_i) at the TRUE bin.

        `delta_cal` (1 = failure observed, 0 = right-censored) should be passed
        whenever the calibration fleet contains censored units.  For a censored
        window `tau` is the CENSORING BOUND, not the true bin — the unit was
        still alive there — so scoring it as if it were the label puts a wrong
        value into the calibration quantile.

        HOW MUCH IT COSTS, measured rather than asserted (synthetic FD-shaped
        fleet, 19% censored calibration windows, alpha=0.20, reduce="random",
        p-fixed matched coverage, nominal 0.80):

            censored rows mislabelled   0.770 +- 0.058   (7 seeds)
            censored rows dropped       0.777 +- 0.042   (7 seeds)

        So this is a CORRECTNESS fix whose measured effect is small: ~0.007 in
        mean coverage and a ~30% reduction in seed variance.  A single seed
        initially read 0.667 vs 0.738 and looked like a large effect; it was
        noise at n_cal ~ 40 units, where one seed's sd is ~0.06.  Recorded here
        because the tempting version of this docstring quotes the single seed.

        A two-sided SET cannot use censored units at all — there is no observed
        tau to be inside it — so they are dropped, and the resulting bias
        toward early failures is reported in `diagnostics["frac_censored_dropped"]`.
        This is exactly where `ConformalSurvivalBound` differs and why it is the
        more valuable half: a one-sided LOWER bound CAN keep those units,
        because the circuit's exact `log_box` turns "still alive at c" into a
        valid bound on the unobservable score instead of a missing label.

        WHICH PER-UNIT REDUCTION, AND WHAT IT BUYS.  Windows of one engine are
        not exchangeable, so a guarantee needs one score per unit — but the two
        sensible reductions give DIFFERENT guarantees, and the difference is
        large enough that picking silently would misreport the result:

          reduce="max"     the unit's worst window.  Guarantees that EVERY
                           window of a new unit is covered, simultaneously.
                           A strong statement, and correspondingly wide: on
                           synthetic FD-shaped data it returned 11.6 of 12 bins
                           at nominal 0.80, i.e. nearly vacuous.
          reduce="random"  ONE uniformly chosen window per unit (default).
                           Gives ordinary window-level marginal coverage while
                           keeping the UNIT as the exchangeable object — which
                           is the statistically clean way to get the usual
                           guarantee here, rather than pretending windows are
                           independent.

        Neither is wrong; they answer different questions.  The choice is
        recorded in `diagnostics["reduce"]` so a table cannot quietly mix them.
        """
        log_pmf = _np(self.pc.log_pmf(X_cal))
        assert_conditional_varies(log_pmf)
        S = hpd_scores(log_pmf, self.log_s_max)
        y = _np(tau_cal).astype(int)
        s = S[np.arange(len(y)), np.clip(y, 0, S.shape[1] - 1)]

        frac_dropped = 0.0
        if delta_cal is not None:
            obs = _np(delta_cal).astype(int) == 1
            frac_dropped = float((~obs).mean())
            if not obs.any():
                raise ValueError(
                    "every calibration window is right-censored, so no true tau "
                    "is observable and a two-sided prediction set cannot be "
                    "calibrated at all. Use ConformalSurvivalBound, which keeps "
                    "censored units via the exact box query.")
            s = s[obs]
            if unit_ids is not None:
                unit_ids = _np(unit_ids)[obs]

        if unit_ids is not None:
            if self.reduce == "random":
                u = _np(unit_ids).astype(int)
                rng = np.random.default_rng(self.seed)
                s = np.array([rng.choice(s[u == v]) for v in np.unique(u)])
            elif self.reduce == "max":
                s, _ = reduce_by_unit(s, unit_ids, "max")
            else:
                raise KeyError(
                    f"unknown reduce {self.reduce!r} (use 'random' for "
                    "window-level coverage or 'max' for unit-level)")
        self.s_cal = np.asarray(s, dtype=float)
        self.n_cal = len(self.s_cal)
        self.evaluator = evaluator_fingerprint(self.pc)
        total = float(self.s_cal.sum())
        self.diagnostics = {
            "n_cal": float(self.n_cal),
            "reduce": self.reduce if unit_ids is not None else "none(per-window)",
            # censored rows carry a censoring BOUND in place of a label; keeping
            # them silently broke the guarantee (see the calibrate docstring)
            "frac_censored_dropped": frac_dropped,
            "censoring_handled": float(delta_cal is not None),
            "alpha_floor": alpha_floor(self.n_cal),
            # near 1 means ONE calibration unit sets every threshold (see
            # `hpd_scores`); this is the number to look at before believing a
            # set size, and log_s_max is the knob that controls it
            "sum_concentration": float(self.s_cal.max() / max(total, 1e-300)),
            "s_cal_max": float(self.s_cal.max()),
            "s_cal_median": float(np.median(self.s_cal)),
            "frac_at_clip": float(np.mean(self.s_cal >= np.exp(self.log_s_max) - 1e-6)),
        }
        return self

    # ── prediction ───────────────────────────────────────────────────────

    def _alpha_for(self, log_pmf: np.ndarray) -> np.ndarray:
        if self.policy is None:
            return np.full(len(log_pmf), self.alpha)
        return self.policy.alpha_for(log_pmf)

    def predict_set(self, X: torch.Tensor, ensure_nonempty: bool = False
                    ) -> Dict[str, np.ndarray]:
        """
        Returns the boolean bin mask, the induced interval in CYCLES, and the
        per-sample alpha actually used.

        The interval is read off the bin EDGES, not the centres.  That is not a
        cosmetic choice: §B.2 measured PICP 0.616 on centres against 0.929 on
        edges for one extra bin of width, with the PIT variance already at
        1/12 — most of the recorded under-coverage was the endpoint convention
        rather than the density.  A set-valued predictor that reported centres
        would re-introduce exactly that artefact and then "fix" it with
        conformal widening.
        """
        _check_fingerprint(self.evaluator, self.pc)
        if self.n_cal == 0:
            raise RuntimeError("calibrate() first")
        log_pmf = _np(self.pc.log_pmf(X))
        assert_conditional_varies(log_pmf)
        S = hpd_scores(log_pmf, self.log_s_max)
        a = self._alpha_for(log_pmf)
        keep = e_prediction_set(self.s_cal, S, a, ensure_nonempty)

        edges = _np(self.pc.bin_edges())
        idx = np.arange(S.shape[1])
        any_kept = keep.any(axis=1)
        lo_bin = np.where(any_kept, np.argmax(keep, axis=1), 0)
        hi_bin = np.where(any_kept, S.shape[1] - 1 - np.argmax(keep[:, ::-1], axis=1), 0)
        sizes = keep.sum(axis=1)
        self.diagnostics.update({
            "mean_set_size_bins": float(sizes.mean()),
            "frac_empty": float(np.mean(~any_kept)),
            "frac_full": float(np.mean(sizes == S.shape[1])),
            "mean_alpha": float(a.mean()),
        })
        return {
            "mask": keep,
            "size_bins": sizes,
            "lo": np.where(any_kept, edges[lo_bin], np.nan),
            "hi": np.where(any_kept, edges[hi_bin + 1], np.nan),
            "alpha": a,
            # the set need not be contiguous — it is a density superlevel set,
            # so a bimodal conditional legitimately produces a gap.  The
            # interval above is the CONVEX HULL; report this when quoting it.
            "contiguous": np.array([
                bool(row.sum() == 0 or row[idx[row].min():idx[row].max() + 1].all())
                for row in keep]),
        }

    def report(self, X: torch.Tensor, tau_true, with_baseline: bool = True,
               unit_ids=None) -> Dict[str, float]:
        """
        Coverage and mean size on a labelled test split, with the p-fixed
        baseline alongside by default.

        PASS `unit_ids` OR MISREPORT THE COVERAGE.  The guarantee is about a
        draw exchangeable with the CALIBRATION scores, and with
        `reduce="random"` that draw is one random window of a new unit — not
        "every window of every test unit", which is what a plain per-window
        average measures.  The difference is not academic: on synthetic
        FD-shaped data the per-window average of the p-fixed baseline read
        0.733 against a nominal 0.80, purely from the protocol mismatch, while
        the matched estimate was on target.  The e-set's conservatism HIDES
        this (it read 0.980 either way), which is exactly why the p-fixed
        column has to be there.  When `unit_ids` is given, `coverage_matched`
        is the number to quote.

        READ THE TWO TOGETHER.  An e-set is a Markov-type bound, so it is
        CONSERVATIVE by construction: over-coverage (say 0.98 against a nominal
        0.80) is the expected behaviour, not evidence that the density is good.
        The p-fixed columns show what the same scores buy without post-hoc
        validity; the width difference between them IS the price of being
        allowed to choose alpha afterwards and to merge the two statistics.
        Reporting `coverage` alone would credit the model for the bound's
        conservatism.
        """
        out = self.predict_set(X)
        y = _np(tau_true).astype(int)
        idx = np.arange(len(y))
        yc = np.clip(y, 0, out["mask"].shape[1] - 1)
        rep = {
            "coverage": float(out["mask"][idx, yc].mean()),
            "nominal": float(1.0 - out["alpha"].mean()),
            "mean_size_bins": float(out["size_bins"].mean()),
            "frac_empty": float(np.mean(out["size_bins"] == 0)),
            "frac_contiguous": float(out["contiguous"].mean()),
            "sum_concentration": self.diagnostics.get("sum_concentration", float("nan")),
            "n_cal": float(self.n_cal),
        }
        keep_p = None
        if with_baseline:
            log_pmf = _np(self.pc.log_pmf(X))
            S = hpd_scores(log_pmf, self.log_s_max)
            keep_p = p_prediction_set(self.s_cal, S, self.alpha)
            rep["p_fixed_coverage"] = float(keep_p[idx, yc].mean())
            rep["p_fixed_mean_size_bins"] = float(keep_p.sum(axis=1).mean())
            rep["e_width_premium"] = rep["mean_size_bins"] - rep["p_fixed_mean_size_bins"]

        if unit_ids is not None:
            # one draw per test unit, matching how the calibration scores were
            # reduced — this is the estimate the guarantee is about
            uu = _np(unit_ids).astype(int)
            rng = np.random.default_rng(self.seed + 1)
            pick = np.array([rng.choice(np.flatnonzero(uu == v)) for v in np.unique(uu)])
            rep["coverage_matched"] = float(out["mask"][pick, yc[pick]].mean())
            rep["mean_size_matched"] = float(out["size_bins"][pick].mean())
            rep["n_test_units"] = float(len(pick))
            if keep_p is not None:
                rep["p_fixed_coverage_matched"] = float(keep_p[pick, yc[pick]].mean())
        return rep


# ═══════════════════════════════════════════════════════════════════════════
# 4. Adaptive coverage policy (Alg. 1 / Alg. 2)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoveragePolicy:
    """
    A per-sample miscoverage level alpha~(x) with mean alpha, spent where it
    buys the most.

    Legal ONLY because e-values are post-hoc valid (Prop. 2.2): with p-values,
    choosing the level as a function of the data invalidates the guarantee.
    The policy is trained on D_cal — which the circuit must never have seen.

    RESTRICTED POLICY CLASS, stated plainly.  The paper trains a general
    (neural) policy; this is a one-parameter exponential tilt in a scalar
    summary statistic:

        z(x) = standardised negative entropy of p(tau|x)   (high = confident)
        alpha~(x) = alpha * exp(lam*z(x)) / E_cal[exp(lam*z)]

    lam = 0 recovers the fixed-alpha ("e-fixed") baseline exactly, so the
    comparison is nested and any gain is attributable to the tilt.  lam > 0
    spends more alpha on confident points (smaller sets there), less on
    uncertain ones.  The normaliser is computed ONCE on the calibration set and
    frozen, so alpha~(x) depends on x and D_cal only — never on the other test
    points, which would break the guarantee in a way that is invisible.

    If a trained policy shows no gain over lam=0, the likely cause is a flat
    conditional (§4.5: on real FD001 the chain's temporal advantage vanished —
    `chain` 0.7528 vs `chain_perm_blocks` 0.7512 at identical parameter count),
    not a defect in the policy.  Check `assert_conditional_varies` first.
    """

    alpha: float = 0.10
    lam: float = 0.0
    z_mean: float = 0.0
    z_sd: float = 1.0
    tilt_mean: float = 1.0
    alpha_min: float = 1e-3
    alpha_max: float = 0.999
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # ── summary statistic ────────────────────────────────────────────────

    @staticmethod
    def summary(log_pmf: np.ndarray) -> np.ndarray:
        """
        Negative Shannon entropy of the exact conditional, in nats.

        This is `t(X_test)` — and it comes free: the score vector over bins
        that the set already needs IS the statistic the policy consumes, so
        one conditional pass feeds both.
        """
        L = _np(log_pmf)
        P = np.exp(L - L.max(axis=1, keepdims=True))
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
        return np.sum(P * np.log(np.maximum(P, 1e-300)), axis=1)   # = -H

    def _fit_scaler(self, log_pmf_cal: np.ndarray) -> np.ndarray:
        z = self.summary(log_pmf_cal)
        self.z_mean = float(z.mean())
        self.z_sd = float(z.std()) or 1.0
        return (z - self.z_mean) / self.z_sd

    def alpha_for(self, log_pmf: np.ndarray) -> np.ndarray:
        z = (self.summary(log_pmf) - self.z_mean) / self.z_sd
        a = self.alpha * np.exp(self.lam * z) / max(self.tilt_mean, 1e-12)
        return np.clip(a, self.alpha_min, self.alpha_max)

    # ── training (Alg. 1, leave-one-out on D_cal) ────────────────────────

    def fit(
        self,
        log_pmf_cal: np.ndarray,
        tau_cal,
        s_cal: np.ndarray,
        lambdas: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
        log_s_max: float = 30.0,
    ) -> "CoveragePolicy":
        """
        Choose lam by leave-one-out on the calibration set: for each candidate,
        every calibration point is scored against the OTHER n-1, and the lam
        with the smallest mean set size that still meets nominal coverage wins.

        LOO rather than a plain re-fit because the point being scored must not
        contribute to its own threshold — the same reason the calibration set
        is held out from the circuit in the first place.
        """
        L = _np(log_pmf_cal)
        y = _np(tau_cal).astype(int)
        S = hpd_scores(L, log_s_max)
        n = len(s_cal)
        floor = alpha_floor(n - 1)
        z_all = self._fit_scaler(L)
        total = float(s_cal.sum())
        rows = []
        for lam in lambdas:
            tilt = np.exp(lam * z_all)
            tilt_mean = float(tilt.mean()) or 1.0
            a = np.clip(self.alpha * tilt / tilt_mean, self.alpha_min, self.alpha_max)
            a = np.maximum(a, floor + 1e-9)
            # LOO: drop point i from the sum, and use n-1 calibration points
            denom = (n - 1 + 1.0) * a - 1.0
            thresh = (total - s_cal) / denom
            keep = S < thresh.reshape(-1, 1)
            covered = keep[np.arange(n), np.clip(y, 0, S.shape[1] - 1)]
            rows.append({
                "lam": float(lam),
                "mean_size": float(keep.sum(axis=1).mean()),
                "coverage": float(covered.mean()),
                "tilt_mean": tilt_mean,
                "mean_alpha": float(a.mean()),
            })
        target = 1.0 - self.alpha
        ok = [r for r in rows if r["coverage"] >= target]
        best = min(ok or rows, key=lambda r: r["mean_size"])
        self.lam = best["lam"]
        self.tilt_mean = best["tilt_mean"]
        # lam = 0 IS the e-fixed baseline, so when it was among the candidates
        # the comparison is nested and the gain is attributable to the tilt
        # alone.  Reported rather than assumed: the paper's own gain was ~8%,
        # and on a flat conditional the honest outcome here is 0%.
        baseline = next((r["mean_size"] for r in rows if r["lam"] == 0.0), float("nan"))
        self.diagnostics = {
            "selected_lam": self.lam,
            "loo_mean_size": best["mean_size"],
            "loo_coverage": best["coverage"],
            "loo_mean_alpha": best["mean_alpha"],
            "baseline_mean_size": baseline,
            "size_gain_vs_fixed": (float(baseline - best["mean_size"])
                                   if np.isfinite(baseline) else float("nan")),
            "met_target": float(bool(ok)),
        }
        return self


def select_lambda_for_target_size(
    policy: CoveragePolicy,
    log_pmf_cal: np.ndarray,
    s_cal: np.ndarray,
    target_size_bins: float,
    bracket: Tuple[float, float] = (1e-4, 0.5),
    iters: int = 40,
    log_s_max: float = 30.0,
) -> float:
    """
    Alg. 2 — bracket-and-bisect the BASE alpha to hit a target mean set size.

    Set size in bins translates directly into an RUL interval width in cycles
    (bin width = cap / n_bins), which is the number a maintenance planner
    actually asks for.  Mean set size is monotone decreasing in alpha, so
    bisection is well posed.

    Returns the selected base alpha and leaves it set on `policy`.
    """
    L = _np(log_pmf_cal)
    S = hpd_scores(L, log_s_max)
    n = len(s_cal)
    total = float(s_cal.sum())
    floor = alpha_floor(n)

    def mean_size(base_alpha: float) -> float:
        saved = policy.alpha
        policy.alpha = base_alpha
        a = np.maximum(policy.alpha_for(L), floor + 1e-9)
        policy.alpha = saved
        thresh = total / ((n + 1.0) * a - 1.0)
        return float((S < thresh.reshape(-1, 1)).sum(axis=1).mean())

    lo, hi = max(bracket[0], floor + 1e-6), bracket[1]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_size(mid) > target_size_bins:
            lo = mid                      # sets too wide -> raise alpha
        else:
            hi = mid
    policy.alpha = 0.5 * (lo + hi)
    policy.diagnostics["alpha_for_target_size"] = policy.alpha
    policy.diagnostics["achieved_size"] = mean_size(policy.alpha)
    policy.diagnostics["target_size"] = float(target_size_bins)
    return policy.alpha


# ═══════════════════════════════════════════════════════════════════════════
# 5. Censoring — the piece that needs the circuit
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 4b. Conformal test martingales — anytime-valid sequential monitoring
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT A MARTINGALE BUYS.  Ville's inequality: for a nonnegative martingale
# starting at 1, P(sup_T M_T >= 1/alpha) <= alpha.  "sup over T" is the point —
# you may watch a unit forever, look continuously, stop whenever you like, and
# the guarantee is unchanged.  Testing each window at level alpha instead would
# need a multiple-testing correction over hundreds of windows AND a horizon
# fixed in advance.
#
# THE BETTING PICTURE.  Start with wealth 1; at each step bet against "this
# unit is still behaving normally" at odds E_t.  Wealth is the product
# M_T = prod_t E_t.  Under the null each bet is fair (E[E_t | past] <= 1), so
# you cannot systematically get rich — which is exactly the martingale
# property, and therefore GETTING RICH IS THE EVIDENCE.  Wealth 20 means you
# multiplied your stake twentyfold betting against normality, which the null
# allowed with probability <= 1/20.  This is also the reason the whole design
# uses e-values: p-values do not multiply into anything meaningful.
#
# WHY THE NAIVE VERSION IS WRONG, AND WHAT REPLACES IT.  Multiplying the
# e-values from `EvalueICAD` is NOT a martingale: they are all computed against
# one frozen calibration set, so they are mutually dependent and each is valid
# only marginally, not CONDITIONALLY on the past.  Measured, the failure is
# subtler than "it blows up": at the default kappa=0.5 the naive product
# respects the bound by ACCIDENT (the bet's negative drift bankrupts it before
# the bias matters), and at kappa=0.8-0.95 it alarms 3x more often than alpha
# allows on exchangeable data.  Since kappa is exactly what §6 tells you to
# ablate, that is a trap.  `naive_product_alarm_rate` carries the numbers.
#
# The rigorous construction is Vovk's conformal test martingale, which bets on
# the P-VALUE SEQUENCE instead.  Under exchangeability the SMOOTHED ONLINE
# conformal p-values are i.i.d. uniform, so for any betting function
# f: [0,1] -> [0, inf) with integral 1, prod_t f(p_t) is a genuine nonnegative
# martingale.  That independence is what the fixed calibration set destroys and
# the online construction restores.
#
# WHAT IT ACTUALLY DETECTS, AND WHY THE METRIC CHANGES.  A conformal test
# martingale detects DEPARTURE FROM EXCHANGEABILITY.  For a degrading turbofan
# that is not an anomaly, it is the physics: degradation IS non-exchangeability
# (§4.2).  So the martingale will eventually fire on every healthy unit,
# correctly.  The useful question is therefore not WHETHER it fires but WHEN —
# how much warning it gives before failure.  That makes it a changepoint /
# early-warning statistic to be scored by LEAD TIME, not an alarm to be scored
# by false-alarm rate.  `martingale_lead_times` reports it that way.  The
# false-alarm guarantee is still real and still testable, but only on a
# genuinely exchangeable (non-degrading) sequence, which is what
# `test_martingale_never_alarms_faster_than_ville_allows` checks.


def online_conformal_pvalues(
    scores,
    n_warm: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Smoothed online conformal p-values (Vovk):

        p_t = ( #{i <= t : s_i > s_t} + tau_t * #{i <= t : s_i = s_t} ) / t,
        tau_t ~ U[0,1] independent

    Under exchangeability of the score sequence these are i.i.d. UNIFORM — not
    merely super-uniform.  Exact uniformity is what makes the product of bets a
    martingale, and it is why the smoothing term is not optional here even
    though it is elsewhere: without it the p-values are discrete and dependent
    on the tie structure, and the product picks up a drift.

    `n_warm` treats the first `n_warm` scores as a warm-start pool (healthy
    windows from OTHER units) and returns p-values only for the rest.  This is
    still exactly the online procedure — we simply do not bet on the first
    n_warm steps — so validity is untouched, while the very first windows of a
    monitored unit are compared against a real reference instead of against a
    handful of their own predecessors.  Without it, p_1 = tau_1 is pure noise
    and the martingale wastes its early bets.

    Returns p-values of length len(scores) - n_warm.
    """
    import bisect

    s = _np(scores).reshape(-1)
    rng = np.random.default_rng(seed)
    seen: list = []
    out = []
    for t, v in enumerate(s, start=1):
        # counts include the incoming point itself, as the definition requires
        lo = bisect.bisect_left(seen, v)
        hi = bisect.bisect_right(seen, v)
        n_gt = len(seen) - hi
        n_eq = (hi - lo) + 1
        bisect.insort(seen, v)
        if t > n_warm:
            out.append((n_gt + rng.uniform() * n_eq) / t)
    return np.clip(np.asarray(out, dtype=float), 1e-12, 1.0)


def _log_betting(p: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """log f_eps(p) = log eps + (eps - 1) log p, for a grid of eps. (T, E)."""
    lp = np.log(p).reshape(-1, 1)
    e = np.asarray(eps, dtype=float).reshape(1, -1)
    return np.log(e) + (e - 1.0) * lp


@dataclass
class ConformalTestMartingale:
    """
    Anytime-valid sequential monitor built on the online conformal p-values.

    betting:
      "power"    a single power martingale f_eps(p) = eps * p^(eps-1).
                 int_0^1 f = 1, so it is a valid bet.  Small eps bets hard on
                 tiny p-values (sharp, rare departures), large eps is flatter.
      "mixture"  the SIMPLE MIXTURE MARTINGALE: average the power martingales
                 over a grid of eps, i.e. M_T = mean_eps prod_t f_eps(p_t).
                 A convex combination of martingales is a martingale, so the
                 grid introduces NO validity error however coarse it is — only
                 power is affected.  This is the default because it removes the
                 eps choice, which is otherwise the one free parameter that
                 decides what the monitor is sensitive to.

    Note the same infinite-variance caveat as `p_to_e`: the eps <= 1/2 members
    of the mixture are heavy-tailed.  That costs nothing in validity (Ville is
    a statement about the supremum, not about moments) but it means individual
    wealth trajectories are jumpy, and a single run's peak is not a stable
    quantity to quote.
    """

    betting: str = "mixture"
    eps: float = 0.5
    n_grid: int = 51
    alpha: float = 0.05
    seed: int = 0

    def _grid(self) -> np.ndarray:
        if self.betting == "power":
            return np.array([float(self.eps)])
        if self.betting == "mixture":
            return np.linspace(0.02, 0.98, self.n_grid)
        raise KeyError(f"unknown betting rule {self.betting!r} (power | mixture)")

    def run(self, scores, warm_scores=None) -> Dict[str, np.ndarray]:
        """
        Monitor one sequence, in TIME ORDER.  `warm_scores` is the healthy
        reference pool (see `online_conformal_pvalues`).

        Returns the wealth trajectory in logs (wealth spans many orders of
        magnitude and float64 overflows quickly), the p-values, and the index
        of the first Ville alarm — `alarm_at = -1` if it never fires.
        """
        s = _np(scores).reshape(-1)
        n_warm = 0
        if warm_scores is not None:
            w = _np(warm_scores).reshape(-1)
            s = np.concatenate([w, s])
            n_warm = len(w)
        p = online_conformal_pvalues(s, n_warm=n_warm, seed=self.seed)
        eps = self._grid()
        log_f = _log_betting(p, eps)                      # (T, E)
        log_wealth_e = np.cumsum(log_f, axis=0)           # per-eps martingales
        # uniform mixture over the grid == numerical integration over eps
        m = log_wealth_e.max(axis=1, keepdims=True)
        log_wealth = (m.ravel() + np.log(np.exp(log_wealth_e - m).mean(axis=1)))
        thresh = np.log(1.0 / self.alpha)
        hit = np.flatnonzero(log_wealth >= thresh)
        return {
            "p": p,
            "log_wealth": log_wealth,
            "alarm_at": int(hit[0]) if len(hit) else -1,
            "max_log_wealth": float(log_wealth.max()) if len(log_wealth) else -np.inf,
            "threshold": float(thresh),
        }


def naive_product_alarm_rate(
    cal_scores,
    test_scores,
    alpha: float = 0.05,
    kappa: float = 0.5,
) -> Dict[str, float]:
    """
    THE THING NOT TO DO — and it fails in a more interesting way than expected.

    Multiplies the marginally-valid e-values from a FIXED calibration set as if
    they were conditionally valid.  This is not a martingale, so Ville does not
    apply and there is no guarantee.  What the guarantee's absence actually
    COSTS turns out to depend entirely on the betting function, and the default
    hides it.

    Measured on exchangeable null data, T=2000, nominal alpha=0.05, over 300
    repetitions.  The per-step log-drift of the bet under the null is
    E[log f(P)] = log kappa + 1 - kappa:

        kappa   drift/step   alarm rate (n_cal=20 / 200)
        0.50      -0.193        2.3% / 3.0%      <- looks valid
        0.80      -0.023       13.7% / 11.3%     <- 2-3x over
        0.95      -0.001       14.3% / 17.3%     <- 3x over
        0.99      -0.000        6.3% / 1.7%

    So at the default kappa=0.5 the naive product appears to respect the bound,
    and it does so BY ACCIDENT: the bet's strong negative drift (-0.19 per step)
    bankrupts the product faster than the shared-calibration-set bias can push
    it up.  Weaken the drift and the bias is exposed — the wealth reaches
    exp(153) in the 99th percentile at kappa=0.8.

    That makes this a trap rather than a mere inelegance, because §6 of the
    design document explicitly lists kappa as a parameter to ABLATE.  Anyone
    following that instruction with the naive product moves from an
    accidentally-safe setting into a 3x-invalid one, with nothing in the output
    to indicate it.  Neither does power justify the shortcut: on a strong
    changepoint the naive product and the proper martingale both detect 100%.

    The conclusion to carry into the write-up is therefore NOT "the naive
    product blows up" (at the default it does not) but: its validity is an
    artefact of one parameter setting, it has no theorem behind it, and the
    conformal test martingale costs nothing to use instead.
    """
    icad = EvalueICAD(kappa=kappa).calibrate(cal_scores)
    e = icad.e_value(test_scores)
    log_wealth = np.cumsum(np.log(np.maximum(e, 1e-300)))
    thresh = np.log(1.0 / alpha)
    return {
        "ever_alarmed": float(np.any(log_wealth >= thresh)),
        "max_log_wealth": float(log_wealth.max()),
        "threshold": float(thresh),
        "final_log_wealth": float(log_wealth[-1]),
    }


def assert_non_overlapping(window: int, stride: int) -> None:
    """
    Refuse a sequential monitor built on OVERLAPPING windows.

    This is the single largest trap in the martingale half, and it is invisible
    without a control.  Consecutive windows at stride < window share timesteps,
    so the score sequence is autocorrelated and therefore NOT exchangeable —
    before any degradation has happened.  The martingale then correctly detects
    non-exchangeability of the wrong kind, and reports it as an alarm.

    Measured (synthetic fleet, window=6, alpha=0.05, monitor run on the HEALTHY
    region only, RUL > 100, so no degradation is present to find):

        stride 6 (disjoint)    healthy-only alarm rate   2%   <- valid
        stride 2 (overlap 4)   healthy-only alarm rate  17%
        stride 1 (overlap 5)   healthy-only alarm rate  57%

    The same sweep on FULL lives looks like a spectacular improvement — firing
    rises 33% -> 75% -> 90% and median lead time 13 -> 30 -> 130 cycles — and
    all of it is the artefact.  At stride 1 the "lead time" equals the RUL cap
    because the monitor fires on the first window of every unit.

    So: stride >= window, or do not use a martingale.  There is no correction
    that repairs this; the null itself is false.
    """
    if int(stride) < int(window):
        raise ValueError(
            f"sequential monitoring needs NON-OVERLAPPING windows, got "
            f"window={window} stride={stride} (overlap {window - stride}). "
            "Overlapping windows are autocorrelated, so the exchangeability "
            "null is false before any degradation occurs and the martingale "
            "alarms on the overlap: measured 57% false-alarm rate at stride 1 "
            "against a nominal 5%, on healthy data. Re-window with "
            f"stride >= {window}.")


def martingale_lead_times(
    scores,
    unit_ids,
    rul,
    warm_scores=None,
    alpha: float = 0.05,
    betting: str = "mixture",
    seed: int = 0,
    window: Optional[int] = None,
    stride: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Score the monitor the way it should be scored: by LEAD TIME.

    For each unit the windows are taken in time order, the martingale is run,
    and the RUL AT THE FIRST ALARM is recorded — how many cycles of warning the
    monitor gave.  Units that never alarm get lead time NaN and are counted
    separately.

    Why not a false-alarm rate: degradation is a genuine loss of
    exchangeability, so firing on a degrading unit is CORRECT behaviour and
    "fraction of units that alarm" would approach 1 for a working monitor.  The
    informative quantities are how early it fires and whether it fires earlier
    on units that fail sooner.  Reporting this as a detection rate would be the
    same category error as scoring a changepoint detector with AUROC.

    PASS `window` AND `stride`.  They are optional only so that non-windowed
    callers still work; when the scores come from sliding windows, omitting
    them skips the one check that decides whether any of this means anything
    (see `assert_non_overlapping`).

    AND ALWAYS RUN THE HEALTHY-REGION CONTROL.  Re-run this on the early-life
    windows alone (e.g. RUL > 100), where no degradation exists: the firing
    rate there must sit at or below alpha.  Every apparent improvement in lead
    time observed so far — from stride, and from dropping the warm start — has
    to be checked against that control before it is believed, because both
    knobs raise the firing rate on healthy data as well.
    """
    if window is not None and stride is not None:
        assert_non_overlapping(window, stride)
    s, u, r = _np(scores), _np(unit_ids).astype(int), _np(rul)
    uniq = np.unique(u)
    leads, fired, n_windows, alarm_idx = [], [], [], []
    for k, v in enumerate(uniq):
        m = u == v
        ctm = ConformalTestMartingale(betting=betting, alpha=alpha, seed=seed + k)
        res = ctm.run(s[m], warm_scores=warm_scores)
        i = res["alarm_at"]
        n_windows.append(int(m.sum()))
        alarm_idx.append(i)
        fired.append(i >= 0)
        leads.append(float(r[m][i]) if i >= 0 else np.nan)
    leads = np.asarray(leads, dtype=float)
    fired = np.asarray(fired, dtype=bool)
    return {
        "unit": uniq,
        "lead_time": leads,
        "fired": fired,
        "alarm_index": np.asarray(alarm_idx),
        "n_windows": np.asarray(n_windows),
        "frac_fired": float(fired.mean()),
        "median_lead_time": float(np.nanmedian(leads)) if fired.any() else float("nan"),
        "mean_lead_time": float(np.nanmean(leads)) if fired.any() else float("nan"),
    }


def censored_log_survival(pc, X: torch.Tensor, t_bin) -> np.ndarray:
    """
    Exact log P(tau > t_bin | x) from the circuit's box query.

    `CategoricalLeaf.log_interval` is INCLUSIVE of both endpoints, so the
    suffix is (t+1, inf) — which is what `SurvivalPC.log_survival` already
    does.  Kept as a named entry point because the censored bound below is the
    one consumer that must not get this off by a bin.
    """
    return _np(pc.log_survival(X, t_bin))


@dataclass
class ConformalSurvivalBound:
    """
    A distribution-free LOWER bound on remaining life: L(x) with
    P(tau >= L(x)) >= 1 - alpha.

    WHY ONE-SIDED.  Right-censored units have no observable tau, so they supply
    no two-sided calibration score.  A lower predictive bound is the query that
    survives censoring (conformalized survival analysis; Candes, Lei, Ren,
    Bates), and it is also the query maintenance actually uses: "this unit has
    at least L cycles left" schedules an inspection; a two-sided interval does
    not.

    WHY THIS IS THE PIECE WORTH BUILDING.  Standard practice drops censored
    units from calibration, which biases D_cal toward early failures.  The
    circuit computes log P(tau > c | x) EXACTLY via `log_box`, and that lets a
    censored unit contribute a rigorous bound on its own unobservable score
    instead of being discarded:

        score  s_i = L_raw(x_i) - tau_i            (positive = bound violated)
        censored at bin c_i:  tau_i >= c_i  =>  s_i <= L_raw(x_i) - c_i

    Replacing an unknown score by a valid UPPER bound can only raise the
    empirical quantile, hence only lower L, hence only make coverage more
    conservative — so the guarantee survives intact while the censored units
    stop being thrown away.  This is the conformal analogue of what
    `SurvivalPC.fit` already does with the exact censored likelihood, and
    `use_censored` is the same ablation on the calibration side.

    HONEST CAVEAT.  Validity still requires censoring to be independent of the
    covariates.  For run-to-failure benchmarks that is roughly true; for real
    fleet data it is not (units are pulled from service for reasons correlated
    with their condition).  Say so in any table.  This is the easiest place in
    the whole design to publish something wrong.
    """

    pc: object                                   # SurvivalPC (duck-typed)
    alpha: float = 0.10
    use_censored: bool = True
    q_hat: float = 0.0
    n_cal: int = 0
    n_censored_used: int = 0
    evaluator: Optional[str] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # ── the raw (uncalibrated) bound ─────────────────────────────────────

    def raw_bound(self, X: torch.Tensor) -> np.ndarray:
        """
        The circuit's own alpha-quantile of p(tau|x), in CYCLES, read at the
        LOWER bin edge.

        Edges rather than centres for the §B.2 reason: a bound quoted at the
        bin centre gives away half a bin for a purely representational reason,
        and conformal would then spend real width buying it back.
        """
        log_pmf = _np(self.pc.log_pmf(X))
        assert_conditional_varies(log_pmf)
        cdf = np.exp(log_pmf).cumsum(axis=1)
        n_bins = log_pmf.shape[1]
        idx = np.clip((cdf < self.alpha).sum(axis=1), 0, n_bins - 1)
        return _np(self.pc.bin_edges())[idx]

    # ── calibration ──────────────────────────────────────────────────────

    def calibrate(
        self,
        X_cal: torch.Tensor,
        rul_cal,
        delta_cal,
        unit_ids=None,
    ) -> "ConformalSurvivalBound":
        """
        `rul_cal` is in CYCLES (not bins): for an observed failure it is the
        true remaining life, for a censored unit it is the last observed lower
        bound.  `delta_cal` is 1 for observed, 0 for right-censored.
        """
        y = _np(rul_cal)
        d = _np(delta_cal).astype(int)
        raw = self.raw_bound(X_cal)
        s = raw - y                                   # observed: exact score
        censored = d == 0
        if self.use_censored:
            # tau_i >= y_i, so s_i <= raw_i - y_i: the SAME expression is a
            # valid upper bound for the censored rows, which is why they can be
            # kept at all.  Conservative, never anti-conservative.
            keep = np.ones(len(s), dtype=bool)
            self.n_censored_used = int(censored.sum())
        else:
            keep = ~censored
            self.n_censored_used = 0
        # raw-bound coverage on the OBSERVED rows only, before conformal
        # widening: it says how far off the circuit's own alpha-quantile was,
        # which is the number that tells you whether q_hat is doing a little
        # work or all of it.  Censored rows have no observable tau, so they
        # cannot appear here even when they are used for calibration.
        obs = ~censored
        raw_cov = (float(np.mean(raw[obs] <= y[obs])) if obs.any() else float("nan"))
        s = s[keep]
        if unit_ids is not None:
            # worst (largest) score per unit: the unit is the exchangeable
            # object, and the max is the conservative reduction for a bound
            s, _ = reduce_by_unit(s, _np(unit_ids)[keep], "max")
        if len(s) == 0:
            raise ValueError(
                "empty calibration set for the survival bound. With "
                "use_censored=False every calibration unit was censored; that "
                "is exactly the situation the exact box query exists for — set "
                "use_censored=True.")
        self.n_cal = len(s)
        self.q_hat = conformal_quantile(s, self.alpha)
        self.evaluator = evaluator_fingerprint(self.pc)
        self.diagnostics = {
            "q_hat": float(self.q_hat),
            "n_cal": float(self.n_cal),
            "n_censored_used": float(self.n_censored_used),
            "frac_censored": float(censored.mean()),
            "alpha_floor": alpha_floor(self.n_cal),
            "raw_cal_coverage": raw_cov,
        }
        return self

    # ── prediction ───────────────────────────────────────────────────────

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Calibrated lower bound in cycles, clipped at 0."""
        _check_fingerprint(self.evaluator, self.pc)
        if self.n_cal == 0:
            raise RuntimeError("calibrate() first")
        return np.clip(self.raw_bound(X) - self.q_hat, 0.0, None)

    def report(self, X: torch.Tensor, rul_true) -> Dict[str, float]:
        """Coverage of the lower bound and how much life it concedes."""
        L = self.predict(X)
        y = _np(rul_true)
        return {
            "coverage": float(np.mean(L <= y)),
            "nominal": float(1.0 - self.alpha),
            "mean_bound": float(L.mean()),
            "mean_slack": float(np.mean(y - L)),
            "q_hat": float(self.q_hat),
            "n_cal": float(self.n_cal),
            "n_censored_used": float(self.n_censored_used),
        }
