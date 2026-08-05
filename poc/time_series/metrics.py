"""
Metrics for the time-series PoC.

Detection uses THRESHOLD-FREE measures only (AUROC, average precision) and
never point-adjusted F1 — point adjustment inflates scores so badly that it can
make random noise look state-of-the-art, which is why TSB-AD (NeurIPS 2024)
removed it from its protocol outright.

RUL is scored on the predictive DISTRIBUTION, not only a point estimate: RMSE
and the NASA asymmetric score are reported for comparability, but CRPS,
interval coverage (PICP) and interval width (MPIW) are what the survival claim
actually stands on.  A model can win RMSE while being badly miscalibrated, and
that trade is the whole subject of the experiment.

One caveat on PICP, and it is not a small one (hand-off §B.2): it compares
interval ENDPOINTS to a continuous target, so it inherits whatever convention
produced those endpoints.  `SurvivalPC.predict` emits bin CENTRES, which costs
up to half a bin of coverage at each end for reasons that have nothing to do
with the model.  Report `picp_edge` next to it, and use `pit_report` below to
ask whether the DENSITY is calibrated — that question has no unit mismatch in
it.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).ravel()


# ─── detection ──────────────────────────────────────────────────────────────

def auroc(scores, y) -> float:
    """Rank-based AUROC (Mann–Whitney U); higher score = more anomalous."""
    s, y = _np(scores), _np(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                       # average ranks for ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_p, n_n = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def average_precision(scores, y) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    s, y = _np(scores), _np(y)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return float((precision * y).sum() / n_pos) if n_pos else float("nan")


def detection_report(scores, y, kinds: Sequence[str] = ()) -> Dict[str, float]:
    """Overall AUROC/AP plus per-anomaly-kind AUROC (normals kept as negatives)."""
    out = {"auroc": auroc(scores, y), "ap": average_precision(scores, y)}
    if len(kinds):
        s, y_a, k = _np(scores), _np(y), np.asarray(kinds)
        for kind in sorted(set(k[y_a == 1])):
            sel = (k == kind) | (y_a == 0)
            out[f"auroc[{kind}]"] = auroc(s[sel], y_a[sel])
    return out


# ─── RUL: point estimates ───────────────────────────────────────────────────

def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((_np(pred) - _np(true)) ** 2)))


def mae(pred, true) -> float:
    return float(np.mean(np.abs(_np(pred) - _np(true))))


def nasa_score(pred, true) -> float:
    """
    NASA/PHM08 asymmetric penalty: late predictions (predicting more life than
    remains) cost exponentially more than early ones, because they are the ones
    that let the machine fail in service.  Lower is better.
    """
    d = _np(pred) - _np(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                 np.exp(d / 10.0) - 1.0)))


# ─── RUL: distributional ────────────────────────────────────────────────────

def crps_from_pmf(pmf, y_bin, bin_width: float = 1.0) -> float:
    """
    Continuous ranked probability score for a discrete predictive distribution:
    CRPS = Σ_k (F_k − 1{y ≤ k})² · Δ.  Proper scoring rule — it rewards a
    distribution that is both sharp and calibrated, and it is the metric a
    point regressor cannot compete on at all.
    """
    P = pmf.detach().cpu().numpy() if isinstance(pmf, torch.Tensor) else np.asarray(pmf)
    y = _np(y_bin).astype(int)
    F = P.cumsum(1)
    H = (np.arange(P.shape[1])[None, :] >= y[:, None]).astype(float)
    return float(((F - H) ** 2).sum(1).mean() * bin_width)


def crps_from_interval(lo, hi, true, alpha: float = 0.10) -> float:
    """
    Interval score (Gneiting & Raftery) for methods that emit only a
    prediction interval, e.g. conformal.  Reported alongside CRPS as the
    fair comparison for interval-only predictors: same units, lower better.
    """
    lo, hi, t = _np(lo), _np(hi), _np(true)
    return float(np.mean((hi - lo)
                         + (2 / alpha) * (lo - t) * (t < lo)
                         + (2 / alpha) * (t - hi) * (t > hi)))


def picp(lo, hi, true) -> float:
    """Prediction-interval coverage probability: fraction of truths inside."""
    lo, hi, t = _np(lo), _np(hi), _np(true)
    return float(np.mean((t >= lo) & (t <= hi)))


def mpiw(lo, hi) -> float:
    """Mean prediction-interval width — the sharpness half of calibration."""
    return float(np.mean(_np(hi) - _np(lo)))


def calibration_error(pmf, y_bin, n_levels: int = 10) -> float:
    """
    Mean absolute deviation between nominal and empirical coverage of the
    predictive CDF (a reliability-diagram summary).  0 = perfectly calibrated.
    """
    P = pmf.detach().cpu().numpy() if isinstance(pmf, torch.Tensor) else np.asarray(pmf)
    y = _np(y_bin).astype(int)
    F = P.cumsum(1)
    u = F[np.arange(len(y)), y]              # PIT values
    levels = np.linspace(0.05, 0.95, n_levels)
    return float(np.mean([abs((u <= q).mean() - q) for q in levels]))


def pit_values(pmf, y_bin, seed: int = 0) -> np.ndarray:
    """
    RANDOMISED probability integral transform for a discrete predictive:
    U = F(k-1) + u·p(k) with u ~ Uniform(0,1).  Under a correctly calibrated
    predictive U is exactly Uniform(0, 1).

    Why this and not PICP: it never compares a discretised quantity to a
    continuous one.  PICP takes bin CENTRES as interval endpoints and scores
    them against a target in cycles, so it charges every prediction up to half
    a bin of spurious error — which is most of the recorded under-coverage
    (hand-off §B.2).  The PIT has no such mismatch, so it is the instrument
    that says whether the DENSITY is calibrated, as opposed to whether the
    interval was read off correctly.
    """
    P = pmf.detach().cpu().numpy() if isinstance(pmf, torch.Tensor) else np.asarray(pmf)
    y = _np(y_bin).astype(int)
    rng = np.random.default_rng(seed)
    F = P.cumsum(1)
    rows = np.arange(len(y))
    below = np.where(y > 0, F[rows, np.maximum(y - 1, 0)], 0.0)
    return below + rng.random(len(y)) * P[rows, y]


def pit_report(pmf, y_bin, seed: int = 0) -> Dict[str, float]:
    """
    The PIT split into the two things it can detect, because they have
    different causes and different fixes:

      pit_mean  vs 0.5     LOCATION — the predictive is shifted (in this
                           project: the censoring bias, §B.4)
      pit_var   vs 1/12    SHAPE — the predictive is the wrong WIDTH.
                           var > 1/12 = overconfident (truth in the tails),
                           var < 1/12 = too diffuse.
      pit_ks               overall deviation from uniform, for reference.

    Measured on the recorded configuration: mean 0.408, var 0.0841 against
    1/12 = 0.0833.  A pure location error with the width already right.
    """
    u = pit_values(pmf, y_bin, seed=seed)
    n = len(u)
    ks = float(np.max(np.abs(np.sort(u) - np.arange(1, n + 1) / n))) if n else float("nan")
    return {"pit_mean": float(u.mean()), "pit_var": float(u.var()),
            "pit_ks": ks, "pit_var_target": 1.0 / 12.0}
