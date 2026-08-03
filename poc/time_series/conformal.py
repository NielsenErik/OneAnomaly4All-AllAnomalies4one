"""
Split conformal calibration ON TOP OF the circuit's own predictive.

Why this exists.  The most interesting negative in the PoC is that **exact is
not calibrated**: the joint density over (window, τ) is exactly normalised, and
its nominal 90% intervals cover 38–52% of the truth, while conformalised
quantile regression — which has no exactness property whatsoever — covers
82–91%.  Exactness buys the query set (marginals, survival, censored
likelihood); it does not buy coverage, because a model can be exactly
normalised with respect to a distribution that is simply wrong.

The honest response is not to hide the column but to close it with the tool
designed for the job.  Split conformal turns any predictive into one with a
finite-sample marginal coverage guarantee, and it does so WITHOUT touching the
density: the circuit still answers p(τ|x), S(t|x), marginals under dead
sensors, all exactly.  The combination — exact density, guaranteed coverage —
is a defensible hybrid, and its selling point over plain CQR is that CQR gives
you an interval and nothing else, whereas here the interval is a decoration on
a full joint distribution that can still be queried.

What this is NOT: evidence for T1.  The exact-censored-survival claim died on
its own pre-registered gate and stays dead.  Conformalising the predictive
fixes coverage; it says nothing about whether the censored likelihood term
earns its place.

Two calibrators, both split-conformal, reported side by side:

  cqr   additive:  [q_lo − q̂, q_hi + q̂] with q̂ the (1−α)(1+1/n) empirical
        quantile of the nonconformity scores s = max(q_lo − y, y − q_hi).
        This is Romano et al.'s CQR applied to the circuit's own quantiles, so
        it is exactly the adversary's machinery on our predictive — the
        cleanest possible comparison.  Finite-sample marginal coverage ≥ 1−α.
  pit   quantile recalibration (Kuleshov et al.): learn the map from nominal
        to empirical CDF level on the calibration set and read the recalibrated
        quantiles off the circuit's exact CDF.  Sharper, usually; NO
        finite-sample guarantee.  Reported because the comparison between the
        two is itself informative about whether the miscalibration is a
        location error or a shape error.

EXCHANGEABILITY.  Windows are split BY UNIT.  Overlapping windows from one
engine are near-duplicates, so calibrating on windows drawn from the engines
used to fit would report coverage that evaporates on a new engine.  Units are
the exchangeable objects here, and `unit_ids` is required for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def split_units(unit_ids, cal_frac: float = 0.3, seed: int = 0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Index masks (fit, calibration) that never share a unit.

    Returns boolean arrays over WINDOWS; the split itself is over units.
    """
    u = _np(unit_ids).astype(int)
    uniq = np.unique(u)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_cal = max(1, int(round(len(uniq) * cal_frac)))
    cal_units = set(uniq[perm[:n_cal]].tolist())
    is_cal = np.array([int(v) in cal_units for v in u])
    return ~is_cal, is_cal


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    The (1−α)(1+1/n) empirical quantile — the finite-sample-valid one.  Using
    the plain (1−α) quantile silently loses the guarantee for small n, which is
    precisely the regime a fleet of 100 engines lives in.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


@dataclass
class ConformalPredictive:
    """
    A calibrated interval predictor wrapped around a fitted `SurvivalPC`.

    The wrapped circuit is NOT retrained and NOT modified: `predict` still
    returns the exact pmf, and the calibrated interval is added alongside.  So
    everything the density could answer before, it still answers.
    """

    pc: object                         # SurvivalPC (duck-typed: predict/log_pmf)
    alpha: float = 0.10
    mode: str = "cqr"                  # "cqr" | "pit"
    q_hat: float = 0.0
    lo_level: float = 0.05
    hi_level: float = 0.95
    n_cal: int = 0
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # ── calibration ──────────────────────────────────────────────────────

    def calibrate(self, X_cal: torch.Tensor, y_cal: torch.Tensor) -> "ConformalPredictive":
        y = _np(y_cal)
        pred = self.pc.predict(X_cal)
        self.n_cal = len(y)
        if self.mode == "cqr":
            lo, hi = _np(pred["q05"]), _np(pred["q95"])
            scores = np.maximum(lo - y, y - hi)
            self.q_hat = conformal_quantile(scores, self.alpha)
            self.diagnostics = {
                "q_hat": self.q_hat,
                "raw_cal_coverage": float(np.mean((y >= lo) & (y <= hi))),
                "n_cal": self.n_cal,
            }
        elif self.mode == "pit":
            # PIT values under the circuit's exact predictive CDF, then pick the
            # nominal levels whose EMPIRICAL coverage is 1−α.
            u = self._pit(pred["pmf"], y)
            self.lo_level = float(np.quantile(u, self.alpha / 2))
            self.hi_level = float(np.quantile(u, 1.0 - self.alpha / 2))
            self.diagnostics = {
                "pit_mean": float(u.mean()), "pit_sd": float(u.std()),
                "lo_level": self.lo_level, "hi_level": self.hi_level,
                "n_cal": self.n_cal,
            }
        else:
            raise KeyError(f"unknown conformal mode {self.mode!r} (cqr | pit)")
        return self

    def _pit(self, pmf: torch.Tensor, y: np.ndarray) -> np.ndarray:
        """F(y) under the discrete predictive, with within-bin interpolation."""
        P = _np(pmf)
        centers = _np(self.pc.bin_centers())
        edges = np.linspace(0.0, float(self.pc.cap), P.shape[1] + 1)
        F = P.cumsum(1)
        idx = np.clip(np.digitize(y, edges[1:-1]), 0, P.shape[1] - 1)
        lo_edge, hi_edge = edges[idx], edges[idx + 1]
        frac = np.clip((y - lo_edge) / np.maximum(hi_edge - lo_edge, 1e-9), 0, 1)
        below = np.where(idx > 0, F[np.arange(len(y)), np.maximum(idx - 1, 0)], 0.0)
        return np.clip(below + frac * P[np.arange(len(y)), idx], 0.0, 1.0)

    # ── prediction ───────────────────────────────────────────────────────

    def predict(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        pred = self.pc.predict(X)
        out = {"mean": _np(pred["mean"]), "pmf": _np(pred["pmf"])}
        if self.mode == "cqr":
            out["lo"] = _np(pred["q05"]) - self.q_hat
            out["hi"] = _np(pred["q95"]) + self.q_hat
        else:
            P = _np(pred["pmf"])
            centers = _np(self.pc.bin_centers())
            cdf = P.cumsum(1)
            pick = lambda lv: centers[np.clip((cdf < lv).sum(1), 0, P.shape[1] - 1)]
            out["lo"] = pick(self.lo_level)
            out["hi"] = pick(self.hi_level)
        cap = float(self.pc.cap)
        # RUL lives in [0, cap] by the piecewise-linear convention, so the
        # interval is clipped there.  When the raw predictive is already nearly
        # that wide, the conformal widening has nowhere to go and coverage
        # cannot reach 1−α however large q̂ is — a real limitation of
        # conformalising a capped predictive, not a calibration failure.  It is
        # measured rather than hidden.
        n = max(len(out["lo"]), 1)
        self.diagnostics["frac_clipped_lo"] = float(np.mean(out["lo"] < 0.0))
        self.diagnostics["frac_clipped_hi"] = float(np.mean(out["hi"] > cap))
        out["lo"] = np.clip(out["lo"], 0.0, cap)
        out["hi"] = np.clip(out["hi"], 0.0, cap)
        self.diagnostics["frac_full_range"] = float(
            np.mean((out["hi"] - out["lo"]) >= 0.98 * cap))
        return out
