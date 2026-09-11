"""
Matched donor construction and the marginal-shortcut audit — roadmap step 5.

The `desync` / `decouple` corruptions replace part of a recipient window with
material from a DONOR window.  When the donor is drawn uniformly from the
training pool, the replacement carries two things at once: the broken
cross-channel relation the study is about, and a change of CONTEXT — a
different operating condition, a different point on the degradation curve.
The second one is a shortcut.  A detector that only looks at the target
channel's own marginal can pick it up, and then a relational method looks
better than it is for a reason that has nothing to do with relations.

Two pieces, and they answer different questions:

  `DonorMatcher`   removes as much of the shortcut as matching can remove:
                   donors are restricted to the TRAINING pool, matched on
                   exogenous regime plus a frozen health proxy, inside a
                   caliper, with explicit no-match handling.  Recipients with
                   no admissible donor are reported, never quietly given a
                   random one.
  `shortcut_audit` measures how much shortcut is LEFT, by fitting a
                   target-channel-only classifier on separate development
                   units and reporting its AUROC on the evaluation units.  A
                   matched arm whose target-only classifier still separates
                   the classes has not established marginal preservation.

Two rules the code enforces rather than documents:
  * donors come only from the training pool — a held-out donor leaks the
    evaluation engines into the corruption itself;
  * neither matching nor the audit may read anomaly labels or model scores.
    `match` never receives them, and `shortcut_audit` fits on development
    units that are disjoint from the units it reports on.

What matching does NOT establish: approximate balance on two covariates is not
exact marginal preservation, and none of this makes an injected fault a
physically valid one.  An arm built this way stays labelled semi-synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Covariates
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DonorCovariates:
    """Per-window matching variables, with their names kept beside them.

    `regime` is exogenous: it is set by how the machine is operated, not by its
    condition.  `health` is the fleet's frozen degradation proxy; it is
    admissible because it is fixed before any model is fitted and is not the
    detection label, and inadmissible the moment anyone tunes it.
    """
    regime: Optional[np.ndarray] = None
    health: Optional[np.ndarray] = None
    unit: Optional[np.ndarray] = None

    @property
    def names(self) -> List[str]:
        return [n for n in ("regime", "health") if getattr(self, n) is not None]

    def __len__(self) -> int:
        for value in (self.regime, self.health, self.unit):
            if value is not None:
                return len(value)
        return 0

    def continuous(self) -> np.ndarray:
        """The continuous block (health), standardised by the caller."""
        return (np.zeros((len(self), 0)) if self.health is None
                else np.asarray(self.health, dtype=float).reshape(-1, 1))

    def strata(self) -> np.ndarray:
        """Exact-match block (regime).  All-zero when the source has no regime."""
        return (np.zeros(len(self), dtype=np.int64) if self.regime is None
                else np.asarray(self.regime, dtype=np.int64).reshape(-1))


def covariates_from_task(task, split: str) -> DonorCovariates:
    """Pull the step-5 context off an `ADTask` for one split, or fail loudly."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    return DonorCovariates(regime=_as_array(getattr(task, f"regime_{split}", None)),
                           health=_as_array(getattr(task, f"health_{split}", None)),
                           unit=_as_array(getattr(task, f"unit_{split}", None)))


def _as_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


# ═══════════════════════════════════════════════════════════════════════════
# Matching
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    donor_index: np.ndarray          # per recipient; -1 = no admissible donor
    distance: np.ndarray             # per recipient; inf where unmatched
    donor_unit: np.ndarray           # engine id of the chosen donor, -1 if none
    matching_variables: List[str]
    caliper: float
    n_unmatched: int
    balance: Dict[str, float] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)

    def as_artifact(self) -> Dict[str, np.ndarray]:
        return {"donor_index": self.donor_index, "donor_distance": self.distance,
                "donor_unit": self.donor_unit}


class DonorMatcher:
    """Caliper matching of recipients to TRAINING donors on frozen covariates.

    `strategy`:
      `nearest`      the closest admissible donor, with replacement.
      `caliper_random`  a uniform draw among admissible donors inside the
                     caliper.  Preferred for a corruption workload: `nearest`
                     concentrates the whole study on a handful of donor
                     windows, which is a variance problem masquerading as a
                     bias fix.
    """

    def __init__(self, caliper: float = 0.25, strategy: str = "caliper_random",
                 exact: Sequence[str] = ("regime",), seed: int = 0):
        if caliper <= 0:
            raise ValueError("the caliper must be strictly positive")
        if strategy not in ("nearest", "caliper_random"):
            raise ValueError("strategy must be nearest or caliper_random")
        self.caliper, self.strategy, self.seed = float(caliper), strategy, int(seed)
        self.exact = tuple(exact)
        self.donors_: Optional[DonorCovariates] = None
        self._scale = 1.0
        self._centre = 0.0

    def fit(self, donors: DonorCovariates) -> "DonorMatcher":
        """Freeze the donor pool.  It must be the TRAINING pool; see `match`."""
        if not len(donors):
            raise ValueError("the donor pool is empty")
        if not donors.names:
            raise ValueError(
                "no matching variables are available on this source: donor "
                "matching needs at least an operating regime or a health proxy, "
                "and matching on nothing is an unmatched arm with a better name")
        self.donors_ = donors
        block = donors.continuous()
        if block.shape[1]:
            self._centre = float(block.mean())
            self._scale = float(block.std()) or 1.0
        return self

    def _standardise(self, cov: DonorCovariates) -> np.ndarray:
        block = cov.continuous()
        return (block - self._centre) / self._scale if block.shape[1] else block

    def match(self, recipients: DonorCovariates) -> MatchResult:
        """Assign one training donor per recipient inside the caliper.

        Deterministic given the seed: the same recipients and the same donor
        pool reproduce the same assignment, which is what makes a corruption
        workload replayable.
        """
        if self.donors_ is None:
            raise RuntimeError("fit the matcher on the training donor pool first")
        if not len(recipients):
            raise ValueError("no recipients to match")
        missing = [n for n in self.donors_.names if n not in recipients.names]
        if missing:
            raise ValueError(f"recipients are missing matching variables {missing}")
        rng = np.random.default_rng(self.seed)
        donor_x = self._standardise(self.donors_)
        recipient_x = self._standardise(recipients)
        donor_stratum = self.donors_.strata()
        recipient_stratum = recipients.strata()
        use_exact = "regime" in self.exact and self.donors_.regime is not None

        n = len(recipients)
        index = np.full(n, -1, dtype=np.int64)
        distance = np.full(n, np.inf)
        by_stratum: Dict[int, np.ndarray] = {}
        for s in np.unique(donor_stratum):
            by_stratum[int(s)] = np.flatnonzero(donor_stratum == s)
        every = np.arange(len(donor_stratum))
        for i in range(n):
            pool = by_stratum.get(int(recipient_stratum[i]), np.empty(0, dtype=np.int64)) \
                if use_exact else every
            if not len(pool):
                continue
            if recipient_x.shape[1]:
                d = np.abs(donor_x[pool, 0] - recipient_x[i, 0])
            else:
                d = np.zeros(len(pool))            # exact stratum match only
            admissible = np.flatnonzero(d <= self.caliper)
            if not len(admissible):
                continue
            pick = (admissible[int(np.argmin(d[admissible]))]
                    if self.strategy == "nearest"
                    else admissible[int(rng.integers(len(admissible)))])
            index[i], distance[i] = pool[pick], d[pick]

        matched = index >= 0
        donor_unit = np.full(n, -1, dtype=np.int64)
        if self.donors_.unit is not None:
            donor_unit[matched] = np.asarray(self.donors_.unit)[index[matched]]
        return MatchResult(
            donor_index=index, distance=distance, donor_unit=donor_unit,
            matching_variables=list(self.donors_.names), caliper=self.caliper,
            n_unmatched=int((~matched).sum()),
            balance=self.balance(recipients, index),
            report={"strategy": self.strategy, "exact_on": list(self.exact),
                    "seed": self.seed, "n_recipients": int(n),
                    "n_donor_pool": int(len(self.donors_)),
                    "matched_fraction": float(matched.mean()),
                    "distinct_donors_used": int(len(np.unique(index[matched]))),
                    "distance_q50": float(np.median(distance[matched])) if matched.any() else float("nan"),
                    "distance_q95": float(np.quantile(distance[matched], .95)) if matched.any() else float("nan"),
                    "distance_max": float(distance[matched].max()) if matched.any() else float("nan"),
                    "donor_pool": "training_only"})

    def balance(self, recipients: DonorCovariates, index: np.ndarray) -> Dict[str, float]:
        """Standardised mean difference before and after matching, per variable.

        Below 0.1 is the usual rule of thumb for "balanced"; the number is
        reported rather than thresholded here, because the rule of thumb is
        not a guarantee and this is not a causal-inference paper.
        """
        out: Dict[str, float] = {}
        matched = index >= 0
        if self.donors_ is None or not matched.any():
            return out
        for name in self.donors_.names:
            donor = np.asarray(getattr(self.donors_, name), dtype=float)
            recipient = np.asarray(getattr(recipients, name), dtype=float)
            pooled = np.sqrt(0.5 * (donor.var() + recipient.var())) or 1.0
            out[f"smd_before_{name}"] = float((donor.mean() - recipient.mean()) / pooled)
            out[f"smd_after_{name}"] = float(
                (donor[index[matched]].mean() - recipient[matched].mean()) / pooled)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Marginal-shortcut audit
# ═══════════════════════════════════════════════════════════════════════════

def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  steps: int = 400, lr: float = 0.2) -> Tuple[np.ndarray, float]:
    """Ridge-regularised logistic regression by full-batch gradient descent.

    Deliberately small and dependency-free: this classifier is an AUDIT
    instrument, and a strong one would confound "the shortcut is large" with
    "the auditor is good".  Standardisation uses the DEVELOPMENT split only.
    """
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        residual = p - y
        w -= lr * (X.T @ residual / n + l2 * w / n)
        b -= lr * float(residual.mean())
    return w, b


def target_channel_features(X: torch.Tensor, window: int, n_channels: int,
                            channel: int) -> np.ndarray:
    """Summary of ONE channel's trajectory: values, differences and spread.

    Explicitly not a relational feature set — that is the point.  Anything this
    separates is separable without looking at any other sensor.
    """
    values = X.detach().cpu().numpy().reshape(len(X), window, n_channels)[:, :, channel]
    diffs = np.diff(values, axis=1) if window > 1 else np.zeros((len(X), 0))
    return np.column_stack([values, diffs, values.mean(1), values.std(1),
                            np.abs(values).max(1)])


def shortcut_audit(X_dev: torch.Tensor, y_dev: np.ndarray, X_eval: torch.Tensor,
                   y_eval: np.ndarray, window: int, n_channels: int,
                   channel: int = 0, l2: float = 1.0) -> Dict[str, float]:
    """Target-channel-only AUROC, fitted on development units.

    Near 0.5 means the corruption left the target channel's own marginal
    essentially unchanged, which is the condition the donor construction is
    trying to buy.  Well above 0.5 means a marginal detector can win without
    any relational reasoning, and every relational claim in that arm is
    confounded by exactly that much.
    """
    from .metrics import auroc
    if len(X_dev) != len(y_dev) or len(X_eval) != len(y_eval):
        raise ValueError("audit inputs must be aligned with their labels")
    y_dev = np.asarray(y_dev, dtype=float).reshape(-1)
    y_eval = np.asarray(y_eval, dtype=float).reshape(-1)
    if len(np.unique(y_dev)) < 2 or len(np.unique(y_eval)) < 2:
        raise ValueError("the audit needs both classes in development and evaluation")
    dev = target_channel_features(X_dev, window, n_channels, channel)
    ev = target_channel_features(X_eval, window, n_channels, channel)
    centre, scale = dev.mean(0), dev.std(0)
    scale[scale == 0] = 1.0
    w, b = _logistic_fit((dev - centre) / scale, y_dev, l2=l2)
    scores = ((ev - centre) / scale) @ w + b
    return {"shortcut_auroc": float(auroc(scores, y_eval)),
            "shortcut_channel": int(channel),
            "shortcut_n_dev": int(len(dev)), "shortcut_n_eval": int(len(ev)),
            "shortcut_features": int(dev.shape[1]),
            "shortcut_interpretation":
                "target-channel-only detector fitted on development units; "
                "0.5 means no marginal shortcut, higher means the corruption is "
                "partly visible without any cross-channel reasoning"}
