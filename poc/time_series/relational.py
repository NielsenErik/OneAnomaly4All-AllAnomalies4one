"""
Relational diagnosis under partial observation — Tier 1.5 and 1.6.

Everything here is a QUERY on an already-trained channel-blocked circuit
(`WindowPC(vtree_method="channel_blocked")`).  Nothing trains, nothing
approximates, nothing samples.  Two pieces:

  subset search (1.5)      which SET of channels is relationally broken, by
                           beam search over channel subsets with

                               R_S(x) = log p(x_S) + log p(x_-S) − log p(x)

                           as the value function.  Every candidate at a level
                           is scored in ONE batched pass over the part of the
                           circuit above the channel boundary, reusing the
                           inside values computed once for the window.  That is
                           the concrete separation from Lüdtke, Bartelt &
                           Stuckenschmidt (2022), *Outlier Explanation via
                           Sum-Product Networks* (arXiv:2207.08414), whose
                           search re-queries the circuit per candidate subset:
                           same statistic class, different cost curve.
                           `naive_subset_search` implements their cost model on
                           this same circuit so the contrast isolates the
                           ALGORITHM rather than the model class.

  mask calibration (1.6)   the null distribution of R_c depends on which OTHER
                           sensors are present — conditioning on fewer
                           variables changes the spread of the statistic — so a
                           single threshold does not keep its false-alarm rate
                           as sensors drop out.  `MaskCalibrator` estimates the
                           per-mask null from HEALTHY VALIDATION windows and
                           gives back a threshold per (mask, channel).  It is
                           cheap here because every mask reuses one pass over
                           the leaves; the same calibration for a Gaussian /
                           GMM block conditional needs a fresh factorization
                           per mask, and that asymmetry is a measured number
                           (`QueryCost`), never an asymptotic argument.

What is NOT claimed: R_S is not new, it is not non-negative, and it is not an
additive allocation of the anomaly score.  The claim is the cost curve and the
missing-sensor operating regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.probabilistic_circuits import QueryCost, RelationalCircuit


# ═══════════════════════════════════════════════════════════════════════════
# Mask libraries
# ═══════════════════════════════════════════════════════════════════════════
#
# A mask is a (C,) bool vector, True = sensor OBSERVED.  Patterns are named so
# a result table can say WHICH failure mode it is reporting rather than "mask
# 7", and so the same library can be replayed across methods.

def full_mask(n_channels: int) -> torch.Tensor:
    return torch.ones(n_channels, dtype=torch.bool)


def random_k_masks(n_channels: int, k: int, n: int = 4,
                   seed: int = 0) -> Dict[str, torch.Tensor]:
    """`n` distinct patterns with exactly `k` sensors dead, uniformly chosen."""
    rng = np.random.default_rng(seed)
    out: Dict[str, torch.Tensor] = {}
    seen: set = set()
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        dead = tuple(sorted(rng.choice(n_channels, size=min(k, n_channels - 1),
                                       replace=False).tolist()))
        if dead in seen:
            continue
        seen.add(dead)
        m = full_mask(n_channels)
        m[list(dead)] = False
        out[f"random{k}:{'-'.join(map(str, dead))}"] = m
    return out


def bank_masks(channel_groups: Sequence[Sequence[int]],
               n_channels: int, max_dead_frac: float = 0.5
               ) -> Dict[str, torch.Tensor]:
    """
    One pattern per sensor BANK: a whole physically-coupled group drops out at
    once.  The adversarial case for a relational statistic — the sensors that
    are most informative about each other disappear together, so what is left
    is the weakest evidence the model has.

    `max_dead_frac` drops groups that take out more than half the sensors.
    Not cosmetic: the C-MAPSS loader's grouping is lopsided (on FD001, group 0
    holds 12 of the 15 surviving channels), and a "bank failure" that leaves
    three sensors is a blackout, not a fault — it scores the model on a
    question no method could answer and drowns the informative masks in the
    same table.  Groups above the cap are TRUNCATED to their first half rather
    than dropped, so every bank still appears.
    """
    out: Dict[str, torch.Tensor] = {}
    cap = max(1, int(n_channels * max_dead_frac))
    for gi, g in enumerate(channel_groups):
        g = [int(c) for c in g if 0 <= int(c) < n_channels]
        if not g:
            continue
        g = sorted(g)[:cap]
        m = full_mask(n_channels)
        m[g] = False
        out[f"bank{gi}:{'-'.join(map(str, g))}"] = m
    return out


def mask_library(n_channels: int,
                 channel_groups: Optional[Sequence[Sequence[int]]] = None,
                 ks: Sequence[int] = (1, 2, 3),
                 n_per_k: int = 3, seed: int = 0) -> Dict[str, torch.Tensor]:
    """The default set: everything observed, random-k dropouts, sensor banks."""
    lib: Dict[str, torch.Tensor] = {"full": full_mask(n_channels)}
    for k in ks:
        if k < n_channels:
            lib.update(random_k_masks(n_channels, k, n_per_k, seed=seed + k))
    if channel_groups:
        lib.update(bank_masks(channel_groups, n_channels))
    return lib


def masks_as_tensor(lib: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
    names = list(lib)
    return names, torch.stack([lib[n] for n in names])


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1.5 — subset search under partial observation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubsetSearchResult:
    """Per-window best channel subset and the search's measured cost."""
    subsets: List[List[int]]              # best S per window
    values: np.ndarray                    # (N,) R_S at that subset
    per_size: Dict[int, np.ndarray]       # size -> (N,) best R at that size
    candidates_scored: int                # distinct (subset) evaluations
    cost: QueryCost = field(default_factory=QueryCost)


def _observed_list(mask: Optional[torch.Tensor], n_channels: int) -> List[int]:
    if mask is None:
        return list(range(n_channels))
    m = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    return [c for c in range(n_channels) if bool(m[c])]


@torch.no_grad()
def subset_search(
    rc: RelationalCircuit,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_size: int = 3,
    beam: int = 4,
) -> SubsetSearchResult:
    """
    Beam search for the channel subset with the largest R_S, per window.

    `mask` is ONE observation pattern (C,) shared by the batch: subsets are
    drawn from the observed sensors only, and both terms of R_S are computed
    inside that pattern, so the statistic means the same thing whether or not
    a sensor is missing.

    The bound that matters: a level of the search costs one batched evaluation
    of the circuit ABOVE the channel boundary per candidate, and the boundary
    values — the leaves, i.e. most of the circuit — are computed ONCE for the
    whole search.  `naive_subset_search` below spends a full circuit pass per
    candidate instead, which is the 2022 SPN-explanation cost model.
    """
    cost = QueryCost()
    block_vals = rc.block_log_values(x, cost)
    n = block_vals[0].shape[0]
    obs = _observed_list(mask, rc.n_blocks)
    if not obs:
        raise ValueError("every sensor is masked out; there is nothing to search")
    max_size = max(1, min(int(max_size), len(obs)))

    beams: List[List[Tuple[float, Tuple[int, ...]]]] = [[] for _ in range(n)]
    best_val = np.full(n, -np.inf)
    best_set: List[Tuple[int, ...]] = [tuple() for _ in range(n)]
    per_size: Dict[int, np.ndarray] = {}
    frontier: List[List[Tuple[int, ...]]] = [[tuple()] for _ in range(n)]
    scored = 0

    for size in range(1, max_size + 1):
        # union of every window's candidate expansions -> ONE batched scoring
        pool: List[Tuple[int, ...]] = []
        index: Dict[Tuple[int, ...], int] = {}
        per_window: List[List[Tuple[int, ...]]] = []
        for i in range(n):
            cands = []
            for base in frontier[i]:
                for c in obs:
                    if c in base:
                        continue
                    s = tuple(sorted(base + (c,)))
                    if s not in index:
                        index[s] = len(pool)
                        pool.append(s)
                    cands.append(s)
            per_window.append(sorted(set(cands)))
        if not pool:
            break
        vals = rc.block_relational_value(
            [list(s) for s in pool], block_vals=block_vals,
            observed=mask if mask is not None else None, cost=cost).cpu().numpy()
        scored += len(pool)

        size_best = np.full(n, -np.inf)
        for i in range(n):
            scores = [(float(vals[i, index[s]]), s) for s in per_window[i]]
            if not scores:
                continue
            scores.sort(key=lambda t: -t[0])
            beams[i] = scores[:beam]
            frontier[i] = [s for _, s in beams[i]]
            size_best[i] = scores[0][0]
            if scores[0][0] > best_val[i]:
                best_val[i], best_set[i] = scores[0][0], scores[0][1]
        per_size[size] = size_best

    return SubsetSearchResult(
        subsets=[list(s) for s in best_set], values=best_val,
        per_size=per_size, candidates_scored=scored, cost=cost)


@torch.no_grad()
def naive_subset_search(
    pc, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
    max_size: int = 3, beam: int = 4,
) -> SubsetSearchResult:
    """
    The same search with the 2022 cost model: every candidate subset is a fresh
    pair of marginal queries over the WHOLE circuit (`log_marginal`), with no
    reuse between candidates.

    Identical statistic, identical beam, identical answers — the only
    difference is what it costs, which is the point of having it.  Used as the
    correctness reference in the tests and as the Tier 2 cost baseline.
    """
    W, C = pc.window, pc.n_channels
    feats = lambda cs: [t * C + c for t in range(W) for c in cs]
    Xp = pc._prep(x)
    obs = _observed_list(mask, C)
    dead = [c for c in range(C) if c not in obs]
    max_size = max(1, min(int(max_size), len(obs)))
    cost = QueryCost()

    with torch.no_grad():
        base = pc.pc.log_marginal(Xp, feats(dead)) if dead else pc.pc.log_prob(Xp)
    cost.passes += 1
    n = len(Xp)
    cache: Dict[Tuple[int, ...], np.ndarray] = {}

    def value(S: Tuple[int, ...]) -> np.ndarray:
        hit = cache.get(S)
        if hit is not None:
            return hit
        rest = [c for c in obs if c not in S]
        with torch.no_grad():
            lp_S = pc.pc.log_marginal(Xp, feats(dead + rest))
            lp_R = pc.pc.log_marginal(Xp, feats(dead + list(S)))
        cost.passes += 2
        out = (lp_S + lp_R - base).cpu().numpy()
        cache[S] = out
        return out

    best_val = np.full(n, -np.inf)
    best_set: List[Tuple[int, ...]] = [tuple() for _ in range(n)]
    per_size: Dict[int, np.ndarray] = {}
    frontier: List[List[Tuple[int, ...]]] = [[tuple()] for _ in range(n)]
    for size in range(1, max_size + 1):
        size_best = np.full(n, -np.inf)
        cands_per_window = []
        for i in range(n):
            cands = sorted({tuple(sorted(b + (c,)))
                            for b in frontier[i] for c in obs if c not in b})
            cands_per_window.append(cands)
        for S in sorted({s for cs in cands_per_window for s in cs}):
            value(S)
        for i in range(n):
            scored = sorted(((float(value(S)[i]), S) for S in cands_per_window[i]),
                            key=lambda t: -t[0])
            if not scored:
                continue
            frontier[i] = [s for _, s in scored[:beam]]
            size_best[i] = scored[0][0]
            if scored[0][0] > best_val[i]:
                best_val[i], best_set[i] = scored[0][0], scored[0][1]
        per_size[size] = size_best
    return SubsetSearchResult(subsets=[list(s) for s in best_set], values=best_val,
                              per_size=per_size, candidates_scored=len(cache),
                              cost=cost)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1.6 — mask-conditional calibration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MaskCalibration:
    """Thresholds for one mask: per channel, plus the worst-channel statistic."""
    mask: torch.Tensor
    per_channel: np.ndarray               # (C,) threshold, nan where unobserved
    worst_channel: float                  # threshold on max_c R_c
    n: int


class MaskCalibrator:
    """
    Per-mask null distribution of R_c from HEALTHY VALIDATION windows.

    Why it is needed: R_c compares a channel against the rest, and "the rest"
    changes when sensors drop out, so the null distribution moves with the
    mask.  A threshold fitted on the full sensor set therefore does not keep
    its false-alarm rate when a sensor fails — which is the operating regime
    the whole claim is about.  `report()` measures both, so the drift is a
    number rather than an assertion.

    Why it is cheap HERE: the boundary values of the validation windows are
    computed once, and each mask is then a substitution plus one pass over the
    circuit above the boundary.  The same calibration for a block-Gaussian or
    GMM conditional needs a fresh Schur complement per mask (Tier 2 measures
    it on the same masks).

    Quantiles are the finite-sample CONSERVATIVE ones (⌈(n+1)(1−α)⌉/n, the
    split-conformal convention already used in `conformal.py`), so a threshold
    from 200 validation windows is not silently anti-conservative.
    """

    def __init__(self, rc: RelationalCircuit, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        self.rc = rc
        self.alpha = float(alpha)
        self.cal: Dict[str, MaskCalibration] = {}
        self.pooled: Optional[MaskCalibration] = None
        self.cost = QueryCost()

    # ── fitting ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, x_healthy: torch.Tensor,
            masks: Dict[str, torch.Tensor]) -> "MaskCalibrator":
        self._require_windows(x_healthy, "fit")
        if not masks:
            raise ValueError("at least one calibration mask is required")
        self.cal = {}
        self.pooled = None
        self.cost = QueryCost()
        block_vals = self.rc.block_log_values(x_healthy, self.cost)
        n = block_vals[0].shape[0]
        stacked = []
        for name, m in masks.items():
            pattern = torch.as_tensor(m, dtype=torch.bool)
            if pattern.shape != (self.rc.n_blocks,) or not bool(pattern.any()):
                raise ValueError("calibration masks must observe at least one complete channel")
            mm = torch.as_tensor(m, dtype=torch.bool).reshape(1, -1)
            mm = mm.expand(n, self.rc.n_blocks).to(block_vals[0].device)
            res = self.rc.relational_map(block_vals=block_vals, mask=mm)
            self.cost = self.cost + res.cost
            R = res.R.cpu().numpy()
            if not np.isfinite(R[:, pattern.cpu().numpy()]).all():
                raise FloatingPointError("non-finite calibration score on an observed channel")
            self.cal[name] = MaskCalibration(
                mask=torch.as_tensor(m, dtype=torch.bool).clone(),
                per_channel=self._quantile(R),
                worst_channel=float(self._quantile(
                    np.nanmax(R, axis=1, keepdims=True))[0]),
                n=n)
            stacked.append(R)
        # the mask-BLIND alternative: one threshold from the pooled nulls.  It
        # is the thing the mask-conditional threshold has to beat, and it is
        # what a practitioner does by default.
        allR = np.concatenate(stacked, axis=0)
        self.pooled = MaskCalibration(
            mask=torch.ones(self.rc.n_blocks, dtype=torch.bool),
            per_channel=self._quantile(allR),
            worst_channel=float(self._quantile(
                np.nanmax(allR, axis=1, keepdims=True))[0]),
            n=len(allR))
        return self

    @staticmethod
    def _require_windows(x: torch.Tensor, what: str) -> None:
        """An empty split gives thresholds of nan and a false-alarm rate of
        0/0, which reads as a perfect result.  Refuse it."""
        if x is None or not len(x):
            raise ValueError(
                f"MaskCalibrator.{what} got no windows: the healthy validation "
                "split is empty, and every rate computed from it would be 0/0")

    def _quantile(self, R: np.ndarray) -> np.ndarray:
        """Conservative (1−α) quantile per column, nan-aware."""
        out = np.full(R.shape[1], np.nan)
        for c in range(R.shape[1]):
            col = R[:, c]
            col = col[np.isfinite(col)]
            if not len(col):
                continue
            k = int(np.ceil((len(col) + 1) * (1 - self.alpha)))
            out[c] = float(np.sort(col)[k - 1]) if k <= len(col) else float("inf")
        return out

    # ── use ──────────────────────────────────────────────────────────────

    def threshold(self, name: str) -> np.ndarray:
        if name not in self.cal:
            raise KeyError(f"no calibration for mask {name!r}; "
                           f"have {sorted(self.cal)}")
        return self.cal[name].per_channel

    def alarms(self, R: np.ndarray, name: Optional[str] = None,
               pooled: bool = False) -> np.ndarray:
        """(N, C) bool: which per-channel relational alarms fire."""
        thr = (self.pooled.per_channel if pooled or name is None
               else self.threshold(name))
        with np.errstate(invalid="ignore"):
            return np.asarray(R > thr[None, :], dtype=bool) & np.isfinite(R)

    def familywise_alarms(self, R: np.ndarray, name: str) -> np.ndarray:
        """Window alarm calibrated on the maximum over observed channels.

        Its guarantee still requires exchangeable calibration/test objects;
        per-channel thresholds do not provide this familywise control.
        """
        if name not in self.cal:
            raise KeyError(f"no calibration for mask {name!r}")
        values = np.asarray(R)
        valid = np.isfinite(values)
        maximum = np.where(valid, values, -np.inf).max(axis=1)
        return valid.any(axis=1) & (maximum > self.cal[name].worst_channel)

    @torch.no_grad()
    def report(self, x_healthy: torch.Tensor,
               masks: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Measured false-alarm rate per mask, mask-conditional vs mask-blind, on
        windows the thresholds were NOT fitted on.  One row per mask; the
        column that matters is how far `fpr_blind` wanders from alpha while
        `fpr_masked` stays on it.
        """
        self._require_windows(x_healthy, "report")
        block_vals = self.rc.block_log_values(x_healthy, self.cost)
        n = block_vals[0].shape[0]
        rows = []
        for name, m in masks.items():
            if name not in self.cal or not torch.equal(
                    torch.as_tensor(m, dtype=torch.bool).cpu(), self.cal[name].mask.cpu()):
                raise ValueError(f"mask {name!r} does not match its calibration pattern")
            mm = torch.as_tensor(m, dtype=torch.bool).reshape(1, -1)
            mm = mm.expand(n, self.rc.n_blocks).to(block_vals[0].device)
            res = self.rc.relational_map(block_vals=block_vals, mask=mm)
            self.cost = self.cost + res.cost
            R = res.R.cpu().numpy()
            obs = np.isfinite(R)
            a_mask = self.alarms(R, name)
            a_blind = self.alarms(R, pooled=True)
            rows.append({
                "mask": name,
                "n_observed_channels": int(torch.as_tensor(m).sum()),
                "n": int(n),
                "fpr_masked": float(a_mask.sum() / max(obs.sum(), 1)),
                "fpr_blind": float(a_blind.sum() / max(obs.sum(), 1)),
                "window_fpr_masked": float(a_mask.any(axis=1).mean()),
                "window_fpr_blind": float(a_blind.any(axis=1).mean()),
                "familywise_fpr": float(self.familywise_alarms(R, name).mean()),
                "alpha": self.alpha,
            })
        return rows
