"""
Data layer for the time-series PoC.

Provides a C-MAPSS-shaped SYNTHETIC fleet simulator plus an optional loader for
the real C-MAPSS text files, behind one interface.  Synthetic is the default so
the PoC runs offline and deterministically; every structural feature that the
proposal depends on is reproduced deliberately:

  * run-to-failure trajectories with per-unit initial wear and degradation rate
    (so units are not exchangeable, as in a real fleet);
  * OPERATING REGIMES that shift sensor offsets/gains — the ground-truth latent
    the routing queries are supposed to recover (C-MAPSS FD002/FD004 have 6);
  * CHANNEL GROUPS with correlated noise, so anomalies are not all trivially
    univariate — the failure mode that arXiv:2606.02670 documents for
    SMAP/MSL/SMD/SWaT and that would make a joint density model pointless;
  * dead / non-responsive sensors (C-MAPSS has several);
  * RIGHT CENSORING: a configurable fraction of units is truncated before
    failure.  This is the whole point of the RUL experiment — those units carry
    information ("still alive at time c") that a plain regressor cannot use.

Anomalies for the detection task come in two kinds, reported separately:
  injected  — point spikes, channel offsets, collective drifts (the usual
              synthetic protocol, and the one a per-channel z-score handles);
  organic   — genuine late-degradation windows, never injected, which is the
              anomaly type an operator actually cares about.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Fleet container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Fleet:
    """A set of run-to-failure trajectories."""
    series: List[np.ndarray]              # per unit: (T_i, C) sensor readings
    rul: List[np.ndarray]                 # per unit: (T_i,) cycles until failure
    regime: List[np.ndarray]              # per unit: (T_i,) operating-regime id
    health: List[np.ndarray]              # per unit: (T_i,) latent health in [0,1]
    censored: List[bool]                  # per unit: True = truncated before failure
    n_channels: int
    n_regimes: int
    channel_groups: List[List[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.series)

    def __repr__(self) -> str:
        lens = [len(s) for s in self.series]
        return (f"Fleet(units={len(self)}, C={self.n_channels}, "
                f"regimes={self.n_regimes}, cycles={min(lens)}–{max(lens)}, "
                f"censored={sum(self.censored)}/{len(self)})")


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic simulator
# ═══════════════════════════════════════════════════════════════════════════

def simulate_fleet(
    n_units: int = 60,
    n_channels: int = 14,
    n_regimes: int = 3,
    n_groups: int = 3,
    min_life: int = 120,
    max_life: int = 300,
    censor_frac: float = 0.35,
    dead_channels: int = 2,
    noise: float = 0.10,
    group_noise: float = 0.25,
    phi_ar: float = 0.85,
    seed: int = 0,
) -> Fleet:
    """
    Simulate a fleet of degrading machines.

    Health h(t) rises from a random initial wear to 1 (failure) along a
    unit-specific convex trajectory h(t) = h0 + (1-h0)·(t/T)^p.  Sensor j reads

        x_j(t) = offset_j[regime(t)] + gain_j[regime(t)] · a_j · h(t)^q_j
                 + group_noise_g(j)(t) + noise_j(t)

    so (i) the regime shifts the operating point discontinuously, (ii) the
    degradation signal is shared across responsive sensors — hence genuinely
    MULTIVARIATE — and (iii) channels within a group share a noise process,
    creating cross-channel dependence that exists in the NORMAL data too (which
    is what a joint density model must exploit to beat per-channel scoring).
    """
    rng = np.random.default_rng(seed)

    groups = [list(g) for g in np.array_split(np.arange(n_channels), n_groups)]
    responsive = np.ones(n_channels, dtype=bool)
    if dead_channels > 0:
        responsive[rng.choice(n_channels, size=dead_channels, replace=False)] = False

    # per-channel degradation response and per-regime operating point
    amp = rng.uniform(0.8, 2.5, size=n_channels) * responsive
    expo = rng.uniform(1.0, 3.0, size=n_channels)
    sign = rng.choice([-1.0, 1.0], size=n_channels)
    offset = rng.normal(0.0, 2.0, size=(n_regimes, n_channels))
    gain = rng.uniform(0.7, 1.3, size=(n_regimes, n_channels))

    series, ruls, regimes, healths, censored = [], [], [], [], []
    for u in range(n_units):
        T = int(rng.integers(min_life, max_life + 1))
        h0 = rng.uniform(0.0, 0.25)
        p = rng.uniform(1.5, 4.0)
        t = np.arange(T)
        h = h0 + (1.0 - h0) * (t / max(T - 1, 1)) ** p

        # regimes persist in blocks (a machine does not change duty every cycle)
        r = np.zeros(T, dtype=int)
        pos = 0
        while pos < T:
            blk = int(rng.integers(5, 25))
            r[pos:pos + blk] = rng.integers(0, n_regimes)
            pos += blk

        # Cross-channel coupling inside each group, deliberately NONLINEAR.
        # A purely linear/Gaussian coupling would make a full-covariance
        # Gaussian (Mahalanobis) the optimal detector, and the whole benchmark
        # would measure nothing: tanh and squared responses put dependence
        # where a second-order method cannot reach it.
        #
        # The latent driver is AR(1) with a long memory (phi_ar), NOT white
        # noise smoothed by a 3-tap kernel.  This matters more than it looks:
        # with weak within-window memory, a window is nearly exchangeable in
        # time, so PERMUTING its timesteps produces a statistically identical
        # window and the `decouple` anomaly is undetectable by any method —
        # the benchmark would be silently vacuous.  phi_ar sets the timescale
        # relative to the window length and should stay well above 0.
        g_noise = np.zeros((T, n_channels))
        for g in groups:
            z = np.empty(T)
            z[0] = rng.normal()
            for tt in range(1, T):
                z[tt] = phi_ar * z[tt - 1] + math.sqrt(1 - phi_ar ** 2) * rng.normal()
            for jj, c in enumerate(g):
                if jj % 3 == 0:
                    resp = z
                elif jj % 3 == 1:
                    resp = np.tanh(2.0 * z)
                else:
                    resp = z ** 2 - z.var()
                g_noise[:, c] = group_noise * resp

        # per-channel idiosyncratic AR(1), so channels are not merely copies of
        # one group driver and each has its own temporal signature
        idio = np.empty((T, n_channels))
        idio[0] = rng.normal(size=n_channels)
        for tt in range(1, T):
            idio[tt] = (phi_ar * idio[tt - 1]
                        + math.sqrt(1 - phi_ar ** 2) * rng.normal(size=n_channels))

        deg = sign * amp * (h[:, None] ** expo)
        x = (offset[r] + gain[r] * deg + g_noise + 0.35 * idio
             + rng.normal(0.0, noise, size=(T, n_channels)))

        rul = (T - 1 - t).astype(float)
        is_cens = bool(rng.random() < censor_frac)
        if is_cens:
            # truncate somewhere in the second half of life: the unit is still
            # alive when observation stops, so its true RUL is only bounded below
            cut = int(rng.integers(int(0.45 * T), int(0.9 * T)))
            x, rul, r, h = x[:cut], rul[:cut], r[:cut], h[:cut]

        series.append(x.astype(np.float32))
        ruls.append(rul)
        regimes.append(r)
        healths.append(h)
        censored.append(is_cens)

    return Fleet(series, ruls, regimes, healths, censored,
                 n_channels=n_channels, n_regimes=n_regimes, channel_groups=groups)


# ═══════════════════════════════════════════════════════════════════════════
# Windowing / normalisation
# ═══════════════════════════════════════════════════════════════════════════

def windowize(
    x: np.ndarray, window: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding windows of a (T, C) series, flattened row-major so that feature
    index = t·C + c (the layout `time_channel_vtree` assumes).

    Returns (windows (N, window·C), right-edge index of each window (N,)).
    """
    T = x.shape[0]
    if T < window:
        return np.zeros((0, window * x.shape[1]), dtype=np.float32), np.zeros(0, dtype=int)
    starts = np.arange(0, T - window + 1, stride)
    out = np.stack([x[s:s + window].reshape(-1) for s in starts]).astype(np.float32)
    return out, starts + window - 1


class Standardizer:
    """
    Per-channel standardisation, optionally per operating regime.

    Regime-wise normalisation is standard practice on C-MAPSS FD002/FD004 and
    it removes the (uninteresting) regime offsets so the model is judged on
    degradation rather than on duty-cycle bookkeeping.  It is OFF by default
    for the detection task on purpose: leaving the regime multimodality in is
    what forces a detector to actually be a mixture model.
    """

    def __init__(self, per_regime: bool = False):
        self.per_regime = per_regime
        self.mu: Dict[int, np.ndarray] = {}
        self.sd: Dict[int, np.ndarray] = {}

    def fit(self, fleet: Fleet, unit_idx: Sequence[int]) -> "Standardizer":
        X = np.concatenate([fleet.series[u] for u in unit_idx], axis=0)
        R = np.concatenate([fleet.regime[u] for u in unit_idx], axis=0)
        keys = range(fleet.n_regimes) if self.per_regime else [-1]
        for k in keys:
            sel = np.ones(len(X), dtype=bool) if k < 0 else (R == k)
            if sel.sum() < 2:
                sel = np.ones(len(X), dtype=bool)
            self.mu[k] = X[sel].mean(0)
            self.sd[k] = X[sel].std(0) + 1e-6
        return self

    def transform(self, x: np.ndarray, regime: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.per_regime or regime is None:
            return (x - self.mu[-1]) / self.sd[-1]
        out = np.empty_like(x)
        for k in self.mu:
            sel = regime == k
            if sel.any():
                out[sel] = (x[sel] - self.mu[k]) / self.sd[k]
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 — anomaly detection
# ═══════════════════════════════════════════════════════════════════════════

ANOMALY_KINDS = ("spike", "offset", "drift", "decouple", "desync", "organic")

# The first three are MARGINAL anomalies: some channel visits values it should
# not, so per-channel scoring can see them.  `decouple` and `desync` are
# STRUCTURAL: every channel's marginal distribution is left exactly intact and
# only the dependence is broken.  A per-channel detector cannot see them even
# in principle — they are the anomaly type that justifies a joint density at
# all, and the type that arXiv:2606.02670 reports is largely absent from the
# standard MTS benchmarks.


def _inject(x: np.ndarray, kind: str, rng: np.random.Generator,
            strength: float, donor: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, List[int]]:
    """Inject one anomaly into a (T, C) segment; returns (x, affected channels)."""
    T, C = x.shape
    x = x.copy()
    if kind == "spike":                       # point anomaly, 1–2 channels
        ch = rng.choice(C, size=rng.integers(1, 3), replace=False)
        t = rng.integers(0, T)
        x[t, ch] += strength * rng.choice([-1.0, 1.0]) * 4.0
    elif kind == "offset":                    # contextual step in a few channels
        ch = rng.choice(C, size=rng.integers(2, max(3, C // 3)), replace=False)
        t = rng.integers(0, max(T // 2, 1))
        x[t:, ch] += strength * rng.choice([-1.0, 1.0]) * 1.5
    elif kind == "drift":                     # collective ramp across a group
        ch = rng.choice(C, size=rng.integers(2, max(3, C // 2)), replace=False)
        ramp = np.linspace(0, strength * 2.0, T)[:, None]
        x[:, ch] += ramp
    elif kind == "decouple":
        # permute a channel along TIME: its marginal over the window is
        # unchanged (same multiset of values), its temporal structure is gone
        ch = rng.choice(C, size=rng.integers(1, 3), replace=False)
        for c in ch:
            x[:, c] = x[rng.permutation(T), c]
    elif kind == "desync":
        # replace a channel with the SAME channel from another normal window:
        # the marginal stays in-distribution, the cross-channel coupling breaks
        if donor is None:
            raise ValueError("desync needs a donor window")
        ch = rng.choice(C, size=rng.integers(1, max(2, C // 4)), replace=False)
        x[:, ch] = donor.reshape(T, C)[:, ch]
    else:
        raise KeyError(kind)
    return x, sorted(int(c) for c in ch)


@dataclass
class ADTask:
    X_train: torch.Tensor          # (N, window·C) normal-only windows
    X_test: torch.Tensor
    y_test: torch.Tensor           # 1 = anomalous
    kind_test: List[str]           # "normal" | spike | offset | drift | organic
    affected_test: List[List[int]] # GROUND-TRUTH corrupted channels per window
    window: int
    n_channels: int
    channel_groups: List[List[int]]
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ADTask(train={tuple(self.X_train.shape)}, "
                f"test={tuple(self.X_test.shape)}, "
                f"anomaly_rate={float(self.y_test.float().mean()):.1%})")


def make_ad_task(
    window: int = 8,
    stride: int = 2,
    healthy_frac: float = 0.35,
    organic_frac: float = 0.85,
    inject_rate: float = 0.12,
    strength: float = 1.0,
    train_units: float = 0.6,
    seed: int = 0,
    fleet: Optional[Fleet] = None,
    **sim_kwargs,
) -> ADTask:
    """
    Build a semi-supervised AD task: train on healthy windows only, test on a
    contaminated mix.

    healthy_frac: a window counts as NORMAL if its health stays below this.
    organic_frac: a window counts as an ORGANIC anomaly if its health exceeds
                  this (genuine late-stage degradation, nothing injected).
    inject_rate:  fraction of held-out healthy windows that receive a
                  synthetic anomaly.
    Windows between healthy_frac and organic_frac are DISCARDED from the test
    set — their label would be arbitrary, and inventing one is exactly the
    labelling flaw Wu & Keogh (TKDE 2021) fault these benchmarks for.
    """
    rng = np.random.default_rng(seed)
    fleet = fleet or simulate_fleet(seed=seed, **sim_kwargs)
    n_units = len(fleet)
    perm = rng.permutation(n_units)
    n_tr = int(n_units * train_units)
    tr_units, te_units = perm[:n_tr], perm[n_tr:]

    std = Standardizer(per_regime=False).fit(fleet, tr_units)

    def windows_of(u: int) -> Tuple[np.ndarray, np.ndarray]:
        x = std.transform(fleet.series[u], fleet.regime[u])
        W, right = windowize(x, window, stride)
        return W, fleet.health[u][right] if len(right) else np.zeros(0)

    X_train = []
    for u in tr_units:
        W, h = windows_of(u)
        X_train.append(W[h < healthy_frac])
    X_train = np.concatenate([w for w in X_train if len(w)], axis=0)

    donor_pool = X_train                              # in-distribution donors
    X_test, y_test, kinds, affected = [], [], [], []
    for u in te_units:
        x = std.transform(fleet.series[u], fleet.regime[u])
        W, right = windowize(x, window, stride)
        h = fleet.health[u][right] if len(right) else np.zeros(0)
        for i in range(len(W)):
            if h[i] > organic_frac:                       # organic anomaly
                X_test.append(W[i]); y_test.append(1); kinds.append("organic")
                affected.append([])           # degradation is fleet-wide, not localised
            elif h[i] < healthy_frac:
                seg = x[right[i] - window + 1:right[i] + 1]
                if rng.random() < inject_rate:            # injected anomaly
                    kind = str(rng.choice(["spike", "offset", "drift",
                                           "decouple", "desync"]))
                    donor = donor_pool[rng.integers(len(donor_pool))]
                    seg, chans = _inject(seg, kind, rng, strength, donor=donor)
                    X_test.append(seg.reshape(-1)); y_test.append(1); kinds.append(kind)
                    affected.append(chans)    # <- ground truth for explanations
                else:                                     # normal
                    X_test.append(W[i]); y_test.append(0); kinds.append("normal")
                    affected.append([])
            # else: ambiguous mid-life window, deliberately dropped

    return ADTask(
        X_train=torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        X_test=torch.from_numpy(np.asarray(X_test, dtype=np.float32)),
        y_test=torch.tensor(y_test, dtype=torch.long),
        kind_test=kinds,
        affected_test=affected,
        window=window,
        n_channels=fleet.n_channels,
        channel_groups=fleet.channel_groups,
        meta={"train_units": len(tr_units), "test_units": len(te_units),
              "n_regimes": fleet.n_regimes, "seed": seed},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Task 2 — remaining useful life (with right censoring)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RULTask:
    X_train: torch.Tensor          # (N, window·C)
    tau_train: torch.Tensor        # (N,) RUL BIN index (observed or censoring bound)
    delta_train: torch.Tensor      # (N,) 1 = failure observed, 0 = right-censored
    regime_train: torch.Tensor     # (N,) dominant regime in the window
    X_test: torch.Tensor
    tau_test: torch.Tensor         # (N,) true RUL bin — test units are uncensored
    rul_test: torch.Tensor         # (N,) true RUL in cycles
    regime_test: torch.Tensor
    window: int
    n_channels: int
    n_bins: int
    cap: float                     # RUL values are capped at this many cycles
    channel_groups: List[List[int]]
    meta: dict = field(default_factory=dict)

    def bin_centers(self) -> torch.Tensor:
        edges = torch.linspace(0, self.cap, self.n_bins + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def __repr__(self) -> str:
        cens = float(1.0 - self.delta_train.float().mean())
        return (f"RULTask(train={tuple(self.X_train.shape)}, "
                f"test={tuple(self.X_test.shape)}, bins={self.n_bins}, "
                f"cap={self.cap:g}, censored_train={cens:.1%})")


def make_rul_task(
    window: int = 8,
    stride: int = 3,
    n_bins: int = 20,
    cap: float = 130.0,
    train_units: float = 0.65,
    seed: int = 0,
    fleet: Optional[Fleet] = None,
    **sim_kwargs,
) -> RULTask:
    """
    Build a RUL task with right-censored training units.

    RUL is capped at `cap` cycles (the standard piecewise-linear convention —
    a unit 400 cycles from failure looks identical to one 300 cycles away) and
    discretised into `n_bins` ordinal bins, so the τ variable gets a
    CategoricalLeaf and every survival query is an exact suffix sum.

    For a censored unit, `tau_train` is the bin of its LAST OBSERVED RUL bound
    and `delta_train` is 0: the correct likelihood contribution is
    P(τ ≥ that bin), not P(τ = that bin).  A regressor that trains on the
    recorded value instead is systematically biased — it is told the unit died
    at the moment observation stopped.
    """
    rng = np.random.default_rng(seed)
    fleet = fleet or simulate_fleet(seed=seed, **sim_kwargs)
    n_units = len(fleet)
    perm = rng.permutation(n_units)
    n_tr = int(n_units * train_units)
    tr_units, te_units = perm[:n_tr], perm[n_tr:]

    std = Standardizer(per_regime=True).fit(fleet, tr_units)
    edges = np.linspace(0, cap, n_bins + 1)

    def to_bin(rul: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(np.minimum(rul, cap), edges[1:-1]), 0, n_bins - 1)

    def build(units, force_uncensored: bool):
        Xs, taus, deltas, regs, raw = [], [], [], [], []
        for u in units:
            if force_uncensored and fleet.censored[u]:
                continue                     # test set is failure-observed only
            x = std.transform(fleet.series[u], fleet.regime[u])
            W, right = windowize(x, window, stride)
            if not len(W):
                continue
            rul = fleet.rul[u][right]
            Xs.append(W)
            taus.append(to_bin(rul))
            raw.append(np.minimum(rul, cap))
            # a censored UNIT censors every window drawn from it: at the time
            # the window was taken, all that is known is τ ≥ recorded value
            deltas.append(np.full(len(W), 0 if fleet.censored[u] else 1))
            regs.append(fleet.regime[u][right])
        cat = lambda a: np.concatenate(a, axis=0)
        return cat(Xs), cat(taus), cat(deltas), cat(regs), cat(raw)

    Xtr, ttr, dtr, rtr, _ = build(tr_units, force_uncensored=False)
    Xte, tte, _, rte, rawte = build(te_units, force_uncensored=True)

    return RULTask(
        X_train=torch.from_numpy(Xtr.astype(np.float32)),
        tau_train=torch.from_numpy(ttr.astype(np.int64)),
        delta_train=torch.from_numpy(dtr.astype(np.int64)),
        regime_train=torch.from_numpy(rtr.astype(np.int64)),
        X_test=torch.from_numpy(Xte.astype(np.float32)),
        tau_test=torch.from_numpy(tte.astype(np.int64)),
        rul_test=torch.from_numpy(rawte.astype(np.float32)),
        regime_test=torch.from_numpy(rte.astype(np.int64)),
        window=window, n_channels=fleet.n_channels, n_bins=n_bins, cap=cap,
        channel_groups=fleet.channel_groups,
        meta={"train_units": len(tr_units), "test_units": len(te_units),
              "censored_units": int(sum(fleet.censored[u] for u in tr_units)),
              "n_regimes": fleet.n_regimes, "seed": seed},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Optional: real C-MAPSS
# ═══════════════════════════════════════════════════════════════════════════

CMAPSS_COLUMNS = (["unit", "cycle", "op1", "op2", "op3"]
                  + [f"s{i}" for i in range(1, 22)])


def load_cmapss_fleet(
    subset: str = "FD001",
    data_dir: Optional[str] = None,
    n_regimes: int = 6,
) -> Fleet:
    """
    Load a real C-MAPSS training file (`train_FD00x.txt`) as a Fleet.

    Drop the NASA `CMaps` files into data/cmapss/ and this becomes a drop-in
    replacement for simulate_fleet in both tasks.  Training units are
    run-to-failure (uncensored); operating regimes are recovered by clustering
    the three operational settings, which is the standard FD002/FD004
    preprocessing step.

    Not downloaded automatically: the NASA prognostics repository has no stable
    programmatic endpoint, so this is opt-in rather than a silent network call.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = data_dir or os.path.join(root, "data", "cmapss")
    path = os.path.join(data_dir, f"train_{subset}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the NASA C-MAPSS 'CMaps' archive and "
            f"place train_{subset}.txt there, or use simulate_fleet().")

    raw = np.loadtxt(path)
    units = raw[:, 0].astype(int)
    ops, sensors = raw[:, 2:5], raw[:, 5:]

    # regime id by k-means on the operational settings (6 discrete conditions)
    from sklearn.cluster import KMeans
    n_regimes = min(n_regimes, len(np.unique(ops.round(1), axis=0)))
    reg_all = (KMeans(n_clusters=n_regimes, n_init=10, random_state=0)
               .fit_predict(ops) if n_regimes > 1 else np.zeros(len(ops), dtype=int))

    # drop constant sensors (C-MAPSS has several that never move)
    keep = sensors.std(0) > 1e-8
    sensors = sensors[:, keep]

    series, ruls, regimes, healths = [], [], [], []
    for u in np.unique(units):
        sel = units == u
        x = sensors[sel].astype(np.float32)
        T = len(x)
        rul = np.arange(T - 1, -1, -1, dtype=float)
        series.append(x)
        ruls.append(rul)
        regimes.append(reg_all[sel])
        healths.append(np.linspace(0.0, 1.0, T))       # proxy: linear wear
    return Fleet(series, ruls, regimes, healths, [False] * len(series),
                 n_channels=series[0].shape[1], n_regimes=n_regimes,
                 channel_groups=[list(range(series[0].shape[1]))])
