"""
Unified dataset registry for the time-series pipeline.

One function, `load_fleets(spec)`, turns a config block into a `FleetPair`;
two more turn that pair into the AD and RUL tasks the models consume.  Three
sources are interchangeable behind it:

    synthetic          the C-MAPSS-shaped simulator in `data.py`
    cmapss:FD001..4    real NASA C-MAPSS
    ncmapss:DS01..8    real NASA N-C-MAPSS (flight conditions, HDF5)

Everything downstream — circuits, baselines, attribution, metrics — sees the
same `ADTask` / `RULTask` objects regardless of source, so "does the result
survive real data?" is a one-line config change rather than a port.

Two protocol invariants hold for every source, and both are load-bearing:

  1. SPLITS ARE BY UNIT.  Overlapping windows from one engine are near-
     duplicates; a random window split would leak the test set into training
     and every number would be meaningless.  Where a benchmark defines an
     official held-out fleet (C-MAPSS test_*, N-C-MAPSS *_test) that fleet is
     used; otherwise units are split here.
  2. THE ANOMALY PROTOCOL IS IDENTICAL.  Train on healthy windows only; test on
     held-out healthy windows, injected anomalies with per-channel ground
     truth, and organic late-life windows.  Only the background process
     changes between synthetic and real, which is exactly what makes the
     comparison informative.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import (
    ADTask,
    Fleet,
    RULTask,
    Standardizer,
    _inject,
    make_ad_task,
    make_rul_task,
    simulate_fleet,
    windowize,
)
from .data_real import (
    FleetPair,
    censor_fleet,
    cmapss_available,
    correlation_groups,
    load_cmapss,
    load_ncmapss,
    ncmapss_available,
    subsample_units,
)

INJECTED_KINDS = ("spike", "offset", "drift", "decouple", "desync")


# ═══════════════════════════════════════════════════════════════════════════
# Source resolution
# ═══════════════════════════════════════════════════════════════════════════

def dataset_id(spec: Dict[str, Any]) -> str:
    """Short slug naming the data source, used in run dirs and result rows."""
    src = str(spec.get("source", "synthetic")).lower()
    if src == "cmapss":
        return f"cmapss:{str(spec.get('subset', 'FD001')).upper()}"
    if src == "ncmapss":
        return f"ncmapss:{str(spec.get('dataset', 'DS02')).upper()}"
    return "synthetic"


def dataset_available(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """(is the data on disk?, human-readable reason if not)."""
    src = str(spec.get("source", "synthetic")).lower()
    if src == "synthetic":
        return True, ""
    if src == "cmapss":
        sub = str(spec.get("subset", "FD001")).upper()
        ok = cmapss_available(sub, spec.get("data_dir"))
        return ok, "" if ok else f"C-MAPSS {sub} files not found (see data/README.md)"
    if src == "ncmapss":
        ds = str(spec.get("dataset", "DS02")).upper()
        ok = ncmapss_available(ds, spec.get("data_dir"))
        return ok, "" if ok else f"N-C-MAPSS {ds}.h5 not found (see data/README.md)"
    return False, f"unknown dataset source {src!r}"


def load_fleets(spec: Dict[str, Any], seed: int = 0) -> FleetPair:
    """Config block -> FleetPair.  Censoring is applied to the FITTING fleet only."""
    src = str(spec.get("source", "synthetic")).lower()
    # `cap: null` means "use this source's default" — the key exists in the
    # schema, so a plain .get() default would never fire.
    cap = spec.get("cap")
    cap = float(cap) if cap is not None else (130.0 if src == "synthetic" else 125.0)
    censor = float(spec.get("censor_frac") or 0.0)

    if src == "synthetic":
        fleet = simulate_fleet(
            n_units=int(spec.get("units", 60)),
            n_channels=int(spec.get("channels", 14)),
            n_regimes=int(spec.get("regimes", 3)),
            n_groups=int(spec.get("groups", 3)),
            min_life=int(spec.get("min_life", 120)),
            max_life=int(spec.get("max_life", 300)),
            censor_frac=censor,
            dead_channels=int(spec.get("dead_channels", 2)),
            noise=float(spec.get("noise", 0.10)),
            group_noise=float(spec.get("group_noise", 0.25)),
            phi_ar=float(spec.get("phi_ar", 0.85)),
            seed=seed,
        )
        return FleetPair(name="synthetic", train=fleet, test=None, cap=cap,
                         meta={"source": "synthetic simulator", "seed": seed,
                               "units": len(fleet)})

    if src == "cmapss":
        pair = load_cmapss(
            subset=str(spec.get("subset", "FD001")),
            data_dir=spec.get("data_dir"),
            cap=cap,
            n_regimes=spec.get("regimes"),
            n_groups=int(spec.get("groups", 3)),
            max_units=spec.get("max_units"),
            with_test=bool(spec.get("official_test", True)),
            seed=seed,
        )
    elif src == "ncmapss":
        pair = load_ncmapss(
            dataset=str(spec.get("dataset", "DS02")),
            data_dir=spec.get("data_dir"),
            channels=tuple(spec.get("channels_groups", ("X_s",))),
            aggregate=str(spec.get("aggregate", "cycle")),
            subsample=int(spec.get("subsample", 10)),
            cap=cap,
            n_regimes=int(spec.get("regimes", 3)),
            n_groups=int(spec.get("groups", 3)),
            max_units=spec.get("max_units"),
            max_rows=spec.get("max_rows"),
            use_cache=bool(spec.get("cache", True)),
            seed=seed,
        )
    else:
        raise KeyError(f"unknown dataset source {src!r} "
                       "(expected synthetic | cmapss | ncmapss)")

    if censor > 0:
        # Real training fleets are run-to-failure; censoring is what a fleet in
        # SERVICE looks like, so it is simulated on real trajectories.
        pair.train = censor_fleet(pair.train, censor, seed=seed)
        pair.meta["censored_units"] = int(sum(pair.train.censored))
    return pair


def _per_regime_flag(spec: Dict[str, Any], pair: FleetPair, default_auto: bool) -> bool:
    v = spec.get("per_regime", "auto")
    if isinstance(v, bool):
        return v
    return default_auto and pair.train.n_regimes > 1


# ═══════════════════════════════════════════════════════════════════════════
# AD task
# ═══════════════════════════════════════════════════════════════════════════

def build_ad_task(pair: FleetPair, spec: Dict[str, Any], seed: int = 0) -> ADTask:
    """
    Build the detection task.  Delegates to `make_ad_task` (unchanged, so the
    synthetic results stay bit-comparable) when there is no official test
    fleet, and to `make_ad_task_split` when there is.
    """
    window = int(spec.get("window", 8))
    stride = int(spec.get("stride", 2))
    kw = dict(
        window=window, stride=stride,
        healthy_frac=float(spec.get("healthy_frac", 0.35)),
        organic_frac=float(spec.get("organic_frac", 0.85)),
        inject_rate=float(spec.get("inject_rate", 0.12)),
        strength=float(spec.get("strength", 1.0)),
        seed=seed,
    )
    if pair.test is None:
        task = make_ad_task(fleet=pair.train,
                            train_units=float(spec.get("train_units", 0.6)), **kw)
    else:
        task = make_ad_task_split(
            pair.train, pair.test,
            per_regime=_per_regime_flag(spec, pair, default_auto=True),
            max_test_windows=spec.get("max_test_windows"), **kw)
    task.meta.update({"dataset": pair.name, "cap": pair.cap, **pair.meta})
    return task


def make_ad_task_split(
    train_fleet: Fleet,
    test_fleet: Fleet,
    window: int = 8,
    stride: int = 2,
    healthy_frac: float = 0.35,
    organic_frac: float = 0.85,
    inject_rate: float = 0.12,
    strength: float = 1.0,
    per_regime: bool = False,
    max_test_windows: Optional[int] = None,
    seed: int = 0,
) -> ADTask:
    """
    Same protocol as `make_ad_task`, but with the two fleets given rather than
    carved out of one — the case where a benchmark ships an official held-out
    set of units (C-MAPSS test_*, N-C-MAPSS *_test).

    Training uses healthy windows of the training fleet ONLY.  The test set is
    every window of the held-out fleet that is unambiguously healthy or
    unambiguously degraded; mid-life windows are dropped, because inventing a
    label for them is the labelling flaw Wu & Keogh (TKDE 2021) fault these
    benchmarks for.  A fraction `inject_rate` of the healthy test windows
    receives one synthetic anomaly with recorded ground-truth channels — the
    only source of localisation truth available on real data.
    """
    rng = np.random.default_rng(seed)
    std = Standardizer(per_regime=per_regime).fit(train_fleet, range(len(train_fleet)))

    X_train = []
    for u in range(len(train_fleet)):
        x = std.transform(train_fleet.series[u], train_fleet.regime[u])
        W, right = windowize(x, window, stride)
        if not len(W):
            continue
        h = train_fleet.health[u][right]
        W = W[h < healthy_frac]
        if len(W):
            X_train.append(W)
    if not X_train:
        raise ValueError(
            "no healthy training windows: healthy_frac is too strict for this "
            "fleet, or `window` exceeds the shortest trajectory")
    X_train = np.concatenate(X_train, axis=0)

    donor_pool = X_train
    X_test, y_test, kinds, affected = [], [], [], []
    for u in range(len(test_fleet)):
        x = std.transform(test_fleet.series[u], test_fleet.regime[u])
        W, right = windowize(x, window, stride)
        if not len(W):
            continue
        h = test_fleet.health[u][right]
        for i in range(len(W)):
            if h[i] > organic_frac:
                X_test.append(W[i]); y_test.append(1)
                kinds.append("organic"); affected.append([])
            elif h[i] < healthy_frac:
                if rng.random() < inject_rate:
                    seg = W[i].reshape(window, -1)
                    kind = str(rng.choice(list(INJECTED_KINDS)))
                    donor = donor_pool[rng.integers(len(donor_pool))]
                    seg, chans = _inject(seg, kind, rng, strength, donor=donor)
                    X_test.append(seg.reshape(-1)); y_test.append(1)
                    kinds.append(kind); affected.append(chans)
                else:
                    X_test.append(W[i]); y_test.append(0)
                    kinds.append("normal"); affected.append([])

    if max_test_windows and len(X_test) > max_test_windows:
        sel = rng.permutation(len(X_test))[:max_test_windows]
        sel = sorted(int(i) for i in sel)
        X_test = [X_test[i] for i in sel]; y_test = [y_test[i] for i in sel]
        kinds = [kinds[i] for i in sel]; affected = [affected[i] for i in sel]

    return ADTask(
        X_train=torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        X_test=torch.from_numpy(np.asarray(X_test, dtype=np.float32)),
        y_test=torch.tensor(y_test, dtype=torch.long),
        kind_test=kinds,
        affected_test=affected,
        window=window,
        n_channels=train_fleet.n_channels,
        channel_groups=train_fleet.channel_groups or [list(range(train_fleet.n_channels))],
        meta={"train_units": len(train_fleet), "test_units": len(test_fleet),
              "n_regimes": train_fleet.n_regimes, "seed": seed,
              "per_regime_norm": per_regime, "official_test_fleet": True},
    )


# ═══════════════════════════════════════════════════════════════════════════
# RUL task
# ═══════════════════════════════════════════════════════════════════════════

def build_rul_task(pair: FleetPair, spec: Dict[str, Any], seed: int = 0) -> RULTask:
    window = int(spec.get("window", 8))
    stride = int(spec.get("stride", 3))
    bins = int(spec.get("bins", 20))
    if pair.test is None:
        task = make_rul_task(window=window, stride=stride, n_bins=bins,
                             cap=pair.cap, train_units=float(spec.get("train_units", 0.65)),
                             seed=seed, fleet=pair.train)
    else:
        task = make_rul_task_split(
            pair.train, pair.test, window=window, stride=stride, n_bins=bins,
            cap=pair.cap,
            per_regime=_per_regime_flag(spec, pair, default_auto=True),
            test_windows=str(spec.get("rul_test_windows", "all")),
            seed=seed)
    task.meta.update({"dataset": pair.name, **pair.meta})
    return task


def make_rul_task_split(
    train_fleet: Fleet,
    test_fleet: Fleet,
    window: int = 8,
    stride: int = 3,
    n_bins: int = 20,
    cap: float = 125.0,
    per_regime: bool = True,
    test_windows: str = "all",
    seed: int = 0,
) -> RULTask:
    """
    RUL task from an explicit train/test fleet pair.

    `test_windows="last"` keeps only the final window of each held-out unit —
    the standard C-MAPSS protocol, one prediction per engine, directly
    comparable to the published RMSE/NASA-score tables.  `"all"` scores every
    window and gives far more test points (better statistics, not comparable to
    the literature).  Both are reported honestly; neither is a substitute for
    the other.

    Censored TRAINING units contribute τ ≥ (last observed bin); the test fleet
    is always fully labelled.
    """
    std = Standardizer(per_regime=per_regime).fit(train_fleet, range(len(train_fleet)))
    edges = np.linspace(0, cap, n_bins + 1)

    def to_bin(rul: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(np.minimum(rul, cap), edges[1:-1]), 0, n_bins - 1)

    def build(fleet: Fleet, last_only: bool):
        Xs, taus, deltas, regs, raw, uids = [], [], [], [], [], []
        for u in range(len(fleet)):
            x = std.transform(fleet.series[u], fleet.regime[u])
            W, right = windowize(x, window, stride)
            if not len(W):
                continue
            rul = fleet.rul[u][right]
            if last_only:
                W, right, rul = W[-1:], right[-1:], rul[-1:]
            Xs.append(W)
            taus.append(to_bin(rul))
            raw.append(np.minimum(rul, cap))
            deltas.append(np.full(len(W), 0 if fleet.censored[u] else 1))
            regs.append(fleet.regime[u][right])
            uids.append(np.full(len(W), int(u)))
        if not Xs:
            raise ValueError("no RUL windows — window longer than every trajectory?")
        cat = lambda a: np.concatenate(a, axis=0)
        return cat(Xs), cat(taus), cat(deltas), cat(regs), cat(raw), cat(uids)

    Xtr, ttr, dtr, rtr, rawtr, utr = build(train_fleet, last_only=False)
    Xte, tte, _, rte, rawte, ute = build(test_fleet,
                                        last_only=(test_windows == "last"))

    return RULTask(
        X_train=torch.from_numpy(Xtr.astype(np.float32)),
        tau_train=torch.from_numpy(ttr.astype(np.int64)),
        delta_train=torch.from_numpy(dtr.astype(np.int64)),
        regime_train=torch.from_numpy(rtr.astype(np.int64)),
        X_test=torch.from_numpy(Xte.astype(np.float32)),
        tau_test=torch.from_numpy(tte.astype(np.int64)),
        rul_test=torch.from_numpy(rawte.astype(np.float32)),
        regime_test=torch.from_numpy(rte.astype(np.int64)),
        window=window, n_channels=train_fleet.n_channels, n_bins=n_bins, cap=cap,
        channel_groups=train_fleet.channel_groups or [list(range(train_fleet.n_channels))],
        meta={"train_units": len(train_fleet), "test_units": len(test_fleet),
              "censored_units": int(sum(train_fleet.censored)),
              "n_regimes": train_fleet.n_regimes, "seed": seed,
              "per_regime_norm": per_regime, "test_windows": test_windows,
              "official_test_fleet": True},
        unit_train=torch.from_numpy(utr.astype(np.int64)),
        unit_test=torch.from_numpy(ute.astype(np.int64)),
        rul_train=torch.from_numpy(rawtr.astype(np.float32)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Description helper (goes into every run's log)
# ═══════════════════════════════════════════════════════════════════════════

def describe_task(task) -> Dict[str, Any]:
    if isinstance(task, ADTask):
        kinds = {}
        for k in task.kind_test:
            kinds[k] = kinds.get(k, 0) + 1
        return {"kind": "ad", "n_train": int(len(task.X_train)),
                "n_test": int(len(task.X_test)), "d": int(task.X_train.shape[1]),
                "window": task.window, "n_channels": task.n_channels,
                "anomaly_rate": float(task.y_test.float().mean()),
                "kind_counts": kinds, **{k: v for k, v in task.meta.items()
                                         if not isinstance(v, (list, dict))}}
    return {"kind": "rul", "n_train": int(len(task.X_train)),
            "n_test": int(len(task.X_test)), "d": int(task.X_train.shape[1]),
            "window": task.window, "n_channels": task.n_channels,
            "n_bins": task.n_bins, "cap": float(task.cap),
            "censored_frac": float(1.0 - task.delta_train.float().mean()),
            **{k: v for k, v in task.meta.items() if not isinstance(v, (list, dict))}}
