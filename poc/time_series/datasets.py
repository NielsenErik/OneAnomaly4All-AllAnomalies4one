"""
Unified dataset registry for the time-series pipeline.

One function, `load_fleets(spec)`, turns a config block into a `FleetPair`;
two more turn that pair into the AD and RUL tasks the models consume.  Every
source in `catalog.SOURCES` is interchangeable behind them:

    synthetic          the C-MAPSS-shaped simulator in `data.py`
    cmapss / ncmapss   real NASA turbofan (text, HDF5)
    phm08 battery      the rest of NASA's prognostics repository
    calce ims milling  (`data_prognostics.py`)
    pcoe
    esa opssat         annotated spacecraft telemetry (`data_space.py`)
    smapmsl

Everything downstream — circuits, baselines, attribution, metrics — sees the
same `ADTask` / `RULTask` objects regardless of source, so "does the result
survive real data?" is a one-line config change rather than a port.

Two protocol invariants hold for every source, and both are load-bearing:

  1. SPLITS ARE BY UNIT (or by TIME, never by window).  Overlapping windows
     from one engine are near-duplicates; a random window split would leak the
     test set into training and every number would be meaningless.  Where a
     benchmark defines an official held-out fleet (C-MAPSS test_*, N-C-MAPSS
     *_test) or a chronological split (ESA-ADB), that split is used; otherwise
     units are split here.
  2. THE ANOMALY PROTOCOL IS FIXED PER LABEL SOURCE.  Fleets without
     annotations use the injection protocol: train on healthy windows only,
     test on held-out healthy windows, injected anomalies with per-channel
     ground truth, and organic late-life windows.  Fleets WITH annotations
     (`Fleet.labels is not None`) use the operators' labels instead, via
     `make_ad_task_labeled` — same `ADTask`, real `y_test`, and on ESA-ADB a
     real `affected_test` too, because its annotations are per channel.
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
from .catalog import SOURCES, Source, get_source, supports
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
from .data_space import ANOMALY, CODE_NAMES, GAP, INVALID, NOMINAL, RARE_EVENT

INJECTED_KINDS = ("spike", "offset", "drift", "decouple", "desync")


# ═══════════════════════════════════════════════════════════════════════════
# Source resolution
# ═══════════════════════════════════════════════════════════════════════════

def dataset_id(spec: Dict[str, Any]) -> str:
    """Short slug naming the data source, used in run dirs and result rows."""
    try:
        src = get_source(spec)
    except KeyError:
        return str(spec.get("source", "?"))
    return src.ident(spec)


def dataset_available(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """(is the data on disk?, human-readable reason if not)."""
    try:
        src = get_source(spec)
    except KeyError as exc:
        return False, str(exc)
    try:
        ok = bool(src.probe(spec))
    except Exception as exc:                      # a malformed spec, not a crash
        return False, f"{src.name}: {exc}"
    return ok, "" if ok else (f"{src.title} not found in {src.root} "
                              "(see data/README.md)")


def load_fleets(spec: Dict[str, Any], seed: int = 0) -> FleetPair:
    """
    Config block -> FleetPair, through the catalogue.

    Censoring is applied to the FITTING fleet only, and only for run-to-failure
    sources: a fleet in SERVICE is what censoring models, and annotated
    telemetry has no failure to be censored before.  Sources with GENUINE
    censoring (batteries and bearings that survived their test) keep it —
    `censor_frac` adds simulated censoring on top and is recorded separately.
    """
    src = get_source(spec)
    pair = src.loader(spec, seed)
    pair.meta.setdefault("source_name", src.name)
    pair.meta.setdefault("source_kind", src.kind)
    pair.meta.setdefault("tasks_supported", list(src.tasks))
    # Caveats travel WITH the data: `prepare_task` prints them into every run
    # log, so a number cannot be produced without its known defects beside it.
    known = list(pair.meta.get("caveats", []))
    for c in src.caveats:
        if c not in known:
            known.append(c)
    if known:
        pair.meta["caveats"] = known

    censor = float(spec.get("censor_frac") or 0.0)
    if censor > 0:
        if pair.train.annotated:
            raise ValueError(
                f"censor_frac={censor} is meaningless for {src.name}: annotated "
                "telemetry has no time-to-failure to censor")
        pair.train = censor_fleet(pair.train, censor, seed=seed)
        pair.meta["censored_units"] = int(sum(pair.train.censored))
        pair.meta["simulated_censoring"] = censor
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
    Build the detection task.  Three paths, chosen by what the data actually
    provides — never by the dataset's name:

      annotated fleet        -> `make_ad_task_labeled`  (real labels)
      official test fleet    -> `make_ad_task_split`    (injected + organic)
      one fleet only         -> `make_ad_task`          (unchanged, so the
                                synthetic results stay bit-comparable)
    """
    window = int(spec.get("window", 8))
    stride = int(spec.get("stride", 2))
    if pair.train.annotated:
        task = make_ad_task_labeled(
            pair.train, pair.test, window=window, stride=stride,
            train_on_clean=bool(spec.get("train_on_clean", True)),
            rare_events=str(spec.get("rare_events", "anomaly")),
            gaps=str(spec.get("gaps", "drop")),
            per_regime=_per_regime_flag(spec, pair, default_auto=True),
            inject_rate=float(spec.get("labeled_inject_rate", 0.0)),
            strength=float(spec.get("strength", 1.0)),
            max_train_windows=spec.get("max_train_windows"),
            max_test_windows=spec.get("max_test_windows"),
            seed=seed)
        task.meta.update({"dataset": pair.name, **pair.meta})
        return task
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
# AD task from REAL annotations
# ═══════════════════════════════════════════════════════════════════════════

def make_ad_task_labeled(
    train_fleet: Fleet,
    test_fleet: Optional[Fleet],
    window: int = 8,
    stride: int = 2,
    train_on_clean: bool = True,
    rare_events: str = "anomaly",           # anomaly | normal | drop
    gaps: str = "drop",                     # drop | normal
    per_regime: bool = True,
    inject_rate: float = 0.0,
    strength: float = 1.0,
    max_train_windows: Optional[int] = None,
    max_test_windows: Optional[int] = None,
    seed: int = 0,
) -> ADTask:
    """
    Detection task from operator annotations — the same `ADTask` the injection
    protocol produces, with two differences that matter:

      `y_test` is REAL.  A window is anomalous if any of its samples carries an
      anomaly code, in any channel.  Nothing about the anomaly was chosen here.
      `affected_test` is real too WHERE the source annotates per channel
      (ESA-ADB does; OPSSAT-AD and SMAP/MSL label whole segments, so their
      affected lists come out empty and the localisation metrics correctly
      report "no ground truth" rather than a fabricated one).

    Three policy knobs, all of which change the numbers and are therefore
    recorded in `meta`:

      `train_on_clean`  drop annotated windows from the fitting set.  ESA-ADB
            is deliberately NOT clean-train — anomalies occur in the first half
            of the mission too — so the honest options are to train on
            everything (unsupervised, contamination included) or to use the
            annotations to clean the training half (semi-supervised, which is
            what ESA-ADB's own baselines do).  Default: clean.
      `rare_events`  rare NOMINAL events are atypical but expected (commanded
            manoeuvres, resets, calibrations).  ESA-ADB's baselines score them
            as anomalies, so that is the default; `normal` treats them as
            nominal and `drop` removes those windows from the test set.
      `gaps`  communication gaps and invalid fragments are missing data, not
            faults.  Dropped by default; scoring them as anomalies would
            measure gap detection.

    Injection can be layered on top (`inject_rate` > 0) so that the same
    circuit is scored against real AND synthetic faults in one run — the direct
    check on whether the injected protocol overstates detectability.
    """
    if test_fleet is None:
        raise ValueError("an annotated source must provide a held-out fleet; "
                         "its split is chronological and cannot be re-drawn here")
    rng = np.random.default_rng(seed)
    std = Standardizer(per_regime=per_regime).fit(train_fleet, range(len(train_fleet)))

    positive = {ANOMALY}
    if rare_events == "anomaly":
        positive.add(RARE_EVENT)
    droppable = {GAP, INVALID} if gaps == "drop" else set()
    if rare_events == "drop":
        droppable = droppable | {RARE_EVENT}

    def windows_and_codes(fleet: Fleet, u: int):
        x = std.transform(fleet.series[u], fleet.regime[u])
        W, right = windowize(x, window, stride)
        if not len(W):
            return None, None, None
        L = (fleet.labels or [])[u]
        # (n_windows, window, C) view of the codes, aligned with W's rows
        starts = right - window + 1
        codes = np.stack([L[s:s + window] for s in starts])
        return W, codes, right

    # ── fitting set ──────────────────────────────────────────────────────
    X_train = []
    n_contaminated = 0
    for u in range(len(train_fleet)):
        W, codes, _ = windows_and_codes(train_fleet, u)
        if W is None:
            continue
        dirty = (codes > NOMINAL).any(axis=(1, 2))
        n_contaminated += int(dirty.sum())
        X_train.append(W[~dirty] if train_on_clean else W)
    X_train = [w for w in X_train if len(w)]
    if not X_train:
        raise ValueError("no training windows survived: the whole fitting half "
                         "is annotated, or `window` exceeds every segment")
    X_train = np.concatenate(X_train, axis=0)
    if max_train_windows and len(X_train) > max_train_windows:
        step = int(np.ceil(len(X_train) / max_train_windows))
        X_train = X_train[::step]                 # stride, never shuffle

    # ── test set ─────────────────────────────────────────────────────────
    donor_pool = X_train
    X_test, y_test, kinds, affected = [], [], [], []
    for u in range(len(test_fleet)):
        W, codes, _ = windows_and_codes(test_fleet, u)
        if W is None:
            continue
        for i in range(len(W)):
            c = codes[i]                          # (window, C)
            present = set(np.unique(c).tolist()) - {NOMINAL}
            if present & droppable:
                continue                          # missing data, not a fault
            hits = present & positive
            if hits:
                chans = sorted(int(j) for j in range(c.shape[1])
                               if bool(np.isin(c[:, j], list(positive)).any()))
                X_test.append(W[i]); y_test.append(1)
                kinds.append(CODE_NAMES[ANOMALY] if ANOMALY in hits
                             else CODE_NAMES[RARE_EVENT])
                affected.append(chans)
            elif present:                         # e.g. rare_events="normal"
                X_test.append(W[i]); y_test.append(0)
                kinds.append("normal"); affected.append([])
            elif inject_rate > 0 and rng.random() < inject_rate:
                seg = W[i].reshape(window, -1)
                kind = str(rng.choice(list(INJECTED_KINDS)))
                donor = donor_pool[rng.integers(len(donor_pool))]
                seg, ch = _inject(seg, kind, rng, strength, donor=donor)
                X_test.append(seg.reshape(-1)); y_test.append(1)
                kinds.append(kind); affected.append(ch)
            else:
                X_test.append(W[i]); y_test.append(0)
                kinds.append("normal"); affected.append([])

    if not X_test:
        raise ValueError("no test windows — check `window` against segment lengths")
    if max_test_windows and len(X_test) > max_test_windows:
        # Keep every positive and stride the negatives: the alternative
        # (striding everything) throws away most of a 1%-density benchmark's
        # anomalies and leaves an AUROC computed on a handful of positives.
        pos = [i for i, v in enumerate(y_test) if v == 1]
        neg = [i for i, v in enumerate(y_test) if v == 0]
        room = max(max_test_windows - len(pos), 1)
        if len(neg) > room:
            neg = neg[:: int(np.ceil(len(neg) / room))]
        sel = sorted(pos + neg)
        X_test = [X_test[i] for i in sel]; y_test = [y_test[i] for i in sel]
        kinds = [kinds[i] for i in sel]; affected = [affected[i] for i in sel]

    n_pos = int(sum(y_test))
    if n_pos == 0:
        raise ValueError(
            "the held-out half contains no annotated anomaly under this policy "
            f"(rare_events={rare_events!r}, gaps={gaps!r}) — AUROC would be "
            "undefined")

    return ADTask(
        X_train=torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        X_test=torch.from_numpy(np.asarray(X_test, dtype=np.float32)),
        y_test=torch.tensor(y_test, dtype=torch.long),
        kind_test=kinds,
        affected_test=affected,
        window=window,
        n_channels=train_fleet.n_channels,
        channel_groups=train_fleet.channel_groups or [list(range(train_fleet.n_channels))],
        meta={"train_segments": len(train_fleet), "test_segments": len(test_fleet),
              "n_regimes": train_fleet.n_regimes, "seed": seed,
              "per_regime_norm": per_regime, "labels_are_real": True,
              "train_on_clean": train_on_clean,
              "contaminated_train_windows_dropped": n_contaminated if train_on_clean else 0,
              "rare_events": rare_events, "gaps": gaps,
              "injected_on_top": inject_rate > 0,
              "n_positive": n_pos,
              # Localisation truth exists only when the annotation SELECTS
              # channels.  A univariate source (OPSSAT, SMAP/MSL with
              # dims='first') always "affects" its one channel, and a label
              # naming every channel localises nothing — both are vacuous, and
              # reporting them as ground truth would make the attribution
              # metrics score a constant answer as perfect.
              "localisation_truth": bool(
                  any(0 < len(a) < train_fleet.n_channels for a in affected))},
    )


# ═══════════════════════════════════════════════════════════════════════════
# RUL task
# ═══════════════════════════════════════════════════════════════════════════

def build_rul_task(pair: FleetPair, spec: Dict[str, Any], seed: int = 0) -> RULTask:
    if pair.train.annotated:
        raise ValueError(
            f"{pair.name} is annotated telemetry: it has no remaining useful "
            "life, so the `rul` stage cannot run on it.  Drop `rul` from "
            "`stages` for this source (catalog.Source.tasks lists what each "
            "source supports).")
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
