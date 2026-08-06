"""
The rest of NASA's prognostics repository (PCoE), behind the same `FleetPair`
interface as C-MAPSS.

C-MAPSS and N-C-MAPSS answer "does the method work on a turbofan?".  These
answer "does it work on a MACHINE?" — degradation physics that is not a gas
path: lithium-ion capacity fade, rolling-element bearing spalling, tool flank
wear, and (through a generic adapter) whatever else in PCoE ships as per-unit
tables.

    phm08     PHM08 Challenge — C-MAPSS-shaped, true test RUL withheld
    battery   NASA Li-ion aging (Saha & Goebel), .mat, EOL = 30% capacity fade
    calce     CALCE CS2 prismatic cells, the usual companion to the above
    ims       IMS/Rexnord bearing run-to-failure, raw 20 kHz vibration
    milling   NASA milling (mill.mat), flank wear VB measured per run
    pcoe      generic per-unit CSV adapter for the remaining PCoE sets
              (IGBT, capacitor, fatigue/Lamb-wave) whose layouts vary by release

What is real here, stated once
──────────────────────────────
The sensor process, the degradation, and the end-of-life labels are real for
every source in this file.  Anomalies for the DETECTION task are still injected
(none of these sets ships anomaly annotations) — for real annotations see
`data_space.py`.  Right censoring, unlike on C-MAPSS, is often GENUINE here:
batteries and bearings that never reached the end-of-life criterion within the
test are marked censored rather than truncated by us, which is the one place in
the project where the censoring term is exercised on real censoring.

Every loader turns a raw record into a per-cycle / per-file FEATURE series, and
which features is a modelling decision, not a detail.  The choices are stated
in each function's docstring and recorded in `pair.meta["features"]`, because a
result on engineered features is a result about those features too.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Fleet
from .data_real import (
    CMAPSS_COLUMNS,
    DATA_ROOT,
    FleetPair,
    _assign_regimes,
    _fit_regimes,
    correlation_groups,
    health_from_rul,
    subsample_units,
)

PHM08_DIR = os.path.join(DATA_ROOT, "phm08")
BATTERY_DIR = os.path.join(DATA_ROOT, "battery_nasa")
CALCE_DIR = os.path.join(DATA_ROOT, "calce")
IMS_DIR = os.path.join(DATA_ROOT, "ims")
MILLING_DIR = os.path.join(DATA_ROOT, "milling")
PCOE_DIR = os.path.join(DATA_ROOT, "pcoe")
CACHE_DIR = os.path.join(DATA_ROOT, "cache")


# ═══════════════════════════════════════════════════════════════════════════
# Shared construction
# ═══════════════════════════════════════════════════════════════════════════

def _fleet_from_units(
    series: List[np.ndarray],
    rul: List[np.ndarray],
    cap: float,
    censored: Optional[List[bool]] = None,
    regimes: Optional[List[np.ndarray]] = None,
    n_regimes: int = 1,
    n_groups: int = 3,
) -> Fleet:
    """Assemble a run-to-failure Fleet, deriving health from RUL under the
    piecewise-linear convention used everywhere else in the pipeline."""
    if not series:
        raise ValueError("no units survived loading — check the filters")
    C = series[0].shape[1]
    regimes = regimes or [np.zeros(len(s), dtype=int) for s in series]
    health = [health_from_rul(r, cap) for r in rul]
    groups = correlation_groups(np.concatenate(series[: min(len(series), 20)]), n_groups)
    return Fleet(series, [np.asarray(r, dtype=float) for r in rul], regimes,
                 health, list(censored or [False] * len(series)),
                 n_channels=C, n_regimes=max(int(n_regimes), 1),
                 channel_groups=groups)


def _rul_to_eol(values: np.ndarray, threshold: float, decreasing: bool = True
                ) -> Tuple[np.ndarray, bool, int]:
    """
    (rul, censored, eol_index) for a health indicator crossing a threshold.

    The unit is followed until the indicator first crosses `threshold` (that
    index is end-of-life, RUL 0) and truncated there.  A unit whose indicator
    never crosses is RIGHT-CENSORED: its RUL is only known to exceed the
    remaining observation, which is exactly the information the censored
    likelihood term uses and a plain regressor cannot.
    """
    v = np.asarray(values, dtype=float)
    hit = np.where(v <= threshold if decreasing else v >= threshold)[0]
    if len(hit):
        eol = int(hit[0])
        return np.arange(eol, -1, -1, dtype=float), False, eol
    n = len(v)
    return np.arange(n - 1, -1, -1, dtype=float), True, n - 1


def _time_features(x: np.ndarray, n_bands: int = 4) -> np.ndarray:
    """
    Per-window vibration features: the standard rotating-machinery set.

    rms and peak track energy; crest factor and kurtosis are the classic
    early-spall indicators (impulsiveness rises long before energy does);
    band energies split the spectrum coarsely so a defect frequency moving into
    a band is visible without knowing the shaft speed.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-12
    peak = float(np.max(np.abs(x)))
    sd = float(x.std()) or 1e-12
    feats = [rms, peak, peak / rms,
             float(np.mean((x / sd) ** 4)), float(np.mean((x / sd) ** 3)),
             float(np.mean(np.abs(np.diff(x))))]
    spec = np.abs(np.fft.rfft(x)) ** 2
    for band in np.array_split(spec[1:], n_bands):
        feats.append(float(np.log1p(band.sum())))
    return np.asarray(feats, dtype=np.float32)


TIME_FEATURE_NAMES = (["rms", "peak", "crest", "kurtosis", "skew", "mean_abs_diff"]
                      + [f"band{i}" for i in range(4)])


def _cache(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".npz")


def _save_fleet_cache(path: str, pair: FleetPair) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = pair.train
    np.savez_compressed(
        path,
        X=np.concatenate(f.series, 0).astype(np.float32),
        rul=np.concatenate(f.rul).astype(np.float32),
        reg=np.concatenate(f.regime).astype(np.int32),
        lens=np.array([len(s) for s in f.series], dtype=np.int64),
        cens=np.array(f.censored, dtype=bool),
        nreg=np.array([f.n_regimes]),
    )


def _load_fleet_cache(path: str, cap: float, n_groups: int) -> Fleet:
    z = np.load(path, allow_pickle=False)
    lens = z["lens"]
    b = np.concatenate([[0], np.cumsum(lens)])
    cut = lambda a: [a[b[i]:b[i + 1]] for i in range(len(lens))]
    return _fleet_from_units(cut(z["X"]), cut(z["rul"]), cap=cap,
                             censored=list(z["cens"]), regimes=cut(z["reg"]),
                             n_regimes=int(z["nreg"][0]), n_groups=n_groups)


# ═══════════════════════════════════════════════════════════════════════════
# PHM08 Challenge
# ═══════════════════════════════════════════════════════════════════════════

PHM08_FILES = ("train.txt", "test.txt", "final_test.txt")


def phm08_available(data_dir: Optional[str] = None) -> bool:
    d = data_dir or PHM08_DIR
    return any(os.path.exists(os.path.join(d, f)) for f in ("train.txt", "PHM08_train.txt"))


def load_phm08(
    data_dir: Optional[str] = None,
    cap: float = 125.0,
    n_regimes: int = 6,
    n_groups: int = 3,
    max_units: Optional[int] = None,
    seed: int = 0,
) -> FleetPair:
    """
    PHM08 Challenge data — the same 26-column layout as C-MAPSS
    (`unit cycle op1 op2 op3 s1..s21`), six operating conditions, run to
    failure in `train.txt`.

    The official `test.txt` / `final_test.txt` units are truncated and their
    true RUL was never released (scores came from the challenge server), so
    they cannot be scored here.  Only `train.txt` becomes a fleet, and the
    train/test split is then by unit, exactly as for the synthetic fleet.
    Treating the challenge test files as labelled data is a silent error worth
    naming: nothing here reads them.
    """
    d = data_dir or PHM08_DIR
    path = next((os.path.join(d, f) for f in ("train.txt", "PHM08_train.txt")
                 if os.path.exists(os.path.join(d, f))), None)
    if path is None:
        raise FileNotFoundError(
            f"{d}/train.txt not found.  Get the PHM08 Prognostics Data "
            "Challenge set (PHM Society data repository / NASA PCoE mirror) "
            "and place train.txt there.  See data/README.md.")

    raw = np.loadtxt(path)
    ops, sens = raw[:, 2:5], raw[:, 5:]
    keep = sens.std(0) > 1e-8
    names = [CMAPSS_COLUMNS[5 + i] for i in range(sens.shape[1]) if keep[i]]
    sens = sens[:, keep]
    reg, _ = _fit_regimes(ops, n_regimes, seed=seed)

    units = raw[:, 0].astype(int)
    series, ruls, regimes = [], [], []
    for u in np.unique(units):
        sel = units == u
        T = int(sel.sum())
        series.append(sens[sel].astype(np.float32))
        ruls.append(np.arange(T - 1, -1, -1, dtype=float))
        regimes.append(reg[sel].astype(int))

    fleet = _fleet_from_units(series, ruls, cap=cap, regimes=regimes,
                              n_regimes=int(reg.max()) + 1, n_groups=n_groups)
    fleet = subsample_units(fleet, max_units, seed)
    return FleetPair(name="phm08", train=fleet, test=None, cap=cap,
                     channel_names=names,
                     meta={"source": "NASA/PHM Society PHM08 (real)",
                           "units": len(fleet), "n_regimes": fleet.n_regimes,
                           "dropped_constant_sensors": int((~keep).sum()),
                           "official_test_labels": "withheld by the challenge",
                           "caveats": ["test.txt / final_test.txt have no public "
                                       "RUL labels and are not loaded"]})


# ═══════════════════════════════════════════════════════════════════════════
# NASA Li-ion battery aging (Saha & Goebel)
# ═══════════════════════════════════════════════════════════════════════════

BATTERY_FEATURES = ("capacity_ah", "v_mean", "v_min", "temp_mean", "temp_max",
                    "current_mean", "duration_s", "energy_wh")


def battery_available(data_dir: Optional[str] = None) -> bool:
    d = data_dir or BATTERY_DIR
    return bool(glob.glob(os.path.join(d, "*.mat")))


def _mat_cycles(path: str, name: Optional[str] = None):
    """Yield (type, ambient_temperature, data) per cycle from a PCoE battery .mat."""
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    key = name if name in keys else keys[0]
    obj = mat[key]
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        obj = obj[0, 0] if obj.ndim == 2 else obj[0]
    cycles = getattr(obj, "cycle", None)
    if cycles is None:
        raise KeyError(f"{path}: no `cycle` field under {key!r} — not a PCoE "
                       "battery file?")
    for c in np.ravel(cycles):
        yield (str(np.ravel(c.type)[0]),
               float(np.ravel(c.ambient_temperature)[0]) if hasattr(c, "ambient_temperature") else np.nan,
               c.data)


def _battery_cycle_row(data) -> Optional[np.ndarray]:
    """One discharge cycle -> the BATTERY_FEATURES row (None if unusable)."""
    def field(name):
        try:
            v = np.ravel(getattr(np.ravel(data)[0], name)).astype(float)
        except Exception:
            return None
        return v if v.size else None

    cap = field("Capacity")
    v = field("Voltage_measured")
    i = field("Current_measured")
    t = field("Temperature_measured")
    tt = field("Time")
    if cap is None or v is None or tt is None:
        return None
    dur = float(tt[-1] - tt[0])
    energy = float(np.trapezoid(np.abs(v * (i if i is not None else 1.0)), tt) / 3600.0)
    return np.asarray([
        float(cap[0]), float(v.mean()), float(v.min()),
        float(t.mean()) if t is not None else np.nan,
        float(t.max()) if t is not None else np.nan,
        float(i.mean()) if i is not None else np.nan,
        dur, energy], dtype=np.float32)


def load_nasa_battery(
    data_dir: Optional[str] = None,
    cells: Optional[Sequence[str]] = None,
    eol_frac: float = 0.7,
    rated_ah: Optional[float] = None,
    cap: float = 60.0,
    n_groups: int = 2,
    min_cycles: int = 20,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    NASA PCoE Li-ion aging as a fleet of cells, one row per DISCHARGE cycle.

    Each `B00xx.mat` holds a `cycle` struct array of charge / discharge /
    impedance operations; only discharges carry `Capacity`, so the degradation
    trajectory is the discharge sequence and the features are per-discharge
    summaries (`BATTERY_FEATURES`).  Charge and impedance cycles are skipped —
    including them would put three different physical operations in one series
    and make "the next timestep" meaningless.

    End of life is the dataset's own criterion: capacity fallen to `eol_frac`
    of rated (0.7 → the 30% fade that stopped the experiments, 2 Ah → 1.4 Ah).
    Cells that never reach it inside their record are RIGHT-CENSORED — real
    censoring, not simulated.

    Ambient temperature varies between cells (some run at 42 °C) and load
    profiles differ (the randomised-use cells use a 0.5 Hz 4 A square wave);
    that variation is the fleet's heterogeneity, and it is left in.
    """
    d = data_dir or BATTERY_DIR
    files = sorted(glob.glob(os.path.join(d, "*.mat")))
    if cells:
        want = {str(c).upper() for c in cells}
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0].upper() in want]
    if not files:
        raise FileNotFoundError(
            f"no .mat files in {d}.  Get the NASA PCoE 'Battery Data Set' "
            "(Saha & Goebel) from data.phmsociety.org/nasa or data.nasa.gov "
            "and place B0005.mat etc. there.  See data/README.md.")

    key = f"battery_{len(files)}_{eol_frac}_{rated_ah or 0}_{min_cycles}"
    path_cache = _cache(key)
    meta = {"source": "NASA PCoE Li-ion battery (real)",
            "features": list(BATTERY_FEATURES), "eol_frac": eol_frac,
            "eol_rule": f"capacity <= {eol_frac:g} x rated",
            "genuine_censoring": True}
    if use_cache and os.path.exists(path_cache):
        fleet = _load_fleet_cache(path_cache, cap, n_groups)
        return FleetPair(name="battery", train=fleet, test=None, cap=cap,
                         channel_names=list(BATTERY_FEATURES),
                         meta={**meta, "cache": path_cache, "units": len(fleet)})

    series, ruls, censored, names = [], [], [], []
    for f in files:
        cell = os.path.splitext(os.path.basename(f))[0]
        rows, temps = [], []
        for ctype, amb, data in _mat_cycles(f, cell):
            if not ctype.lower().startswith("discharge"):
                continue
            row = _battery_cycle_row(data)
            if row is not None:
                rows.append(row)
                temps.append(amb)
        if len(rows) < min_cycles:
            continue
        X = np.nan_to_num(np.stack(rows), nan=0.0).astype(np.float32)
        rated = float(rated_ah) if rated_ah else float(np.nanmax(X[:5, 0]))
        rul, cens, eol = _rul_to_eol(X[:, 0], eol_frac * rated, decreasing=True)
        series.append(X[: eol + 1])
        ruls.append(rul)
        censored.append(cens)
        names.append(cell)

    fleet = _fleet_from_units(series, ruls, cap=cap, censored=censored,
                             n_groups=n_groups)
    pair = FleetPair(name="battery", train=fleet, test=None, cap=cap,
                     channel_names=list(BATTERY_FEATURES),
                     meta={**meta, "cells": names, "units": len(fleet),
                           "censored_units": int(sum(censored))})
    if use_cache:
        _save_fleet_cache(path_cache, pair)
        pair.meta["cache"] = path_cache
    return pair


# ═══════════════════════════════════════════════════════════════════════════
# CALCE (CS2 prismatic cells)
# ═══════════════════════════════════════════════════════════════════════════

CALCE_FEATURES = ("capacity_ah", "v_mean", "v_max", "current_mean", "duration_s")


def calce_available(data_dir: Optional[str] = None) -> bool:
    d = data_dir or CALCE_DIR
    return bool(glob.glob(os.path.join(d, "*", "*.xls*"))
                or glob.glob(os.path.join(d, "*.csv")))


def load_calce(
    data_dir: Optional[str] = None,
    cells: Optional[Sequence[str]] = None,
    eol_frac: float = 0.7,
    rated_ah: float = 1.1,
    cap: float = 200.0,
    n_groups: int = 2,
    min_cycles: int = 20,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    CALCE CS2 cells (1100 mAh prismatic), the standard companion to the NASA
    battery set.

    Two on-disk forms are accepted:
      `data/calce/CS2_35.csv`      a per-cycle table (columns `cycle` and
                                   `capacity`, however spelled) — preferred,
                                   because it is what most published
                                   preprocessing produces;
      `data/calce/CS2_35/*.xlsx`   the raw Arbin exports, aggregated here per
                                   `Cycle_Index` (needs openpyxl/xlrd).

    Same EOL convention as the NASA cells: capacity ≤ `eol_frac` × 1.1 Ah.
    """
    import pandas as pd

    d = data_dir or CALCE_DIR
    csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
    dirs = sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p))
    if cells:
        want = {str(c).upper() for c in cells}
        csvs = [f for f in csvs
                if os.path.splitext(os.path.basename(f))[0].upper() in want]
        dirs = [p for p in dirs if os.path.basename(p).upper() in want]
    if not csvs and not dirs:
        raise FileNotFoundError(
            f"nothing to read in {d}.  Get the CALCE CS2 data "
            "(calce.umd.edu battery data) and place either per-cell CSV "
            "summaries or the raw CS2_xx/ folders there.  See data/README.md.")

    def cap_col(df):
        for c in df.columns:
            if "discharge" in c.lower() and "capacit" in c.lower():
                return c
        for c in df.columns:
            if "capacit" in c.lower():
                return c
        return None

    def find(df, *words):
        for c in df.columns:
            low = c.lower()
            if all(w in low for w in words):
                return c
        return None

    series, ruls, censored, names = [], [], [], []

    for f in csvs:
        df = pd.read_csv(f)
        c_cap = cap_col(df)
        if c_cap is None:
            continue
        cyc = find(df, "cycle") or df.columns[0]
        g = df.groupby(df[cyc].astype(int))
        X = np.stack([
            g[c_cap].max().to_numpy(dtype=float),
            g[find(df, "voltage") or c_cap].mean().to_numpy(dtype=float),
            g[find(df, "voltage") or c_cap].max().to_numpy(dtype=float),
            g[find(df, "current") or c_cap].mean().to_numpy(dtype=float),
            g[find(df, "test", "time") or cyc].max().to_numpy(dtype=float),
        ], axis=1).astype(np.float32)
        if len(X) >= min_cycles:
            rul, cens, eol = _rul_to_eol(X[:, 0], eol_frac * rated_ah)
            series.append(np.nan_to_num(X[: eol + 1])); ruls.append(rul)
            censored.append(cens); names.append(os.path.basename(f)[:-4])

    for p in dirs:
        files = sorted(glob.glob(os.path.join(p, "*.xls*")))
        if not files:
            continue
        parts = []
        for f in files:
            try:
                parts.append(pd.read_excel(f))
            except ImportError as exc:                    # pragma: no cover
                raise ImportError(
                    "reading raw CALCE .xlsx needs openpyxl "
                    "(`pip install openpyxl`), or provide per-cell CSV "
                    "summaries instead") from exc
            except Exception:
                continue
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        c_cap = cap_col(df)
        cyc = find(df, "cycle", "index") or find(df, "cycle")
        if c_cap is None or cyc is None:
            continue
        g = df.groupby(df[cyc].astype(int))
        X = np.stack([
            g[c_cap].max().to_numpy(dtype=float),
            g[find(df, "voltage") or c_cap].mean().to_numpy(dtype=float),
            g[find(df, "voltage") or c_cap].max().to_numpy(dtype=float),
            g[find(df, "current") or c_cap].mean().to_numpy(dtype=float),
            g[find(df, "test", "time") or cyc].max().to_numpy(dtype=float),
        ], axis=1).astype(np.float32)
        if len(X) >= min_cycles:
            rul, cens, eol = _rul_to_eol(X[:, 0], eol_frac * rated_ah)
            series.append(np.nan_to_num(X[: eol + 1])); ruls.append(rul)
            censored.append(cens); names.append(os.path.basename(p))

    fleet = _fleet_from_units(series, ruls, cap=cap, censored=censored,
                              n_groups=n_groups)
    return FleetPair(name="calce", train=fleet, test=None, cap=cap,
                     channel_names=list(CALCE_FEATURES),
                     meta={"source": "CALCE CS2 (real)", "cells": names,
                           "features": list(CALCE_FEATURES),
                           "eol_rule": f"capacity <= {eol_frac:g} x {rated_ah} Ah",
                           "units": len(fleet),
                           "censored_units": int(sum(censored))})


# ═══════════════════════════════════════════════════════════════════════════
# IMS / Rexnord bearing run-to-failure
# ═══════════════════════════════════════════════════════════════════════════

# Which bearing actually failed in each test, from the dataset documentation.
# Bearings that did NOT fail are censored rather than dropped: "this bearing
# survived the whole test" is information, and the censored likelihood is the
# only part of the pipeline that can use it.
IMS_FAILURES = {"1st_test": {3: "inner race", 4: "roller element"},
                "2nd_test": {1: "outer race"},
                "3rd_test": {3: "outer race"}}
IMS_CHANNELS_PER_BEARING = {"1st_test": 2, "2nd_test": 1, "3rd_test": 1}
IMS_FILE_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


def ims_available(data_dir: Optional[str] = None) -> bool:
    d = data_dir or IMS_DIR
    return any(os.path.isdir(os.path.join(d, t)) for t in IMS_FAILURES)


def _ims_test_dir(root: str, test: str) -> Optional[str]:
    for cand in (os.path.join(root, test),
                 os.path.join(root, test, "txt"),
                 os.path.join(root, test, test)):
        if os.path.isdir(cand) and any(IMS_FILE_RE.match(f)
                                       for f in os.listdir(cand)):
            return cand
    return None


def load_ims(
    test: str = "2nd_test",
    data_dir: Optional[str] = None,
    unit: str = "bearing",                 # bearing | rig
    stride: int = 1,
    max_files: Optional[int] = None,
    cap: float = 200.0,
    n_groups: int = 3,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    IMS bearing test-to-failure data as a feature fleet.

    Each ASCII file is one 1-second snapshot of 20 480 samples at 20 kHz,
    recorded every 10 minutes; a test is 1-2 thousand such files ending in a
    failure.  Raw samples are not the modelling unit — the degradation lives on
    the file-to-file timescale — so each file is reduced to `TIME_FEATURE_NAMES`
    per accelerometer channel and one file becomes one timestep.

    `unit='bearing'` (default): one unit per bearing, channels = features of
    that bearing's accelerometer(s).  The four bearings share a shaft and a
    load, so their trajectories are not independent, but they fail separately —
    which is what makes per-unit splitting meaningful.
    `unit='rig'`: one unit for the whole test, channels = every bearing's
    features.  Fewer units (one), wider windows, and the cross-bearing coupling
    becomes something the joint density can exploit.

    RUL is time-to-end-of-test in FILES (10 min apart) for the bearings the
    documentation lists as failed; surviving bearings are right-censored.
    """
    root = data_dir or IMS_DIR
    tdir = _ims_test_dir(root, test)
    if tdir is None:
        raise FileNotFoundError(
            f"no IMS files for {test!r} under {root}.  Get the NASA PCoE "
            "'Bearing Data Set' (IMS/Rexnord) and unpack 1st_test/, 2nd_test/, "
            "3rd_test/ there.  See data/README.md.")

    key = f"ims_{test}_{unit}_{stride}_{max_files or 0}"
    path_cache = _cache(key)
    meta = {"source": "NASA PCoE IMS bearings (real)", "test": test,
            "unit": unit, "features": list(TIME_FEATURE_NAMES),
            "sampling": "20 kHz, 20480 samples/file, one file per 10 min",
            "failed_bearings": IMS_FAILURES.get(test, {}),
            "genuine_censoring": True}
    if use_cache and os.path.exists(path_cache):
        fleet = _load_fleet_cache(path_cache, cap, n_groups)
        return FleetPair(name=f"ims:{test}", train=fleet, test=None, cap=cap,
                         meta={**meta, "cache": path_cache, "units": len(fleet)})

    files = sorted(f for f in os.listdir(tdir) if IMS_FILE_RE.match(f))[::max(stride, 1)]
    if max_files:
        files = files[-int(max_files):]          # the END is where the failure is
    if not files:
        raise ValueError(f"{tdir} contains no timestamp-named IMS files")

    per_bearing = IMS_CHANNELS_PER_BEARING.get(test, 1)
    rows: List[np.ndarray] = []
    for f in files:
        arr = np.loadtxt(os.path.join(tdir, f))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        rows.append(np.concatenate([_time_features(arr[:, c])
                                    for c in range(arr.shape[1])]))
    M = np.stack(rows).astype(np.float32)         # (files, n_cols·n_features)
    n_cols = M.shape[1] // len(TIME_FEATURE_NAMES)
    n_bearings = max(n_cols // per_bearing, 1)
    F = len(TIME_FEATURE_NAMES)

    series, ruls, censored, chan_names = [], [], [], []
    failed = IMS_FAILURES.get(test, {})
    if unit == "rig":
        series.append(M)
        ruls.append(np.arange(len(M) - 1, -1, -1, dtype=float))
        censored.append(False)                    # the rig did reach failure
        chan_names = [f"c{c}_{n}" for c in range(n_cols) for n in TIME_FEATURE_NAMES]
    else:
        for b in range(n_bearings):
            cols = [c * F + k
                    for c in range(b * per_bearing, (b + 1) * per_bearing)
                    for k in range(F)]
            series.append(M[:, cols])
            ruls.append(np.arange(len(M) - 1, -1, -1, dtype=float))
            censored.append((b + 1) not in failed)
        chan_names = [f"ch{c}_{n}" for c in range(per_bearing)
                      for n in TIME_FEATURE_NAMES]

    fleet = _fleet_from_units(series, ruls, cap=cap, censored=censored,
                              n_groups=n_groups)
    pair = FleetPair(name=f"ims:{test}", train=fleet, test=None, cap=cap,
                     channel_names=chan_names,
                     meta={**meta, "files": len(files), "units": len(fleet),
                           "bearings": n_bearings,
                           "censored_units": int(sum(censored))})
    if use_cache:
        _save_fleet_cache(path_cache, pair)
        pair.meta["cache"] = path_cache
    return pair


# ═══════════════════════════════════════════════════════════════════════════
# NASA milling
# ═══════════════════════════════════════════════════════════════════════════

MILL_SIGNALS = ("smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle")


def milling_available(data_dir: Optional[str] = None) -> bool:
    d = data_dir or MILLING_DIR
    return bool(glob.glob(os.path.join(d, "mill*.mat")))


def load_milling(
    data_dir: Optional[str] = None,
    vb_eol: float = 0.6,
    cap: float = 20.0,
    n_groups: int = 3,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    NASA milling data set (`mill.mat`): 16 cases of a face-milling cutter run
    to wear-out, 167 runs in total, each holding six 9000-sample signals
    (spindle motor AC/DC current, table and spindle vibration, table and
    spindle acoustic emission) plus the measured flank wear VB.

    One case = one unit, one run = one timestep, channels = `TIME_FEATURE_NAMES`
    per signal.  Operating condition (depth of cut, feed, material) is constant
    within a case and becomes the `regime`, so per-regime normalisation removes
    the cutting-parameter offsets the way it removes C-MAPSS's conditions.

    RUL is runs-to-EOL, where EOL is the first run with VB ≥ `vb_eol` mm
    (0.6 mm is the usual wear-out criterion for this set); cases whose VB never
    reaches it are right-censored.  VB is missing for some runs and is
    interpolated — recorded in meta, because an interpolated label is a weaker
    label.
    """
    from scipy.io import loadmat

    d = data_dir or MILLING_DIR
    files = sorted(glob.glob(os.path.join(d, "mill*.mat")))
    if not files:
        raise FileNotFoundError(
            f"no mill*.mat in {d}.  Get the NASA PCoE 'Milling Data Set' and "
            "place mill.mat there.  See data/README.md.")

    key = f"milling_{vb_eol}"
    path_cache = _cache(key)
    meta = {"source": "NASA PCoE milling (real)",
            "features": [f"{s}_{n}" for s in MILL_SIGNALS for n in TIME_FEATURE_NAMES],
            "eol_rule": f"VB >= {vb_eol} mm", "genuine_censoring": True}
    if use_cache and os.path.exists(path_cache):
        fleet = _load_fleet_cache(path_cache, cap, n_groups)
        return FleetPair(name="milling", train=fleet, test=None, cap=cap,
                         meta={**meta, "cache": path_cache, "units": len(fleet)})

    mat = loadmat(files[0], squeeze_me=False, struct_as_record=False)
    key_name = next((k for k in mat if not k.startswith("__")), None)
    runs = np.ravel(mat[key_name])

    by_case: Dict[int, List[Any]] = {}
    for r in runs:
        by_case.setdefault(int(np.ravel(r.case)[0]), []).append(r)

    series, ruls, censored, regimes, n_interp = [], [], [], [], 0
    cond_ids: Dict[Tuple[float, float, float], int] = {}
    for case, entries in sorted(by_case.items()):
        entries.sort(key=lambda e: float(np.ravel(e.run)[0]))
        rows, vbs = [], []
        for e in entries:
            feats = []
            for s in MILL_SIGNALS:
                sig = np.ravel(getattr(e, s, np.zeros(1))).astype(float)
                feats.append(_time_features(sig) if sig.size > 8
                             else np.zeros(len(TIME_FEATURE_NAMES), dtype=np.float32))
            rows.append(np.concatenate(feats))
            vb = np.ravel(getattr(e, "VB", [np.nan])).astype(float)
            vbs.append(float(vb[0]) if vb.size else np.nan)
        X = np.nan_to_num(np.stack(rows)).astype(np.float32)
        vb = np.asarray(vbs, dtype=float)
        if np.isnan(vb).any() and not np.isnan(vb).all():
            n_interp += int(np.isnan(vb).sum())
            idx = np.arange(len(vb))
            vb = np.interp(idx, idx[~np.isnan(vb)], vb[~np.isnan(vb)])
        elif np.isnan(vb).all():
            vb = np.zeros(len(vb))
        rul, cens, eol = _rul_to_eol(vb, vb_eol, decreasing=False)
        e0 = entries[0]
        cond = tuple(float(np.ravel(getattr(e0, k, [0]))[0])
                     for k in ("DOC", "feed", "material"))
        rid = cond_ids.setdefault(cond, len(cond_ids))
        series.append(X[: eol + 1]); ruls.append(rul); censored.append(cens)
        regimes.append(np.full(eol + 1, rid, dtype=int))

    fleet = _fleet_from_units(series, ruls, cap=cap, censored=censored,
                              regimes=regimes, n_regimes=len(cond_ids),
                              n_groups=n_groups)
    pair = FleetPair(name="milling", train=fleet, test=None, cap=cap,
                     channel_names=list(meta["features"]),
                     meta={**meta, "units": len(fleet), "cases": len(by_case),
                           "n_regimes": len(cond_ids),
                           "interpolated_vb_values": n_interp,
                           "censored_units": int(sum(censored))})
    if use_cache:
        _save_fleet_cache(path_cache, pair)
        pair.meta["cache"] = path_cache
    return pair


# ═══════════════════════════════════════════════════════════════════════════
# Generic PCoE adapter (IGBT, capacitor, fatigue/Lamb wave, …)
# ═══════════════════════════════════════════════════════════════════════════
#
# The remaining PCoE sets ship in per-release layouts that differ between
# downloads (MATLAB structs of varying shape, per-specimen CSV dumps, mixed
# units).  Rather than guess a parser and mis-read a file silently — the
# failure mode this project has already hit five times — they are handled by
# an explicit, config-driven CSV adapter.  Convert once to per-unit CSVs and
# state the columns; the loader then does nothing clever.

PCOE_PRESETS: Dict[str, Dict[str, Any]] = {
    "igbt": {
        "subdir": "igbt",
        "eol_column": "collector_emitter_voltage",
        "eol_threshold": None,          # None -> end of record is EOL
        "note": "PCoE IGBT accelerated ageing: thermal/electrical cycling of "
                "IGBT devices. Convert each device to one CSV of per-cycle "
                "measurements before use.",
    },
    "capacitor": {
        "subdir": "capacitor",
        "eol_column": "capacitance",
        "eol_threshold": 0.8,           # 20% capacitance loss, the usual rule
        "eol_decreasing": True,
        "eol_relative": True,
        "note": "PCoE electrolytic capacitor ageing: EOL is conventionally a "
                "20% capacitance drop or 100% ESR rise.",
    },
    "fatigue": {
        "subdir": "fatigue",
        "eol_column": "crack_length",
        "eol_threshold": None,
        "eol_decreasing": False,
        "note": "PCoE aluminium lap-joint fatigue with Lamb-wave features and "
                "optical crack-length ground truth. Convert each specimen to "
                "one CSV: a row per inspection, Lamb-wave features as columns, "
                "crack_length as the health indicator.",
    },
}


def pcoe_available(preset: str = "igbt", data_dir: Optional[str] = None) -> bool:
    p = PCOE_PRESETS.get(preset)
    if p is None:
        return False
    d = os.path.join(data_dir or PCOE_DIR, p["subdir"])
    return bool(glob.glob(os.path.join(d, "*.csv")))


def load_pcoe_csv(
    preset: str = "igbt",
    data_dir: Optional[str] = None,
    unit_glob: str = "*.csv",
    time_col: Optional[str] = None,
    sensor_cols: Optional[Sequence[str]] = None,
    regime_col: Optional[str] = None,
    eol_column: Optional[str] = None,
    eol_threshold: Optional[float] = None,
    eol_decreasing: Optional[bool] = None,
    eol_relative: Optional[bool] = None,
    cap: float = 100.0,
    n_groups: int = 3,
    min_length: int = 12,
    seed: int = 0,
) -> FleetPair:
    """
    One CSV per unit -> a `FleetPair`, for the PCoE sets without a stable
    published layout (IGBT, capacitor, fatigue/Lamb-wave).

    Expected on disk:  data/pcoe/<preset>/<unit>.csv, a row per measurement
    step, numeric columns only (any `time_col` is dropped from the features and
    any `regime_col` becomes the operating regime).

    End of life: the first row where `eol_column` crosses `eol_threshold`
    (interpreted relative to the unit's initial value when `eol_relative`), or
    the end of the record when no threshold is given.  Units that never cross
    are right-censored.

    This adapter is deliberately dumb.  It reads what you tell it to read; it
    does not sniff MATLAB structs or guess which column is the health
    indicator, because a wrong guess here is invisible in the loss curve.
    """
    import pandas as pd

    p = PCOE_PRESETS.get(preset)
    if p is None:
        raise KeyError(f"unknown PCoE preset {preset!r}; "
                       f"expected one of {sorted(PCOE_PRESETS)}")
    d = os.path.join(data_dir or PCOE_DIR, p["subdir"])
    files = sorted(glob.glob(os.path.join(d, unit_glob)))
    if not files:
        raise FileNotFoundError(
            f"no {unit_glob} in {d}.  {p['note']}  See data/README.md.")

    eol_column = eol_column or p.get("eol_column")
    eol_threshold = p.get("eol_threshold") if eol_threshold is None else eol_threshold
    decreasing = p.get("eol_decreasing", True) if eol_decreasing is None else eol_decreasing
    relative = p.get("eol_relative", False) if eol_relative is None else eol_relative

    series, ruls, censored, regimes, names = [], [], [], [], []
    cols_used: List[str] = []
    for f in files:
        df = pd.read_csv(f)
        num = df.select_dtypes(include="number")
        feat_cols = list(sensor_cols) if sensor_cols else [
            c for c in num.columns if c not in {time_col, regime_col}]
        if not feat_cols:
            continue
        X = np.nan_to_num(num[feat_cols].to_numpy(dtype=np.float32))
        if len(X) < min_length:
            continue
        if eol_column and eol_column in num.columns:
            v = num[eol_column].to_numpy(dtype=float)
            thr = (eol_threshold * float(v[0]) if (relative and eol_threshold is not None)
                   else eol_threshold)
            if thr is None:
                rul, cens, eol = np.arange(len(X) - 1, -1, -1, dtype=float), False, len(X) - 1
            else:
                rul, cens, eol = _rul_to_eol(v, thr, decreasing=decreasing)
        else:
            rul, cens, eol = np.arange(len(X) - 1, -1, -1, dtype=float), False, len(X) - 1
        series.append(X[: eol + 1]); ruls.append(rul); censored.append(cens)
        if regime_col and regime_col in df.columns:
            codes = pd.factorize(df[regime_col])[0][: eol + 1]
            regimes.append(np.asarray(codes, dtype=int))
        else:
            regimes.append(np.zeros(eol + 1, dtype=int))
        names.append(os.path.basename(f)[:-4])
        cols_used = feat_cols

    n_reg = int(max((r.max() for r in regimes if len(r)), default=0)) + 1
    fleet = _fleet_from_units(series, ruls, cap=cap, censored=censored,
                              regimes=regimes, n_regimes=n_reg, n_groups=n_groups)
    return FleetPair(name=f"pcoe:{preset}", train=fleet, test=None, cap=cap,
                     channel_names=list(cols_used),
                     meta={"source": f"NASA PCoE {preset} (real, user-converted)",
                           "preset": preset, "units": len(fleet), "unit_names": names,
                           "features": list(cols_used),
                           "eol_column": eol_column, "eol_threshold": eol_threshold,
                           "censored_units": int(sum(censored)),
                           "caveats": ["layout is user-supplied: the loader trusts "
                                       "the column names given in the config"]})
