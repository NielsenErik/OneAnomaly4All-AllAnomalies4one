"""
REAL spacecraft telemetry with REAL anomaly annotations, behind the same
`Fleet` / `FleetPair` interface the turbofan loaders use.

Why this module exists
──────────────────────
Every detection number in the PoC so far was scored against anomalies WE
injected, because neither C-MAPSS nor N-C-MAPSS ships annotations.  Injection
buys the per-channel ground truth the localisation claim needs, but it also
means the anomalies are ours.  The three sources here are the other half of the
argument: operator-annotated faults in real flight telemetry, where the label
is somebody else's.

    esa      ESA Anomalies Dataset / ESA-ADB (Kotowski et al., 2024; v2 2025)
             — the current state of the art.  Real telemetry from three ESA
             missions, annotated by spacecraft operations engineers, PER
             CHANNEL.  That last point is what makes it uniquely valuable
             here: it is the only source in the whole catalogue that gives
             REAL localisation ground truth, so the exact-attribution result
             (0.902 vs 0.498 for sampling-SHAP on injected faults) can be
             re-run against faults nobody in this project designed.
    opssat   OPSSAT-AD (ESA CubeSat) — 2123 short single-channel segments,
             ~20% anomalous, segment-level labels.  Small, univariate, messy
             (gaps, sampling-rate changes); a robustness check, not a
             multivariate benchmark.
    smapmsl  SMAP / MSL (Hundman et al., 2018) — the field's default, included
             for comparability and NOT recommended as evidence.  Its caveats
             are attached to every run (see `SMAP_MSL_CAVEATS`) rather than
             left in a footnote.

The unifying idea
─────────────────
A `Fleet` whose `labels` field is non-None carries, for every timestep and
every channel, the operators' annotation code:

    0 NOMINAL   1 ANOMALY   2 RARE_EVENT   3 GAP   4 INVALID

(the ESA-ADB `AnnotationLabel` enum, reused verbatim for the other sources so
one task builder handles all three).  `datasets.make_ad_task_labeled` turns
that into an `ADTask` with real `y_test` AND real `affected_test`, so nothing
downstream — circuit, baselines, attribution, metrics — needs to know which
source it is looking at.

Nothing is downloaded automatically; `python -m poc.time_series.check_data`
reports what is on disk and how to get the rest.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .data import Fleet
from .data_real import DATA_ROOT, FleetPair, correlation_groups

ESA_DIR = os.path.join(DATA_ROOT, "esa_adb")
OPSSAT_DIR = os.path.join(DATA_ROOT, "opssat")
SMAP_MSL_DIR = os.path.join(DATA_ROOT, "smap_msl")
CACHE_DIR = os.path.join(DATA_ROOT, "cache")

# ESA-ADB annotation codes (kplabs-pl/ESA-ADB, notebooks/data-prep/utils.py).
NOMINAL, ANOMALY, RARE_EVENT, GAP, INVALID = 0, 1, 2, 3, 4
CODE_NAMES = {NOMINAL: "normal", ANOMALY: "anomaly", RARE_EVENT: "rare_event",
              GAP: "gap", INVALID: "invalid"}


# ═══════════════════════════════════════════════════════════════════════════
# Caveats — carried in meta so they land in every run's log
# ═══════════════════════════════════════════════════════════════════════════

SMAP_MSL_CAVEATS = (
    "Wu & Keogh (TKDE 2021): large parts are solvable by a one-line moving "
    "standard deviation — run the `trivial` baseline and report it.",
    "Unrealistic anomaly density and probable mislabelling, MSL especially.",
    "Run-to-failure bias: anomalies cluster near the END of each test series, "
    "so 'flag the last points' scores well.",
    "The 82 channels are unsynchronised fragments — this is NOT a multivariate "
    "dataset. dims='first' (the default here) models each channel on its own, "
    "which is what most recent papers do silently.",
    "Columns 1.. of each .npy are one-hot COMMAND encodings, not sensors; "
    "dims='all' therefore mixes telemetry with command metadata.",
)

ESA_CAVEATS = (
    "Anomalies are present in the training half too (ESA-ADB is semi-"
    "supervised, not clean-train): `train_on_clean` drops annotated training "
    "windows, which is a choice, not a property of the data.",
    "Rare nominal events are atypical-but-expected; ESA-ADB's own baselines "
    "treat them as anomalies (rare_events='anomaly' here). Report the setting.",
    "Telemetry is resampled with zero-order hold (30 s Mission1 / 18 s "
    "Mission2), so a window spans real but irregularly-sampled time.",
)

OPSSAT_CAVEATS = (
    "Labels are per SEGMENT, not per timestep: every window of an anomalous "
    "segment inherits the label, which inflates any point-wise metric.",
    "Univariate by construction (one channel per segment); the 9 channels have "
    "very different scales, so normalisation is per channel (regime=channel).",
    "Raw telemetry has gaps, artifacts and sampling-rate changes — segment "
    "lengths vary by two orders of magnitude.",
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _blank_fleet_fields(series: List[np.ndarray]) -> Tuple[List, List, List]:
    """
    RUL / health / censoring for a fleet that has no failure to run to.

    Zeros, deliberately: an annotated telemetry segment has no remaining life,
    and any value put here would be an invented label.  `make_ad_task_labeled`
    never reads them; `build_rul_task` is refused for these sources upstream
    (`catalog.Source.tasks`), so the zeros can never quietly become a target.
    """
    rul = [np.zeros(len(s), dtype=float) for s in series]
    health = [np.zeros(len(s), dtype=float) for s in series]
    return rul, health, [False] * len(series)


def _make_annotated_fleet(
    series: List[np.ndarray],
    labels: List[np.ndarray],
    regimes: Optional[List[np.ndarray]] = None,
    n_regimes: int = 1,
    groups: Optional[List[List[int]]] = None,
) -> Fleet:
    rul, health, censored = _blank_fleet_fields(series)
    C = series[0].shape[1]
    regimes = regimes or [np.zeros(len(s), dtype=int) for s in series]
    return Fleet(series, rul, regimes, health, censored,
                 n_channels=C, n_regimes=n_regimes,
                 channel_groups=groups or [list(range(C))],
                 labels=labels)


def _thin(X: np.ndarray, L: np.ndarray, max_samples: Optional[int]
          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep at most `max_samples` rows by taking a CONTIGUOUS tail-preserving
    stride, never a random subset.

    Random subsampling of a time series destroys exactly the thing being
    modelled — a window over shuffled rows is not a window.  Striding keeps the
    ordering and the window semantics; it lengthens the effective sampling
    period, which is recorded in meta.
    """
    if not max_samples or len(X) <= max_samples:
        return X, L
    step = int(np.ceil(len(X) / max_samples))
    return X[::step], L[::step]


def _pack_annotated(fleet: Fleet, tag: str) -> Dict[str, np.ndarray]:
    lens = np.array([len(s) for s in fleet.series], dtype=np.int64)
    return {
        f"{tag}_X": np.concatenate(fleet.series, 0).astype(np.float32),
        f"{tag}_L": np.concatenate(fleet.labels or [], 0).astype(np.uint8),
        f"{tag}_reg": np.concatenate(fleet.regime).astype(np.int32),
        f"{tag}_lens": lens,
        f"{tag}_nreg": np.array([fleet.n_regimes]),
    }


def _unpack_annotated(z, tag: str) -> Optional[Fleet]:
    if f"{tag}_lens" not in z:
        return None
    lens = z[f"{tag}_lens"]
    b = np.concatenate([[0], np.cumsum(lens)])
    cut = lambda a: [a[b[i]:b[i + 1]] for i in range(len(lens))]
    return _make_annotated_fleet(cut(z[f"{tag}_X"]), cut(z[f"{tag}_L"]),
                                 cut(z[f"{tag}_reg"]), int(z[f"{tag}_nreg"][0]))


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".npz")


def _save_cache(path: str, pair: FleetPair) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = _pack_annotated(pair.train, "train")
    if pair.test is not None:
        blob.update(_pack_annotated(pair.test, "test"))
    np.savez_compressed(path, **blob)


def _load_cache(path: str, name: str, cap: float, meta: Dict[str, Any]) -> FleetPair:
    z = np.load(path, allow_pickle=False)
    train = _unpack_annotated(z, "train")
    test = _unpack_annotated(z, "test")
    return FleetPair(name=name, train=train, test=test, cap=cap,
                     meta={**meta, "cache": path})


# ═══════════════════════════════════════════════════════════════════════════
# ESA Anomalies Dataset / ESA-ADB
# ═══════════════════════════════════════════════════════════════════════════
#
# Two on-disk forms are accepted, because both exist in the wild:
#
#   RAW (Zenodo 10.5281/zenodo.12528696, ~31 GB over three missions)
#       data/esa_adb/ESA-Mission1/
#           channels/channel_1.zip ...      pickled DataFrames, DatetimeIndex
#           telecommands/*.zip
#           labels.csv          ID, Channel, StartTime, EndTime
#           anomaly_types.csv   ID, Category (Anomaly | Rare Event | ...), ...
#           telecommands.csv    Telecommand, Priority
#
#   PREPROCESSED (what kplabs-pl/ESA-ADB's data-prep scripts emit for TimeEval)
#       data/esa_adb/preprocessed/multivariate/ESA-Mission1-semisupervised/
#           84_months.train.csv   84_months.test.csv
#       columns: timestamp, <channel_*/telecommand_* values>, is_anomaly_<param>
#
# The preprocessed form is preferred when present: it is what the published
# baselines consume, so numbers stay comparable, and reading it costs one
# pandas call with `usecols` instead of a few hundred pickle loads.

ESA_MISSIONS = ("Mission1", "Mission2", "Mission3")

ESA_MISSION_INFO: Dict[str, Dict[str, Any]] = {
    # split dates and resampling from notebooks/data-prep/Mission{1,2}_*.py
    "Mission1": {"test_split": "2007-01-01", "train_split": "2007-01-01",
                 "resample_s": 30, "lightweight": list(range(41, 47)),
                 "derivative_channels": list(range(4, 12)), "months": 84},
    "Mission2": {"test_split": "2001-10-01", "train_split": "2001-10-01",
                 "resample_s": 18, "lightweight": list(range(18, 29)),
                 "derivative_channels": [], "months": 21},
    # Mission3 ships in the dataset but is not part of the benchmark: no
    # published split, so we halve the timeline and say so in meta.
    "Mission3": {"test_split": None, "train_split": None,
                 "resample_s": 15, "lightweight": [],
                 "derivative_channels": [], "months": None},
}


def _esa_mission_name(mission: str) -> str:
    m = str(mission).strip().replace("ESA-", "").replace("-", "").replace("_", "")
    m = m.capitalize() if not m.lower().startswith("mission") else "Mission" + m[7:]
    if m not in ESA_MISSIONS:
        raise KeyError(f"unknown ESA mission {mission!r}; expected one of {ESA_MISSIONS}")
    return m


def esa_raw_dir(mission: str, data_dir: Optional[str] = None) -> Optional[str]:
    d = os.path.join(data_dir or ESA_DIR, f"ESA-{_esa_mission_name(mission)}")
    return d if os.path.isdir(os.path.join(d, "channels")) else None


def esa_preprocessed_files(mission: str, data_dir: Optional[str] = None,
                           split: Optional[str] = None
                           ) -> Optional[Tuple[str, str]]:
    """(train.csv, test.csv) for the TimeEval-preprocessed form, if present."""
    root = os.path.join(data_dir or ESA_DIR, "preprocessed", "multivariate")
    name = _esa_mission_name(mission)
    for sub in sorted(glob.glob(os.path.join(root, f"ESA-{name}*"))):
        trains = sorted(glob.glob(os.path.join(sub, "*.train.csv")))
        if split:
            trains = [t for t in trains if os.path.basename(t).startswith(f"{split}.")]
        for tr in trains:
            te = tr.replace(".train.csv", ".test.csv")
            if os.path.exists(te):
                return tr, te
    return None


def esa_available(mission: str = "Mission1", data_dir: Optional[str] = None) -> bool:
    try:
        return (esa_preprocessed_files(mission, data_dir) is not None
                or esa_raw_dir(mission, data_dir) is not None)
    except KeyError:
        return False


def _resolve_esa_channels(spec: Union[str, Sequence],
                          available: Sequence[str],
                          mission: str) -> List[str]:
    """
    'lightweight' | 'all' | explicit names/indices -> concrete column names.

    `lightweight` is ESA-ADB's own suggested subset (Mission1 channels 41-46,
    Mission2 18-28): chosen by the benchmark authors to be challenging,
    operationally interesting, and analysable in isolation.  It is the default
    because the full set is 76-100 channels, and a joint circuit over all of
    them at 30 s resolution is a different (much larger) experiment.
    """
    if isinstance(spec, str):
        key = spec.lower()
        if key == "all":
            return list(available)
        if key in ("lightweight", "light", "subset"):
            want = ESA_MISSION_INFO[mission]["lightweight"]
            if not want:
                return list(available)
            names = [f"channel_{i}" for i in want]
            got = [n for n in names if n in available]
            if not got:
                raise KeyError(
                    f"lightweight subset {names[0]}..{names[-1]} not found among "
                    f"{len(available)} available channels for {mission}")
            return got
        raise KeyError(f"unknown ESA channel selection {spec!r} "
                       "(expected 'lightweight', 'all', or an explicit list)")
    out = []
    for c in spec:
        n = f"channel_{c}" if isinstance(c, (int, np.integer)) else str(c)
        if n not in available:
            raise KeyError(f"channel {n!r} not in {mission} "
                           f"({len(available)} channels available)")
        out.append(n)
    return out


def _esa_from_preprocessed(
    train_csv: str, test_csv: str, mission: str, channels: Union[str, Sequence],
    include_telecommands: bool, max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Tuple[Fleet, Fleet, List[str], Dict[str, Any]]:
    import pandas as pd

    header = pd.read_csv(train_csv, nrows=0).columns.tolist()
    params = [c for c in header
              if c != "timestamp" and not c.startswith("is_anomaly_")]
    chan_names = [c for c in params if c.startswith("channel")]
    tc_names = [c for c in params if not c.startswith("channel")]
    keep = _resolve_esa_channels(channels, chan_names, mission)
    if include_telecommands:
        keep = keep + tc_names

    usecols = keep + [f"is_anomaly_{c}" for c in keep
                      if f"is_anomaly_{c}" in header]

    def read(path: str, max_samples: Optional[int]):
        df = pd.read_csv(path, usecols=usecols)
        X = df[keep].to_numpy(dtype=np.float32)
        L = np.zeros_like(X, dtype=np.uint8)
        for j, c in enumerate(keep):
            col = f"is_anomaly_{c}"
            if col in df.columns:
                L[:, j] = df[col].to_numpy(dtype=np.uint8)
        return _thin(np.nan_to_num(X), L, max_samples)

    Xtr, Ltr = read(train_csv, max_train_samples)
    Xte, Lte = read(test_csv, max_test_samples)
    meta = {"form": "preprocessed", "train_csv": os.path.basename(train_csv),
            "test_csv": os.path.basename(test_csv),
            "telecommands": len(keep) - len([c for c in keep if c.startswith("channel")])}
    return (_make_annotated_fleet([Xtr], [Ltr]),
            _make_annotated_fleet([Xte], [Lte]), keep, meta)


def _esa_from_raw(
    root: str, mission: str, channels: Union[str, Sequence], resample_s: int,
    rare_events: str, derivative: bool, max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Tuple[Fleet, Fleet, List[str], Dict[str, Any]]:
    """
    Parse the Zenodo release directly, reproducing the reference preprocessing:
    zero-order-hold resampling to the mission's dominant rate, per-channel
    annotation from labels.csv × anomaly_types.csv, and (Mission1) the first
    difference of the monotonic channels 4-11.
    """
    import pandas as pd

    info = ESA_MISSION_INFO[mission]
    avail = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(root, "channels", "*.zip")))
    if not avail:
        raise FileNotFoundError(f"{root}/channels/*.zip is empty")
    keep = _resolve_esa_channels(channels, avail, mission)

    labels_df = pd.read_csv(os.path.join(root, "labels.csv"))
    for col in ("StartTime", "EndTime"):
        labels_df[col] = pd.to_datetime(labels_df[col], errors="coerce", utc=False)
    types_path = os.path.join(root, "anomaly_types.csv")
    cats: Dict[Any, str] = {}
    if os.path.exists(types_path):
        tdf = pd.read_csv(types_path)
        cat_col = "Category" if "Category" in tdf.columns else tdf.columns[-1]
        cats = dict(zip(tdf["ID"], tdf[cat_col]))

    def code_of(row) -> int:
        cat = str(cats.get(row["ID"], row.get("Category", "Anomaly"))).lower()
        if "rare" in cat:
            return RARE_EVENT
        if "gap" in cat or "communication" in cat:
            return GAP
        if "invalid" in cat:
            return INVALID
        return ANOMALY

    rule = pd.Timedelta(seconds=int(resample_s))
    frames, label_frames = {}, {}
    for name in keep:
        df = pd.read_pickle(os.path.join(root, "channels", f"{name}.zip"))
        if isinstance(df, pd.Series):
            df = df.to_frame(name=name)
        df = df.rename(columns={df.columns[0]: "value"})[["value"]]
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated()].sort_index()
        idx = int(name.split("_")[1]) if name.split("_")[-1].isdigit() else -1
        if derivative and idx in info["derivative_channels"]:
            df["value"] = np.diff(df["value"].to_numpy(), append=df["value"].iloc[-1])
        df["label"] = np.uint8(NOMINAL)
        for _, row in labels_df[labels_df["Channel"] == name].iterrows():
            df.loc[row["StartTime"]:row["EndTime"], "label"] = code_of(row)
        rs = df.resample(rule).ffill()
        # An annotated sample must never be lost to resampling: take the
        # strongest code seen inside each bucket, not the last one.
        rs["label"] = df["label"].resample(rule).max().reindex(rs.index).ffill()
        frames[name] = rs["value"]
        label_frames[name] = rs["label"].fillna(NOMINAL).astype(np.uint8)

    values = pd.concat(frames, axis=1).ffill().bfill()
    codes = pd.concat(label_frames, axis=1).ffill().bfill().astype(np.uint8)

    split = info["test_split"]
    ts = pd.to_datetime(split) if split else values.index[len(values) // 2]
    tr_sel, te_sel = values.index <= ts, values.index > ts

    def take(sel, max_samples):
        return _thin(values[sel].to_numpy(dtype=np.float32),
                     codes[sel].to_numpy(dtype=np.uint8), max_samples)

    Xtr, Ltr = take(tr_sel, max_train_samples)
    Xte, Lte = take(te_sel, max_test_samples)
    if not len(Xtr) or not len(Xte):
        raise ValueError(f"ESA {mission}: split at {ts} left an empty half")
    meta = {"form": "raw", "resample_s": int(resample_s),
            "split_at": str(ts), "split_source": "published" if split else "median",
            "rare_events": rare_events}
    return (_make_annotated_fleet([Xtr], [Ltr]),
            _make_annotated_fleet([Xte], [Lte]), keep, meta)


def load_esa_adb(
    mission: str = "Mission1",
    data_dir: Optional[str] = None,
    channels: Union[str, Sequence] = "lightweight",
    form: str = "auto",                       # auto | preprocessed | raw
    split: Optional[str] = None,              # preprocessed: 84_months, 21_months, ...
    resample_s: Optional[int] = None,
    include_telecommands: bool = False,
    rare_events: str = "anomaly",
    derivative: bool = True,
    max_train_samples: Optional[int] = 400_000,
    max_test_samples: Optional[int] = 200_000,
    n_groups: int = 3,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    Load one ESA mission as a two-segment annotated `FleetPair`.

    The fitting fleet is the first half of the mission timeline, the held-out
    fleet the second — the benchmark's own chronological split (84 months for
    Mission1, 21 for Mission2), which exists so no algorithm can see the
    future.  There is exactly one "unit" on each side: a mission is one
    continuous multivariate stream, not a fleet of units.

    `max_*_samples` strides the stream rather than sampling it (see `_thin`);
    the defaults keep a run to a few hundred thousand windows.  Set them to
    None for the full 84 months and expect a long night.
    """
    m = _esa_mission_name(mission)
    info = ESA_MISSION_INFO[m]
    resample_s = int(resample_s or info["resample_s"])
    ch_key = channels if isinstance(channels, str) else "-".join(map(str, channels))
    key = (f"esa_{m}_{ch_key}_{form}_{split or 'default'}_{resample_s}_"
           f"{int(include_telecommands)}_{max_train_samples or 0}_{max_test_samples or 0}")
    base_meta = {"mission": m, "channels_spec": str(channels),
                 "rare_events": rare_events, "source": "ESA-ADB (real, annotated)",
                 "caveats": list(ESA_CAVEATS), "labels_are_real": True,
                 "per_channel_truth": True}
    cache = _cache_path(key)
    if use_cache and os.path.exists(cache):
        return _load_cache(cache, f"esa:{m}", cap=0.0, meta=base_meta)

    pre = esa_preprocessed_files(m, data_dir, split)
    raw = esa_raw_dir(m, data_dir)
    if form == "preprocessed" and pre is None:
        raise FileNotFoundError(
            f"no preprocessed ESA-{m} CSVs under {data_dir or ESA_DIR}/preprocessed; "
            "run kplabs-pl/ESA-ADB's data-prep script or use form='raw'")
    if form == "raw" and raw is None:
        raise FileNotFoundError(f"no raw ESA-{m} folder under {data_dir or ESA_DIR}")
    if pre is None and raw is None:
        raise FileNotFoundError(
            f"ESA-{m} not found in {data_dir or ESA_DIR}.  Download the ESA "
            "Anomalies Dataset (Zenodo 10.5281/zenodo.12528696) and unpack "
            "ESA-Mission1/ESA-Mission2 there.  See data/README.md.")

    if pre is not None and form in ("auto", "preprocessed"):
        train, test, names, meta = _esa_from_preprocessed(
            *pre, mission=m, channels=channels,
            include_telecommands=include_telecommands,
            max_train_samples=max_train_samples, max_test_samples=max_test_samples)
    else:
        train, test, names, meta = _esa_from_raw(
            raw, m, channels, resample_s, rare_events, derivative,
            max_train_samples, max_test_samples)

    groups = correlation_groups(train.series[0], n_groups)
    train.channel_groups = test.channel_groups = groups
    pair = FleetPair(name=f"esa:{m}", train=train, test=test, cap=0.0,
                     channel_names=names,
                     meta={**base_meta, **meta,
                           "n_channels": train.n_channels,
                           "train_samples": int(len(train.series[0])),
                           "test_samples": int(len(test.series[0]))})
    if use_cache:
        _save_cache(cache, pair)
        pair.meta["cache"] = cache
    return pair


# ═══════════════════════════════════════════════════════════════════════════
# OPSSAT-AD
# ═══════════════════════════════════════════════════════════════════════════

def opssat_file(data_dir: Optional[str] = None) -> Optional[str]:
    d = data_dir or OPSSAT_DIR
    for name in ("segments.csv", "dataset.csv"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def opssat_available(data_dir: Optional[str] = None) -> bool:
    p = opssat_file(data_dir)
    return bool(p) and os.path.basename(p) == "segments.csv"


def load_opssat(
    data_dir: Optional[str] = None,
    min_length: int = 16,
    max_length: Optional[int] = 2000,
    channels: Optional[Sequence[str]] = None,
    train_frac: float = 0.7,
    n_groups: int = 1,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    OPSSAT-AD as a fleet of univariate segments.

    `segments.csv` (kplabs-pl/OPS-SAT-AD) has one row per sample with columns
    `timestamp, channel, value, sampling, anomaly, train, segment`.  Each
    segment becomes one unit; the dataset's own `train` flag decides which
    fleet it lands in (falling back to a seeded per-segment split if the
    column is absent).

    Two honest consequences of the format, both recorded in meta:
      * the label is per SEGMENT, so every window inside an anomalous segment
        is labelled anomalous — a point-wise metric is optimistic here;
      * the 9 channels have unrelated scales, so `regime` is set to the channel
        id and per-regime normalisation is the right setting (`per_regime`
        defaults to auto, which switches on because n_regimes > 1).
    """
    import pandas as pd

    path = opssat_file(data_dir)
    if path is None or os.path.basename(path) != "segments.csv":
        raise FileNotFoundError(
            f"{(data_dir or OPSSAT_DIR)}/segments.csv not found.  Download "
            "OPSSAT-AD (Zenodo 10.5281/zenodo.12588359) and place segments.csv "
            "there.  See data/README.md.")

    key = f"opssat_{min_length}_{max_length or 0}_{'-'.join(channels or []) or 'all'}_{seed}"
    cache = _cache_path(key)
    base_meta = {"source": "OPSSAT-AD (real, annotated)",
                 "caveats": list(OPSSAT_CAVEATS), "labels_are_real": True,
                 "per_channel_truth": False, "segment_level_labels": True}
    if use_cache and os.path.exists(cache):
        return _load_cache(cache, "opssat", cap=0.0, meta=base_meta)

    df = pd.read_csv(path)
    need = {"channel", "value", "anomaly", "segment"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing columns {sorted(missing)}")
    if channels:
        df = df[df["channel"].isin(list(channels))]
    chan_names = sorted(df["channel"].astype(str).unique())
    chan_id = {c: i for i, c in enumerate(chan_names)}

    rng = np.random.default_rng(seed)
    tr: Dict[str, list] = {"X": [], "L": [], "R": []}
    te: Dict[str, list] = {"X": [], "L": [], "R": []}
    n_drop = 0
    for seg, g in df.groupby("segment", sort=True):
        v = g["value"].to_numpy(dtype=np.float32).reshape(-1, 1)
        if len(v) < min_length:
            n_drop += 1
            continue
        if max_length and len(v) > max_length:
            v = v[:max_length]
        is_anom = int(np.asarray(g["anomaly"])[0]) == 1
        L = np.full((len(v), 1), ANOMALY if is_anom else NOMINAL, dtype=np.uint8)
        cid = chan_id[str(np.asarray(g["channel"])[0])]
        R = np.full(len(v), cid, dtype=int)
        if "train" in g.columns:
            in_train = int(np.asarray(g["train"])[0]) == 1
        else:
            in_train = bool(rng.random() < train_frac)
        d = tr if in_train else te
        d["X"].append(np.nan_to_num(v)); d["L"].append(L); d["R"].append(R)

    if not tr["X"] or not te["X"]:
        raise ValueError("OPSSAT: one side of the split is empty — check the "
                         "`train` column or lower `min_length`")
    n_reg = max(len(chan_names), 1)
    train = _make_annotated_fleet(tr["X"], tr["L"], tr["R"], n_reg)
    test = _make_annotated_fleet(te["X"], te["L"], te["R"], n_reg)
    pair = FleetPair(name="opssat", train=train, test=test, cap=0.0,
                     channel_names=["value"],
                     meta={**base_meta, "channels": chan_names,
                           "train_segments": len(train), "test_segments": len(test),
                           "dropped_short_segments": n_drop,
                           "anomalous_test_segments":
                               int(sum(int((l > 0).any()) for l in test.labels or []))})
    if use_cache:
        _save_cache(cache, pair)
        pair.meta["cache"] = cache
    return pair


# ═══════════════════════════════════════════════════════════════════════════
# SMAP / MSL
# ═══════════════════════════════════════════════════════════════════════════

def smap_msl_available(data_dir: Optional[str] = None) -> bool:
    d = data_dir or SMAP_MSL_DIR
    return (os.path.exists(os.path.join(d, "labeled_anomalies.csv"))
            and os.path.isdir(os.path.join(d, "test")))


def _parse_sequences(text: str) -> List[Tuple[int, int]]:
    """'[[1899, 2099], [3000, 3100]]' -> [(1899, 2099), (3000, 3100)]."""
    nums = [int(n) for n in re.findall(r"-?\d+", str(text))]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def load_smap_msl(
    spacecraft: str = "SMAP",                # SMAP | MSL | both
    data_dir: Optional[str] = None,
    dims: str = "first",                     # first | all
    channels: Optional[Sequence[str]] = None,
    n_groups: int = 1,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    SMAP/MSL as a fleet of per-channel segments — included for comparability,
    with its flaws attached rather than hidden (`SMAP_MSL_CAVEATS`).

    Each channel contributes two units: its `train/<chan>.npy` (assumed
    nominal — the benchmark provides no training labels) and its
    `test/<chan>.npy` labelled from `labeled_anomalies.csv`.

    `dims='first'` keeps only column 0, the telemetry value being predicted;
    columns 1.. are one-hot encoded commands, not sensors.  That makes every
    unit univariate and lets all channels share one fleet — which is exactly
    the "quietly collapse to univariate" step the ESA team criticises, made
    explicit here.  `dims='all'` keeps the command encodings and then requires
    every selected channel to have the same width (SMAP 25, MSL 55), so it
    cannot silently mix the two spacecraft.
    """
    d = data_dir or SMAP_MSL_DIR
    if not smap_msl_available(d):
        raise FileNotFoundError(
            f"{d}/labeled_anomalies.csv or {d}/test/ not found.  Get the "
            "SMAP/MSL archive from khundman/telemanom (data.zip) and unpack "
            "train/, test/ and labeled_anomalies.csv there.  See data/README.md.")

    sc = str(spacecraft).upper()
    key = f"smapmsl_{sc}_{dims}_{'-'.join(channels or []) or 'all'}"
    cache = _cache_path(key)
    base_meta = {"source": "SMAP/MSL (real labels, contested)",
                 "caveats": list(SMAP_MSL_CAVEATS), "labels_are_real": True,
                 "per_channel_truth": False, "spacecraft": sc, "dims": dims,
                 "not_recommended_as_primary_evidence": True}
    if use_cache and os.path.exists(cache):
        return _load_cache(cache, f"smapmsl:{sc}", cap=0.0, meta=base_meta)

    import csv
    rows = []
    with open(os.path.join(d, "labeled_anomalies.csv")) as f:
        for r in csv.DictReader(f):
            cid = r.get("chan_id") or r.get("channel_id") or r.get("channel")
            craft = str(r.get("spacecraft", "")).upper()
            if sc != "BOTH" and craft and craft != sc:
                continue
            if channels and cid not in set(channels):
                continue
            rows.append((cid, _parse_sequences(r.get("anomaly_sequences", "[]")),
                         r.get("class", "")))
    if not rows:
        raise ValueError(f"no channels for spacecraft={spacecraft!r} in labeled_anomalies.csv")

    def read(split: str, cid: str) -> Optional[np.ndarray]:
        p = os.path.join(d, split, f"{cid}.npy")
        if not os.path.exists(p):
            return None
        a = np.load(p).astype(np.float32)
        return a.reshape(len(a), -1)

    tr_X: List[np.ndarray] = []
    te_X: List[np.ndarray] = []
    tr_L: List[np.ndarray] = []
    te_L: List[np.ndarray] = []
    widths, used, n_anom = set(), [], 0
    for cid, seqs, klass in rows:
        Xte = read("test", cid)
        if Xte is None:
            continue
        Xtr = read("train", cid)
        if dims == "first":
            Xte = Xte[:, :1]
            Xtr = None if Xtr is None else Xtr[:, :1]
        widths.add(Xte.shape[1])
        Lte = np.zeros_like(Xte, dtype=np.uint8)
        for a, b in seqs:
            a, b = max(0, int(a)), min(len(Xte) - 1, int(b))
            if b >= a:
                Lte[a:b + 1, :] = ANOMALY
                n_anom += 1
        te_X.append(Xte); te_L.append(Lte)
        if Xtr is not None and len(Xtr) > 0:
            tr_X.append(Xtr)
            tr_L.append(np.zeros_like(Xtr, dtype=np.uint8))
        used.append(cid)

    if len(widths) > 1:
        raise ValueError(
            f"dims='all' mixes channels of different widths {sorted(widths)} — "
            "SMAP and MSL have 25 and 55 columns; select one spacecraft or use "
            "dims='first'")
    if not te_X:
        raise ValueError("no test arrays found under " + os.path.join(d, "test"))
    if not tr_X:
        raise ValueError("no train arrays found under " + os.path.join(d, "train"))

    train = _make_annotated_fleet(tr_X, tr_L)
    test = _make_annotated_fleet(te_X, te_L)
    pair = FleetPair(name=f"smapmsl:{sc}", train=train, test=test, cap=0.0,
                     channel_names=[f"dim{i}" for i in range(train.n_channels)],
                     meta={**base_meta, "channels": used,
                           "n_channels_used": len(used),
                           "anomaly_sequences": n_anom,
                           "train_segments": len(train), "test_segments": len(test)})
    if use_cache:
        _save_cache(cache, pair)
        pair.meta["cache"] = cache
    return pair
