"""
REAL turbofan data: NASA C-MAPSS and N-C-MAPSS, behind the same `Fleet`
interface the synthetic simulator uses.

This module exists to close the credibility gap named in the hand-off: every
number in the PoC so far comes from a generator we wrote, in which we also
chose the anomaly types.  Swapping the *background process* for real engine
data — while keeping the evaluation protocol identical — is what makes the
results about machines rather than about our simulator.

What is real and what is not, stated plainly (this belongs in any write-up):

  * The NORMAL data is real.  Sensor readings, cross-channel coupling,
    operating regimes, unit-to-unit variability, degradation dynamics: all of
    it comes from NASA's simulations of real engine models, not from us.
  * The RUL labels are real (C-MAPSS run-to-failure; N-C-MAPSS `Y`).
  * The INJECTED anomalies are still ours, because neither dataset ships
    anomaly annotations.  That is a deliberate trade: injection is the only way
    to get the per-channel GROUND TRUTH that the localisation claim — the
    project's actual contribution — is scored against.  The alternative
    (unlabelled real faults) can measure detection but not explanation.
  * RIGHT CENSORING is simulated (`censor_fleet`).  C-MAPSS train units all run
    to failure; a fleet in service does not.  Truncating real trajectories
    keeps the degradation dynamics real while restoring the censoring
    mechanism the survival experiment is about.
  * `organic` anomalies (late-life windows) are real degradation, never
    injected.

Layout expected on disk (nothing is downloaded automatically — NASA's
prognostics repository has no stable programmatic endpoint):

    data/cmapss/train_FD001.txt  test_FD001.txt  RUL_FD001.txt   (… FD002-4)
    data/ncmapss/N-CMAPSS_DS02-006.h5   (… any of the DS0x files)

`python -m poc.time_series.check_data` reports what is present and how to get
the rest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Fleet

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
CMAPSS_DIR = os.path.join(DATA_ROOT, "cmapss")
NCMAPSS_DIR = os.path.join(DATA_ROOT, "ncmapss")
CACHE_DIR = os.path.join(DATA_ROOT, "cache")

CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# regimes per subset, from the dataset description: FD001/FD003 are single
# condition, FD002/FD004 have six.
CMAPSS_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
CMAPSS_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# The nine N-C-MAPSS releases.  File names carry a suffix that differs per
# dataset, so lookup is by prefix.
NCMAPSS_DATASETS = ("DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07",
                    "DS08a", "DS08c", "DS08d")


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FleetPair:
    """
    A dataset as the pipeline consumes it: a fitting fleet and (when the
    benchmark defines one) an official held-out fleet.

    Keeping the two apart matters more here than in most benchmarks: windows
    from the same engine are strongly dependent, so a random window split leaks
    almost perfectly.  Every split in this pipeline is BY UNIT.
    """
    name: str
    train: Fleet
    test: Optional[Fleet] = None
    cap: float = 125.0
    channel_names: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        t = f", test={len(self.test)} units" if self.test is not None else ""
        return (f"FleetPair({self.name}: train={len(self.train)} units{t}, "
                f"C={self.train.n_channels}, regimes={self.train.n_regimes}, "
                f"cap={self.cap:g})")


def health_from_rul(rul: np.ndarray, cap: float) -> np.ndarray:
    """
    Health proxy in [0, 1] from remaining life, under the piecewise-linear RUL
    convention: 0 while the unit is more than `cap` cycles from failure, rising
    linearly to 1 at failure.

    The AD task labels windows by health (healthy below `healthy_frac`, organic
    anomaly above `organic_frac`), so this function is what turns real RUL
    labels into the same labelling the synthetic task uses — no new protocol,
    only a new background process.
    """
    return 1.0 - np.clip(np.asarray(rul, dtype=float) / max(cap, 1e-9), 0.0, 1.0)


def correlation_groups(X: np.ndarray, n_groups: int = 3) -> List[List[int]]:
    """
    Channel groups by correlation clustering — the real-data stand-in for the
    simulator's known groups.  Used only by the `channel_groups` structure
    option and by the report; nothing in the density depends on it.
    """
    C = X.shape[1]
    n_groups = max(1, min(int(n_groups), C))
    if n_groups == 1 or C < 3:
        return [list(range(C))]
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    try:
        from sklearn.cluster import AgglomerativeClustering
        lab = AgglomerativeClustering(
            n_clusters=n_groups, metric="precomputed", linkage="average"
        ).fit_predict(1.0 - np.abs(corr))
    except Exception:                                    # pragma: no cover
        lab = np.arange(C) % n_groups
    groups = [[int(c) for c in np.where(lab == g)[0]] for g in range(n_groups)]
    return [g for g in groups if g]


def censor_fleet(fleet: Fleet, frac: float, seed: int = 0,
                 lo: float = 0.45, hi: float = 0.90) -> Fleet:
    """
    Right-censor a fraction of units by truncating them somewhere in the second
    half of life, and mark them censored.

    On real run-to-failure data this is the only way to study the censoring
    question at all: NASA ran every training engine to failure, so the
    mechanism the survival experiment is about — units still alive when
    observation stops — is absent by construction.  The trajectories stay real;
    only the observation window is cut.  The correct likelihood contribution
    for such a unit is P(τ ≥ c), not P(τ = c), and that is the whole ablation.
    """
    if frac <= 0:
        return fleet
    rng = np.random.default_rng(seed)
    series, ruls, regimes, healths, censored = [], [], [], [], []
    for i in range(len(fleet)):
        x, r = fleet.series[i], fleet.rul[i]
        reg, h = fleet.regime[i], fleet.health[i]
        is_c = bool(rng.random() < frac) and len(x) > 12
        if is_c:
            cut = int(rng.integers(max(int(lo * len(x)), 4), max(int(hi * len(x)), 6)))
            x, r, reg, h = x[:cut], r[:cut], reg[:cut], h[:cut]
        series.append(x); ruls.append(r); regimes.append(reg)
        healths.append(h); censored.append(is_c or fleet.censored[i])
    return Fleet(series, ruls, regimes, healths, censored,
                 n_channels=fleet.n_channels, n_regimes=fleet.n_regimes,
                 channel_groups=fleet.channel_groups)


def subsample_units(fleet: Fleet, max_units: Optional[int], seed: int = 0) -> Fleet:
    """Keep at most `max_units` units (deterministically), for cheap smoke runs."""
    if not max_units or max_units >= len(fleet):
        return fleet
    idx = np.random.default_rng(seed).permutation(len(fleet))[:max_units]
    idx = sorted(int(i) for i in idx)
    return Fleet([fleet.series[i] for i in idx], [fleet.rul[i] for i in idx],
                 [fleet.regime[i] for i in idx], [fleet.health[i] for i in idx],
                 [fleet.censored[i] for i in idx],
                 n_channels=fleet.n_channels, n_regimes=fleet.n_regimes,
                 channel_groups=fleet.channel_groups)


def _regimes_from_settings(ops: np.ndarray, n_regimes: int, seed: int = 0) -> np.ndarray:
    """Operating regime id by k-means on the operational settings."""
    if n_regimes <= 1:
        return np.zeros(len(ops), dtype=int)
    from sklearn.cluster import KMeans
    n_regimes = min(n_regimes, len(np.unique(ops.round(2), axis=0)))
    if n_regimes <= 1:
        return np.zeros(len(ops), dtype=int)
    return KMeans(n_clusters=n_regimes, n_init=10,
                  random_state=seed).fit_predict(ops).astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# C-MAPSS
# ═══════════════════════════════════════════════════════════════════════════

def cmapss_available(subset: str = "FD001", data_dir: Optional[str] = None) -> bool:
    d = data_dir or CMAPSS_DIR
    return os.path.exists(os.path.join(d, f"train_{subset}.txt"))


def _cmapss_raw(path: str) -> np.ndarray:
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw


def load_cmapss(
    subset: str = "FD001",
    data_dir: Optional[str] = None,
    cap: float = 125.0,
    n_regimes: Optional[int] = None,
    n_groups: int = 3,
    max_units: Optional[int] = None,
    with_test: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    Load one C-MAPSS subset as a `FleetPair`.

    train_FD00x.txt  run-to-failure engines           -> fitting fleet
    test_FD00x.txt   engines truncated before failure -> held-out fleet
    RUL_FD00x.txt    true remaining life at the last  -> makes the held-out
                     recorded cycle of each test unit    fleet's labels exact

    This is the standard benchmark split (units, not windows), so RUL numbers
    are comparable to the prognostics literature.  Constant sensors are dropped
    using the TRAIN file's variance and the same mask is applied to test, so
    both fleets always have identical channel semantics.
    """
    d = data_dir or CMAPSS_DIR
    subset = subset.upper()
    if subset not in CMAPSS_SUBSETS:
        raise KeyError(f"unknown C-MAPSS subset {subset!r}; expected one of {CMAPSS_SUBSETS}")
    train_path = os.path.join(d, f"train_{subset}.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found.\n"
            "Download NASA's C-MAPSS 'CMaps' archive (Turbofan Engine Degradation\n"
            "Simulation Data Set, NASA Prognostics Data Repository) and place\n"
            f"train_/test_/RUL_{subset}.txt in {d}/ .  See data/README.md.")

    n_regimes = CMAPSS_REGIMES.get(subset, 1) if n_regimes is None else n_regimes
    tr = _cmapss_raw(train_path)
    ops_tr, sens_tr = tr[:, 2:5], tr[:, 5:]

    keep = sens_tr.std(0) > 1e-8                       # constant sensors carry nothing
    names = [CMAPSS_COLUMNS[5 + i] for i in range(sens_tr.shape[1]) if keep[i]]
    sens_tr = sens_tr[:, keep]

    reg_tr = _regimes_from_settings(ops_tr, n_regimes, seed=seed)
    groups = correlation_groups(sens_tr, n_groups)

    def build(raw: np.ndarray, reg: np.ndarray, sens: np.ndarray,
              rul_end: Optional[np.ndarray]) -> Fleet:
        units = raw[:, 0].astype(int)
        series, ruls, regimes, healths = [], [], [], []
        for k, u in enumerate(np.unique(units)):
            sel = units == u
            x = sens[sel].astype(np.float32)
            T = len(x)
            tail = 0.0 if rul_end is None else float(rul_end[k])
            rul = np.arange(T - 1, -1, -1, dtype=float) + tail
            series.append(x)
            ruls.append(rul)
            regimes.append(reg[sel].astype(int))
            healths.append(health_from_rul(rul, cap))
        return Fleet(series, ruls, regimes, healths, [False] * len(series),
                     n_channels=sens.shape[1], n_regimes=int(max(reg) + 1),
                     channel_groups=groups)

    train = build(tr, reg_tr, sens_tr, None)
    train = subsample_units(train, max_units, seed)

    test = None
    test_path = os.path.join(d, f"test_{subset}.txt")
    rul_path = os.path.join(d, f"RUL_{subset}.txt")
    if with_test and os.path.exists(test_path) and os.path.exists(rul_path):
        te = _cmapss_raw(test_path)
        sens_te = te[:, 5:][:, keep]
        reg_te = _regimes_from_settings(te[:, 2:5], n_regimes, seed=seed)
        rul_end = np.loadtxt(rul_path).reshape(-1)
        test = build(te, reg_te, sens_te, rul_end)
        test = subsample_units(test, max_units, seed + 1)

    return FleetPair(
        name=f"cmapss:{subset}", train=train, test=test, cap=cap,
        channel_names=names,
        meta={"subset": subset, "n_regimes": n_regimes, "cap": cap,
              "dropped_constant_sensors": int((~keep).sum()),
              "train_units": len(train),
              "test_units": len(test) if test is not None else 0,
              "source": "NASA C-MAPSS (real)"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# N-C-MAPSS
# ═══════════════════════════════════════════════════════════════════════════

def ncmapss_file(dataset: str, data_dir: Optional[str] = None) -> Optional[str]:
    """Resolve `DS02` (say) to the actual N-CMAPSS_DS02-006.h5 on disk."""
    d = data_dir or NCMAPSS_DIR
    if not os.path.isdir(d):
        return None
    want = dataset.upper().replace("N-CMAPSS_", "")
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".h5"):
            continue
        stem = f[:-3].upper().replace("N-CMAPSS_", "")
        if stem == want or stem.split("-")[0] == want:
            return os.path.join(d, f)
    return None


def ncmapss_available(dataset: str = "DS02", data_dir: Optional[str] = None) -> bool:
    return ncmapss_file(dataset, data_dir) is not None


def _decode_names(arr, n: int, prefix: str) -> List[str]:
    """N-CMAPSS stores variable names as an array of byte strings."""
    try:
        flat = np.asarray(arr).reshape(-1)
        out = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in flat]
        out = [s.strip() for s in out if str(s).strip()]
        if len(out) == n:
            return out
    except Exception:
        pass
    return [f"{prefix}{i}" for i in range(n)]


def load_ncmapss(
    dataset: str = "DS02",
    data_dir: Optional[str] = None,
    channels: Sequence[str] = ("X_s",),
    aggregate: str = "cycle",
    subsample: int = 10,
    cap: float = 65.0,
    n_regimes: int = 3,
    n_groups: int = 3,
    max_units: Optional[int] = None,
    max_rows: Optional[int] = None,
    use_cache: bool = True,
    seed: int = 0,
) -> FleetPair:
    """
    Load one N-C-MAPSS release (Arias Chao et al., *Data* 2021) as a FleetPair.

    N-C-MAPSS is real flight-condition data at 1 Hz, so a single release is
    millions of rows — three orders of magnitude more than C-MAPSS.  Two
    reductions are offered, and the choice is a modelling decision, not a
    detail:

      aggregate="cycle"  (default)  one row per (unit, flight cycle): the
          per-cycle mean of every channel.  The resulting series has the same
          shape and semantics as C-MAPSS (one row per flight, degradation
          across rows), so windows mean the same thing across all three data
          sources and results are directly comparable.  Also what fits
          comfortably in memory.
      aggregate="raw"               keep the within-flight time base, taking
          every `subsample`-th row.  Windows then span seconds of one flight,
          which is a different (much shorter-horizon) question.

    Channels: `X_s` are the 14 physical sensor measurements; add "W" for the 4
    flight-condition inputs (altitude, Mach, TRA, T2) and "X_v"/"T" for the
    virtual sensors / health parameters — the latter two are model internals
    that a real engine does not report, so they are OFF by default.  Turning
    them on inflates every score and must be reported if used.

    Results are cached as .npz under data/cache/ ; the HDF5 parse is the slow
    part and a batch re-reads the same release dozens of times.
    """
    path = ncmapss_file(dataset, data_dir)
    if path is None:
        raise FileNotFoundError(
            f"No N-CMAPSS file for {dataset!r} in {data_dir or NCMAPSS_DIR}.\n"
            "Download the N-CMAPSS turbofan dataset (NASA Prognostics Data\n"
            "Repository, 'Turbofan Engine Degradation Simulation Data Set 2')\n"
            "and place the .h5 files there.  See data/README.md.")

    key = (f"ncmapss_{dataset}_{'-'.join(channels)}_{aggregate}_{subsample}_"
           f"{n_regimes}_{max_units or 0}_{max_rows or 0}")
    cache = os.path.join(CACHE_DIR, key + ".npz")
    if use_cache and os.path.exists(cache):
        pair = _load_cached_pair(cache, cap=cap, n_groups=n_groups, name=f"ncmapss:{dataset}")
        pair.meta["cache"] = cache
        return pair

    try:
        import h5py
    except ImportError as exc:                          # pragma: no cover
        raise ImportError(
            "N-C-MAPSS is stored as HDF5; install h5py (`pip install h5py`) "
            "to use it.  C-MAPSS and the synthetic fleet need no extra deps."
        ) from exc

    blocks: Dict[str, Dict[str, np.ndarray]] = {"dev": {}, "test": {}}
    names: List[str] = []
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        for split in ("dev", "test"):
            mats, cols = [], []
            for grp in channels:
                k = f"{grp}_{split}"
                if k not in keys:
                    continue
                arr = np.asarray(f[k], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                mats.append(arr)
                cols += _decode_names(f.get(f"{grp}_var"), arr.shape[1], f"{grp}_")
            if not mats:
                continue
            X = np.concatenate(mats, axis=1)
            A = np.asarray(f[f"A_{split}"], dtype=np.float32)
            a_names = _decode_names(f.get("A_var"), A.shape[1], "a")
            Y = np.asarray(f[f"Y_{split}"], dtype=np.float32).reshape(-1)
            W = (np.asarray(f[f"W_{split}"], dtype=np.float32)
                 if f"W_{split}" in keys else None)
            blocks[split] = {"X": X, "A": A, "Y": Y, "W": W, "a_names": a_names}
            names = cols

    if not blocks["dev"]:
        raise KeyError(f"{path} has no *_dev arrays — unexpected N-CMAPSS layout")

    def to_fleet(b: Dict[str, np.ndarray], sd: int) -> Optional[Fleet]:
        if not b:
            return None
        X, A, Y = b["X"], b["A"], b["Y"]
        a_names = list(b["a_names"])
        u_col = a_names.index("unit") if "unit" in a_names else 0
        c_col = a_names.index("cycle") if "cycle" in a_names else 1
        units, cycles = A[:, u_col].astype(int), A[:, c_col].astype(int)

        if max_rows and len(X) > max_rows:              # deterministic head-cut
            X, Y, units, cycles = X[:max_rows], Y[:max_rows], units[:max_rows], cycles[:max_rows]
            if b["W"] is not None:
                b["W"] = b["W"][:max_rows]

        W = b["W"]
        reg_all = (_regimes_from_settings(W, n_regimes, seed=sd)
                   if W is not None else np.zeros(len(X), dtype=int))

        series, ruls, regimes, healths = [], [], [], []
        for u in np.unique(units):
            sel = units == u
            xu, yu, cu, ru = X[sel], Y[sel], cycles[sel], reg_all[sel]
            if aggregate == "cycle":
                order = np.unique(cu)
                xs = np.stack([xu[cu == c].mean(0) for c in order]).astype(np.float32)
                ys = np.array([yu[cu == c].mean() for c in order], dtype=float)
                rs = np.array([np.bincount(ru[cu == c]).argmax() for c in order], dtype=int)
            else:
                step = max(int(subsample), 1)
                xs, ys, rs = xu[::step], yu[::step].astype(float), ru[::step]
            if len(xs) < 4:
                continue
            series.append(xs)
            ruls.append(ys)
            regimes.append(rs)
            healths.append(health_from_rul(ys, cap))
        if not series:
            return None
        return Fleet(series, ruls, regimes, healths, [False] * len(series),
                     n_channels=series[0].shape[1],
                     n_regimes=int(max(int(r.max()) for r in regimes) + 1),
                     channel_groups=[list(range(series[0].shape[1]))])

    train = to_fleet(blocks["dev"], seed)
    test = to_fleet(blocks["test"], seed + 1)
    train = subsample_units(train, max_units, seed)
    if test is not None:
        test = subsample_units(test, max_units, seed + 1)

    groups = correlation_groups(np.concatenate(train.series[: min(len(train), 20)]),
                                n_groups)
    train.channel_groups = groups
    if test is not None:
        test.channel_groups = groups

    pair = FleetPair(name=f"ncmapss:{dataset}", train=train, test=test, cap=cap,
                     channel_names=names,
                     meta={"dataset": dataset, "file": os.path.basename(path),
                           "aggregate": aggregate, "channels": list(channels),
                           "n_regimes": n_regimes, "cap": cap,
                           "train_units": len(train),
                           "test_units": len(test) if test else 0,
                           "source": "NASA N-C-MAPSS (real)"})
    if use_cache:
        _save_cached_pair(cache, pair)
        pair.meta["cache"] = cache
    return pair


# ── fleet cache (npz; ragged series stored flat with offsets) ───────────────

def _pack(fleet: Fleet, tag: str) -> Dict[str, np.ndarray]:
    lens = np.array([len(s) for s in fleet.series], dtype=np.int64)
    return {
        f"{tag}_X": np.concatenate(fleet.series, axis=0).astype(np.float32),
        f"{tag}_rul": np.concatenate(fleet.rul).astype(np.float32),
        f"{tag}_reg": np.concatenate(fleet.regime).astype(np.int32),
        f"{tag}_health": np.concatenate(fleet.health).astype(np.float32),
        f"{tag}_lens": lens,
        f"{tag}_cens": np.array(fleet.censored, dtype=bool),
        f"{tag}_nreg": np.array([fleet.n_regimes]),
    }


def _unpack(z, tag: str) -> Optional[Fleet]:
    if f"{tag}_lens" not in z:
        return None
    lens = z[f"{tag}_lens"]
    bounds = np.concatenate([[0], np.cumsum(lens)])
    cut = lambda a: [a[bounds[i]:bounds[i + 1]] for i in range(len(lens))]
    X = cut(z[f"{tag}_X"])
    return Fleet(X, cut(z[f"{tag}_rul"]), cut(z[f"{tag}_reg"]), cut(z[f"{tag}_health"]),
                 list(z[f"{tag}_cens"]), n_channels=X[0].shape[1],
                 n_regimes=int(z[f"{tag}_nreg"][0]), channel_groups=[])


def _save_cached_pair(path: str, pair: FleetPair) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = _pack(pair.train, "train")
    if pair.test is not None:
        blob.update(_pack(pair.test, "test"))
    np.savez_compressed(path, **blob)


def _load_cached_pair(path: str, cap: float, n_groups: int, name: str) -> FleetPair:
    z = np.load(path, allow_pickle=False)
    train = _unpack(z, "train")
    test = _unpack(z, "test")
    groups = correlation_groups(np.concatenate(train.series[: min(len(train), 20)]),
                               n_groups)
    train.channel_groups = groups
    if test is not None:
        test.channel_groups = groups
    return FleetPair(name=name, train=train, test=test, cap=cap,
                     meta={"source": "cache", "train_units": len(train),
                           "test_units": len(test) if test else 0})
