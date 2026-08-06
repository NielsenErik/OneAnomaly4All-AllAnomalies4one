"""
The dataset catalogue: one table, every source the time-series pipeline can
run on, with the loader, the availability probe, the tasks it supports, the
download instructions and the caveats attached to each.

Why a table and not a chain of `if source == ...`
─────────────────────────────────────────────────
There are now twelve sources across three families (a simulator, run-to-failure
prognostics, annotated spacecraft telemetry).  Every place that used to branch
on the source name — `load_fleets`, `dataset_available`, `dataset_id`,
`check_data`, the config validator — reads this table instead, so adding a
thirteenth source is one entry, not five edits that can disagree with each
other.

Two fields carry weight beyond bookkeeping:

  `tasks`     which stages the source can honestly support.  Annotated
              telemetry has no remaining useful life, so `rul` is not in its
              tuple and the runner refuses the combination up front instead of
              training a model against a column of zeros.
  `caveats`   the known defects of the source, in the source's own entry.
              They are copied into `pair.meta` by the loader and printed by the
              pipeline at load time, so a result on SMAP/MSL cannot be produced
              without its triviality warning appearing in the same log.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import data_prognostics as prog
from . import data_space as space
from .data_real import (
    CMAPSS_DIR,
    NCMAPSS_DIR,
    FleetPair,
    cmapss_available,
    load_cmapss,
    load_ncmapss,
    ncmapss_available,
)


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Source:
    name: str                                     # `dataset.source` in a config
    kind: str                                     # "synthetic" | "fleet" | "telemetry"
    title: str
    tasks: Tuple[str, ...]                        # stages this source supports
    loader: Callable[[Dict[str, Any], int], FleetPair]
    probe: Callable[[Dict[str, Any]], bool]
    ident: Callable[[Dict[str, Any]], str]
    root: str                                     # where the files are expected
    howto: str                                    # how to get them
    caveats: Tuple[str, ...] = ()

    @property
    def annotated(self) -> bool:
        """True when the source ships REAL anomaly labels (no injection needed)."""
        return self.kind == "telemetry"


def _s(spec: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = spec.get(key, default)
    return default if v is None else v


def _cap(spec: Dict[str, Any], default: float) -> float:
    v = spec.get("cap")
    return float(v) if v is not None else float(default)


# ═══════════════════════════════════════════════════════════════════════════
# Loaders — config block in, FleetPair out
# ═══════════════════════════════════════════════════════════════════════════

def _load_synthetic(spec: Dict[str, Any], seed: int) -> FleetPair:
    from .data import simulate_fleet
    fleet = simulate_fleet(
        n_units=int(_s(spec, "units", 60)),
        n_channels=int(_s(spec, "channels", 14)),
        n_regimes=int(_s(spec, "regimes", 3)),
        n_groups=int(_s(spec, "groups", 3)),
        min_life=int(_s(spec, "min_life", 120)),
        max_life=int(_s(spec, "max_life", 300)),
        censor_frac=float(_s(spec, "censor_frac", 0.0)),
        dead_channels=int(_s(spec, "dead_channels", 2)),
        noise=float(_s(spec, "noise", 0.10)),
        group_noise=float(_s(spec, "group_noise", 0.25)),
        phi_ar=float(_s(spec, "phi_ar", 0.85)),
        seed=seed,
    )
    return FleetPair(name="synthetic", train=fleet, test=None,
                     cap=_cap(spec, 130.0),
                     meta={"source": "synthetic simulator", "seed": seed,
                           "units": len(fleet)})


def _load_cmapss(spec: Dict[str, Any], seed: int) -> FleetPair:
    return load_cmapss(
        subset=str(_s(spec, "subset", "FD001")),
        data_dir=spec.get("data_dir"),
        cap=_cap(spec, 125.0),
        # NOT `spec["regimes"]`: that key is the SYNTHETIC generator's knob and
        # defaults to 3, which silently overrode the authoritative per-subset
        # table.  None => use the table; `cmapss_regimes` overrides.
        n_regimes=spec.get("cmapss_regimes"),
        n_groups=int(_s(spec, "groups", 3)),
        max_units=spec.get("max_units"),
        with_test=bool(_s(spec, "official_test", True)),
        seed=seed,
    )


def _load_ncmapss(spec: Dict[str, Any], seed: int) -> FleetPair:
    return load_ncmapss(
        dataset=str(_s(spec, "dataset", "DS02")),
        data_dir=spec.get("data_dir"),
        channels=tuple(_s(spec, "channels_groups", ("X_s",))),
        aggregate=str(_s(spec, "aggregate", "cycle")),
        subsample=int(_s(spec, "subsample", 10)),
        cap=_cap(spec, 65.0),
        n_regimes=int(_s(spec, "regimes", 3)),
        n_groups=int(_s(spec, "groups", 3)),
        max_units=spec.get("max_units"),
        max_rows=spec.get("max_rows"),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_phm08(spec: Dict[str, Any], seed: int) -> FleetPair:
    return prog.load_phm08(
        data_dir=spec.get("data_dir"),
        cap=_cap(spec, 125.0),
        n_regimes=int(_s(spec, "phm08_regimes", 6)),
        n_groups=int(_s(spec, "groups", 3)),
        max_units=spec.get("max_units"),
        seed=seed,
    )


def _load_battery(spec: Dict[str, Any], seed: int) -> FleetPair:
    return prog.load_nasa_battery(
        data_dir=spec.get("data_dir"),
        cells=spec.get("cells"),
        eol_frac=float(_s(spec, "eol_frac", 0.7)),
        rated_ah=spec.get("rated_ah"),
        cap=_cap(spec, 60.0),
        n_groups=int(_s(spec, "groups", 2)),
        min_cycles=int(_s(spec, "min_cycles", 20)),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_calce(spec: Dict[str, Any], seed: int) -> FleetPair:
    return prog.load_calce(
        data_dir=spec.get("data_dir"),
        cells=spec.get("cells"),
        eol_frac=float(_s(spec, "eol_frac", 0.7)),
        rated_ah=float(_s(spec, "rated_ah", 1.1)),
        cap=_cap(spec, 200.0),
        n_groups=int(_s(spec, "groups", 2)),
        min_cycles=int(_s(spec, "min_cycles", 20)),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_ims(spec: Dict[str, Any], seed: int) -> FleetPair:
    return prog.load_ims(
        test=str(_s(spec, "ims_test", "2nd_test")),
        data_dir=spec.get("data_dir"),
        unit=str(_s(spec, "ims_unit", "bearing")),
        stride=int(_s(spec, "ims_file_stride", 1)),
        max_files=spec.get("max_files"),
        cap=_cap(spec, 200.0),
        n_groups=int(_s(spec, "groups", 3)),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_milling(spec: Dict[str, Any], seed: int) -> FleetPair:
    return prog.load_milling(
        data_dir=spec.get("data_dir"),
        vb_eol=float(_s(spec, "vb_eol", 0.6)),
        cap=_cap(spec, 20.0),
        n_groups=int(_s(spec, "groups", 3)),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_pcoe(spec: Dict[str, Any], seed: int) -> FleetPair:
    return prog.load_pcoe_csv(
        preset=str(_s(spec, "preset", "igbt")),
        data_dir=spec.get("data_dir"),
        unit_glob=str(_s(spec, "unit_glob", "*.csv")),
        time_col=spec.get("time_col"),
        sensor_cols=spec.get("sensor_cols"),
        regime_col=spec.get("regime_col"),
        eol_column=spec.get("eol_column"),
        eol_threshold=spec.get("eol_threshold"),
        cap=_cap(spec, 100.0),
        n_groups=int(_s(spec, "groups", 3)),
        min_length=int(_s(spec, "min_length", 12)),
        seed=seed,
    )


def _load_esa(spec: Dict[str, Any], seed: int) -> FleetPair:
    return space.load_esa_adb(
        mission=str(_s(spec, "mission", "Mission1")),
        data_dir=spec.get("data_dir"),
        channels=_s(spec, "esa_channels", "lightweight"),
        form=str(_s(spec, "esa_form", "auto")),
        split=spec.get("esa_split"),
        resample_s=spec.get("resample_s"),
        include_telecommands=bool(_s(spec, "include_telecommands", False)),
        rare_events=str(_s(spec, "rare_events", "anomaly")),
        max_train_samples=spec.get("max_train_samples", 400_000),
        max_test_samples=spec.get("max_test_samples", 200_000),
        n_groups=int(_s(spec, "groups", 3)),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_opssat(spec: Dict[str, Any], seed: int) -> FleetPair:
    return space.load_opssat(
        data_dir=spec.get("data_dir"),
        min_length=int(_s(spec, "min_length", 16)),
        max_length=spec.get("max_length", 2000),
        channels=spec.get("opssat_channels"),
        train_frac=float(_s(spec, "train_units", 0.7)),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


def _load_smapmsl(spec: Dict[str, Any], seed: int) -> FleetPair:
    return space.load_smap_msl(
        spacecraft=str(_s(spec, "spacecraft", "SMAP")),
        data_dir=spec.get("data_dir"),
        dims=str(_s(spec, "dims", "first")),
        channels=spec.get("smap_channels"),
        use_cache=bool(_s(spec, "cache", True)),
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# The catalogue
# ═══════════════════════════════════════════════════════════════════════════

SOURCES: Dict[str, Source] = {}


def _register(src: Source) -> Source:
    SOURCES[src.name] = src
    return src


_register(Source(
    name="synthetic", kind="synthetic", title="C-MAPSS-shaped simulator",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_synthetic, probe=lambda spec: True,
    ident=lambda spec: "synthetic", root="(none)",
    howto="Always available; needs no files.",
    caveats=("Evidence about machinery, not about machines: the background "
             "process, the degradation and the anomalies are all ours.",),
))

_register(Source(
    name="cmapss", kind="fleet", title="NASA C-MAPSS turbofan (FD001-FD004)",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_cmapss,
    probe=lambda spec: cmapss_available(str(_s(spec, "subset", "FD001")).upper(),
                                        spec.get("data_dir")),
    ident=lambda spec: f"cmapss:{str(_s(spec, 'subset', 'FD001')).upper()}",
    root=CMAPSS_DIR,
    howto="NASA PCoE 'Turbofan Engine Degradation Simulation Data Set' (CMaps "
          "archive, ~13 MB), also on Kaggle. Place train_/test_/RUL_FD00x.txt.",
    caveats=("Ships no anomaly annotations: detection anomalies are injected.",),
))

_register(Source(
    name="ncmapss", kind="fleet", title="NASA N-C-MAPSS (real flight conditions)",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_ncmapss,
    probe=lambda spec: ncmapss_available(str(_s(spec, "dataset", "DS02")).upper(),
                                         spec.get("data_dir")),
    ident=lambda spec: f"ncmapss:{str(_s(spec, 'dataset', 'DS02')).upper()}",
    root=NCMAPSS_DIR,
    howto="NASA PCoE 'Turbofan Engine Degradation Simulation Data Set 2' "
          "(Arias Chao et al. 2021), HDF5, 1-5 GB per release. Needs h5py.",
    caveats=("X_v / T channels are model internals a real engine does not "
             "report; they are off by default and inflate every score if on.",),
))

_register(Source(
    name="phm08", kind="fleet", title="PHM08 Prognostics Challenge",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_phm08,
    probe=lambda spec: prog.phm08_available(spec.get("data_dir")),
    ident=lambda spec: "phm08", root=prog.PHM08_DIR,
    howto="PHM Society data repository (data.phmsociety.org) / NASA PCoE "
          "mirror. Place train.txt (26 columns, C-MAPSS layout).",
    caveats=("True RUL for the challenge test files was never released; only "
             "train.txt is loaded and the split is by unit.",),
))

_register(Source(
    name="battery", kind="fleet", title="NASA Li-ion battery aging (Saha & Goebel)",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_battery,
    probe=lambda spec: prog.battery_available(spec.get("data_dir")),
    ident=lambda spec: "battery", root=prog.BATTERY_DIR,
    howto="NASA PCoE 'Battery Data Set' (data.nasa.gov / data.phmsociety.org/"
          "nasa). Place B0005.mat, B0006.mat, ... Needs scipy.",
    caveats=("One row per DISCHARGE cycle: charge and impedance operations are "
             "skipped, so a 'timestep' is a cycle, not a second.",
             "Cells differ in ambient temperature and load profile (some "
             "randomised-use, some at 42 C); that heterogeneity is left in."),
))

_register(Source(
    name="calce", kind="fleet", title="CALCE CS2 prismatic cells",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_calce,
    probe=lambda spec: prog.calce_available(spec.get("data_dir")),
    ident=lambda spec: "calce", root=prog.CALCE_DIR,
    howto="CALCE battery data (calce.umd.edu). Place per-cell CSV summaries "
          "(cycle, capacity, ...) or the raw CS2_xx/ Arbin exports.",
    caveats=("Raw .xlsx parsing needs openpyxl and is slow; a per-cycle CSV "
             "summary is the supported path.",),
))

_register(Source(
    name="ims", kind="fleet", title="IMS / Rexnord bearing run-to-failure",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_ims,
    probe=lambda spec: prog.ims_available(spec.get("data_dir")),
    ident=lambda spec: f"ims:{str(_s(spec, 'ims_test', '2nd_test'))}",
    root=prog.IMS_DIR,
    howto="NASA PCoE 'Bearing Data Set' (IMS, University of Cincinnati). "
          "Unpack 1st_test/, 2nd_test/, 3rd_test/ of ASCII snapshot files.",
    caveats=("Timesteps are 10-minute snapshots reduced to vibration features; "
             "the raw 20 kHz waveform is not modelled directly.",
             "Only the documented failed bearings are uncensored; the others "
             "are genuinely right-censored survivors."),
))

_register(Source(
    name="milling", kind="fleet", title="NASA milling (tool flank wear)",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_milling,
    probe=lambda spec: prog.milling_available(spec.get("data_dir")),
    ident=lambda spec: "milling", root=prog.MILLING_DIR,
    howto="NASA PCoE 'Milling Data Set'. Place mill.mat. Needs scipy.",
    caveats=("VB (flank wear) is missing for some runs and is interpolated — "
             "an interpolated EOL label is a weaker label.",
             "16 cases only: the fleet is small, so per-unit splits are coarse."),
))

_register(Source(
    name="pcoe", kind="fleet", title="Generic PCoE per-unit CSV adapter",
    tasks=("ad", "explain", "rul", "calibration", "scaling"),
    loader=_load_pcoe,
    probe=lambda spec: prog.pcoe_available(str(_s(spec, "preset", "igbt")),
                                           spec.get("data_dir")),
    ident=lambda spec: f"pcoe:{str(_s(spec, 'preset', 'igbt'))}",
    root=prog.PCOE_DIR,
    howto="For the PCoE sets without a stable layout (IGBT, capacitor, "
          "fatigue/Lamb-wave): convert each unit to data/pcoe/<preset>/"
          "<unit>.csv and name the columns in the config.",
    caveats=("The layout is user-supplied — the loader trusts the column names "
             "in the config and cannot detect a wrong one.",),
))

_register(Source(
    name="esa", kind="telemetry", title="ESA Anomalies Dataset / ESA-ADB",
    tasks=("ad", "explain"),
    loader=_load_esa,
    probe=lambda spec: space.esa_available(str(_s(spec, "mission", "Mission1")),
                                           spec.get("data_dir")),
    ident=lambda spec: f"esa:{str(_s(spec, 'mission', 'Mission1'))}",
    root=space.ESA_DIR,
    howto="Zenodo 10.5281/zenodo.12528696 (~31 GB, CC BY 3.0 IGO). Unpack "
          "ESA-Mission1/ and ESA-Mission2/ into data/esa_adb/, or place the "
          "TimeEval-preprocessed CSVs under data/esa_adb/preprocessed/.",
    caveats=space.ESA_CAVEATS,
))

_register(Source(
    name="opssat", kind="telemetry", title="OPSSAT-AD (ESA CubeSat)",
    tasks=("ad", "explain"),
    loader=_load_opssat,
    probe=lambda spec: space.opssat_available(spec.get("data_dir")),
    ident=lambda spec: "opssat", root=space.OPSSAT_DIR,
    howto="Zenodo 10.5281/zenodo.12588359 (also on Kaggle). Place "
          "segments.csv in data/opssat/.",
    caveats=space.OPSSAT_CAVEATS,
))

_register(Source(
    name="smapmsl", kind="telemetry", title="SMAP / MSL (Hundman et al. 2018)",
    tasks=("ad", "explain"),
    loader=_load_smapmsl,
    probe=lambda spec: space.smap_msl_available(spec.get("data_dir")),
    ident=lambda spec: f"smapmsl:{str(_s(spec, 'spacecraft', 'SMAP')).upper()}",
    root=space.SMAP_MSL_DIR,
    howto="khundman/telemanom `data.zip`. Unpack train/, test/ and "
          "labeled_anomalies.csv into data/smap_msl/.",
    caveats=space.SMAP_MSL_CAVEATS,
))


# ═══════════════════════════════════════════════════════════════════════════
# Lookup helpers
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_NAMES: Tuple[str, ...] = tuple(SOURCES)
ANNOTATED_SOURCES: Tuple[str, ...] = tuple(n for n, s in SOURCES.items() if s.annotated)


def get_source(spec_or_name) -> Source:
    """Accepts a config `dataset` block or a bare source name."""
    name = (spec_or_name if isinstance(spec_or_name, str)
            else str(spec_or_name.get("source", "synthetic"))).lower()
    if name not in SOURCES:
        raise KeyError(f"unknown dataset source {name!r}; expected one of "
                       f"{sorted(SOURCES)}")
    return SOURCES[name]


def supports(spec_or_name, stage: str) -> bool:
    return stage in get_source(spec_or_name).tasks
