"""
Catalogue tests: every dataset source, exercised against FABRICATED files
written in the documented on-disk format.

Why fabricate.  None of these datasets can be downloaded in CI (31 GB for
ESA-ADB alone), and a loader that is only ever run by hand on one machine is a
loader whose failure mode is a silent mis-parse three months later — this
project's recurring bug shape.  So each test writes a miniature dataset in the
format `data/README.md` promises, loads it through the SAME entry point the
runner uses (`load_fleets`), and asserts the thing the pipeline depends on.

What is therefore NOT tested: that the real files match the documented format.
That is what `python -m poc.time_series.check_data --load` is for, and it is
the first thing to run after downloading anything.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from poc.time_series.catalog import SOURCES, get_source, supports
from poc.time_series.data_space import ANOMALY, NOMINAL, RARE_EVENT
from poc.time_series.datasets import (
    build_ad_task,
    build_rul_task,
    dataset_available,
    dataset_id,
    load_fleets,
)

pd = pytest.importorskip("pandas")


# ═══════════════════════════════════════════════════════════════════════════
# The catalogue itself
# ═══════════════════════════════════════════════════════════════════════════

def test_every_source_is_well_formed():
    for name, src in SOURCES.items():
        assert src.name == name
        assert src.kind in ("synthetic", "fleet", "telemetry")
        assert src.tasks, f"{name} supports no stage"
        assert "ad" in src.tasks
        assert src.howto and src.root
        # A source whose data is real must say what is wrong with it.  The
        # simulator is exempt only because its single caveat is that it is a
        # simulator, which it also states.
        assert src.caveats, f"{name} declares no caveats"


def test_annotated_sources_refuse_rul():
    """Zeros are not labels: `rul` must be refused, not run, on telemetry."""
    for name, src in SOURCES.items():
        if src.kind == "telemetry":
            assert not supports(name, "rul")
        else:
            assert supports(name, "rul")


def test_unknown_source_is_rejected():
    ok, why = dataset_available({"source": "nope"})
    assert not ok and "nope" in why
    with pytest.raises(KeyError):
        get_source({"source": "nope"})


def test_config_validate_rejects_rul_on_telemetry(tmp_path):
    from poc.time_series.config import load_config
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nstages: [ad, rul]\ndataset: {source: opssat}\n")
    with pytest.raises(ValueError, match="does not support stage"):
        load_config(str(p))


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — miniature datasets in the documented formats
# ═══════════════════════════════════════════════════════════════════════════

def _wave(n, C, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, n)[:, None]
    base = np.sin(t + np.arange(C)[None, :]) + rng.normal(0, 0.05, (n, C))
    return (base + drift * np.linspace(0, 1, n)[:, None]).astype(np.float32)


@pytest.fixture
def esa_preprocessed(tmp_path):
    """The TimeEval-preprocessed form: one train.csv and one test.csv."""
    root = tmp_path / "esa_adb" / "preprocessed" / "multivariate" / "ESA-Mission1-semisupervised"
    root.mkdir(parents=True)
    chans = [f"channel_{i}" for i in range(41, 45)]

    def write(path, n, anomalies, seed):
        X = _wave(n, len(chans), seed=seed)
        cols = {"timestamp": pd.date_range("2000-01-01", periods=n, freq="30s")}
        for j, c in enumerate(chans):
            cols[c] = X[:, j]
        L = np.zeros((n, len(chans)), dtype=np.uint8)
        for (a, b, ch, code) in anomalies:
            L[a:b, ch] = code
        for j, c in enumerate(chans):
            cols[f"is_anomaly_{c}"] = L[:, j]
        pd.DataFrame(cols).to_csv(path, index=False)

    # different seeds per half: the two halves of a mission are different
    # stretches of time, and identical halves would make the leak test vacuous
    write(root / "84_months.train.csv", 600, [(100, 110, 0, ANOMALY)], seed=1)
    write(root / "84_months.test.csv", 600,
          [(200, 260, 1, ANOMALY), (400, 430, 2, RARE_EVENT)], seed=2)
    return str(tmp_path / "esa_adb")


@pytest.fixture
def opssat_csv(tmp_path):
    d = tmp_path / "opssat"
    d.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for seg in range(24):
        n = int(rng.integers(40, 80))
        anom = int(seg % 4 == 0)
        ch = f"CADC000{seg % 3}"
        v = rng.normal(0, 1, n) + (5.0 if anom else 0.0)
        for k in range(n):
            rows.append({"timestamp": k, "channel": ch, "value": float(v[k]),
                         "sampling": 1.0, "anomaly": anom,
                         "train": int(seg % 3 != 0), "segment": seg})
    pd.DataFrame(rows).to_csv(d / "segments.csv", index=False)
    return str(d)


@pytest.fixture
def smap_msl_dir(tmp_path):
    d = tmp_path / "smap_msl"
    (d / "train").mkdir(parents=True)
    (d / "test").mkdir()
    rows = ["chan_id,spacecraft,anomaly_sequences,class,num_values"]
    for i, cid in enumerate(["A-1", "A-2", "P-1"]):
        np.save(d / "train" / f"{cid}.npy", _wave(300, 25, seed=i))
        np.save(d / "test" / f"{cid}.npy", _wave(260, 25, seed=i + 10))
        rows.append(f'{cid},SMAP,"[[120, 180]]",[point],260')
    (d / "labeled_anomalies.csv").write_text("\n".join(rows) + "\n")
    return str(d)


@pytest.fixture
def phm08_dir(tmp_path):
    d = tmp_path / "phm08"
    d.mkdir()
    rng = np.random.default_rng(0)
    lines = []
    for u in range(1, 13):
        # 150-250 cycles: real PHM08 units run 128-357, and a fleet shorter
        # than `cap` (125) has no window the health proxy calls healthy
        T = int(rng.integers(150, 250))
        for t in range(1, T + 1):
            ops = rng.normal(0, 1, 3)
            sens = np.concatenate([rng.normal(0, 1, 20) + t / T, [1.0]])  # 1 constant
            lines.append(" ".join(f"{v:.4f}" for v in [u, t, *ops, *sens]))
    (d / "train.txt").write_text("\n".join(lines) + "\n")
    return str(d)


@pytest.fixture
def battery_dir(tmp_path):
    """A .mat in the PCoE battery layout, written with scipy."""
    sio = pytest.importorskip("scipy.io")
    d = tmp_path / "battery_nasa"
    d.mkdir()
    # two cells reach the 30%-fade EOL, two survive the record (censored) —
    # a fleet of two would leave the RUL split's held-out side empty
    for cell, n_cycles, fade in (("B0005", 60, 0.45), ("B0006", 40, 0.15),
                                 ("B0007", 55, 0.50), ("B0018", 45, 0.12)):
        cycles = []
        for k in range(n_cycles):
            cap = 2.0 * (1.0 - fade * k / n_cycles)
            t = np.linspace(0, 3000, 50)
            data = {"Voltage_measured": np.linspace(4.2, 3.0, 50),
                    "Current_measured": np.full(50, -2.0),
                    "Temperature_measured": np.linspace(24, 38, 50),
                    "Time": t, "Capacity": np.array([[cap]])}
            cycles.append(("discharge", 24.0, data))
            cycles.append(("charge", 24.0, {"Time": t}))
        arr = np.zeros((1, len(cycles)),
                       dtype=[("type", "O"), ("ambient_temperature", "O"),
                              ("data", "O")])
        for i, (ty, amb, data) in enumerate(cycles):
            arr[0, i] = (np.array([ty]), np.array([[amb]]),
                         np.array([[tuple(data.values())]],
                                  dtype=[(k, "O") for k in data]))
        sio.savemat(str(d / f"{cell}.mat"), {cell: {"cycle": arr}})
    return str(d)


@pytest.fixture
def ims_dir(tmp_path):
    d = tmp_path / "ims" / "2nd_test"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n_files = 40
    for i in range(n_files):
        # bearing 1 degrades (rising impulsiveness), the other three do not
        sig = rng.normal(0, 1, (256, 4))
        sig[:, 0] *= 1.0 + 4.0 * (i / n_files) ** 3
        name = f"2004.02.{12 + i // 24:02d}.{i % 24:02d}.32.39"
        np.savetxt(d / name, sig, fmt="%.5f", delimiter="\t")
    return str(tmp_path / "ims")


@pytest.fixture
def pcoe_dir(tmp_path):
    d = tmp_path / "pcoe" / "capacitor"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for u in range(6):
        n = 60
        cap = 1.0 - np.linspace(0, 0.3 if u % 2 == 0 else 0.1, n)
        pd.DataFrame({
            "time": np.arange(n),
            "capacitance": cap,
            "esr": 0.1 + np.linspace(0, 0.05, n) + rng.normal(0, 1e-3, n),
            "temperature": 25 + rng.normal(0, 0.5, n),
            "ripple": rng.normal(0, 1, n),
        }).to_csv(d / f"cap_{u}.csv", index=False)
    return str(tmp_path / "pcoe")


# ═══════════════════════════════════════════════════════════════════════════
# Annotated telemetry — real labels all the way to an ADTask
# ═══════════════════════════════════════════════════════════════════════════

def test_esa_preprocessed_roundtrip(esa_preprocessed):
    spec = {"source": "esa", "data_dir": esa_preprocessed, "cache": False,
            "esa_channels": "all", "window": 8, "stride": 2}
    assert dataset_available(spec) == (True, "")
    assert dataset_id(spec) == "esa:Mission1"
    pair = load_fleets(spec, seed=0)

    assert pair.train.annotated and pair.test.annotated
    assert pair.train.n_channels == 4
    assert pair.meta["per_channel_truth"] is True
    assert pair.meta["caveats"], "ESA caveats must travel with the data"

    task = build_ad_task(pair, spec, seed=0)
    assert task.y_test.sum() > 0
    assert task.meta["labels_are_real"] is True
    # The whole point of ESA-ADB here: per-channel annotations give REAL
    # localisation ground truth, which no turbofan source can provide.
    assert task.meta["localisation_truth"] is True
    hit = [a for a in task.affected_test if a]
    assert hit and all(0 <= c < 4 for a in hit for c in a)


def test_esa_train_is_clean_when_asked(esa_preprocessed):
    """`train_on_clean` must actually remove the annotated training windows."""
    base = {"source": "esa", "data_dir": esa_preprocessed, "cache": False,
            "esa_channels": "all", "window": 8, "stride": 2}
    pair = load_fleets(base, seed=0)
    clean = build_ad_task(pair, {**base, "train_on_clean": True}, seed=0)
    dirty = build_ad_task(pair, {**base, "train_on_clean": False}, seed=0)
    assert clean.meta["contaminated_train_windows_dropped"] > 0
    assert len(clean.X_train) < len(dirty.X_train)


def test_esa_rare_event_policy_changes_labels(esa_preprocessed):
    base = {"source": "esa", "data_dir": esa_preprocessed, "cache": False,
            "esa_channels": "all", "window": 8, "stride": 2}
    pair = load_fleets(base, seed=0)
    as_anom = build_ad_task(pair, {**base, "rare_events": "anomaly"}, seed=0)
    as_norm = build_ad_task(pair, {**base, "rare_events": "normal"}, seed=0)
    dropped = build_ad_task(pair, {**base, "rare_events": "drop"}, seed=0)
    assert int(as_anom.y_test.sum()) > int(as_norm.y_test.sum())
    assert "rare_event" in as_anom.kind_test
    assert "rare_event" not in as_norm.kind_test
    assert len(dropped.X_test) < len(as_anom.X_test)


def test_opssat_roundtrip(opssat_csv):
    spec = {"source": "opssat", "data_dir": opssat_csv, "cache": False,
            "window": 8, "stride": 4, "min_length": 16}
    assert dataset_available(spec)[0]
    pair = load_fleets(spec, seed=0)
    assert pair.train.annotated and pair.train.n_channels == 1
    # regime == channel id, so per-regime normalisation handles the scales
    assert pair.train.n_regimes == 3

    task = build_ad_task(pair, spec, seed=0)
    assert 0 < int(task.y_test.sum()) < len(task.y_test)
    # segment-level labels give no per-channel truth, and the task must say so
    assert task.meta["localisation_truth"] is False


def test_smap_msl_roundtrip_and_caveats(smap_msl_dir):
    spec = {"source": "smapmsl", "data_dir": smap_msl_dir, "cache": False,
            "window": 8, "stride": 4}
    pair = load_fleets(spec, seed=0)
    assert pair.train.n_channels == 1, "dims='first' must collapse to telemetry"
    assert len(pair.test) == 3
    assert any("Wu & Keogh" in c for c in pair.meta["caveats"])

    task = build_ad_task(pair, spec, seed=0)
    assert int(task.y_test.sum()) > 0

    wide = load_fleets({**spec, "dims": "all"}, seed=0)
    assert wide.train.n_channels == 25


def test_labeled_injection_can_be_layered_on_top(opssat_csv):
    """Real and injected faults in ONE task: the check on injection realism."""
    spec = {"source": "opssat", "data_dir": opssat_csv, "cache": False,
            "window": 8, "stride": 4, "labeled_inject_rate": 0.5, "strength": 4.0}
    task = build_ad_task(load_fleets(spec, seed=0), spec, seed=0)
    assert task.meta["injected_on_top"] is True
    assert {"spike", "offset", "drift", "decouple", "desync"} & set(task.kind_test)
    # ...but NOT localisation truth.  OPSSAT is univariate, so every injected
    # window "affects" its only channel: the answer is a constant, and scoring
    # attribution against it would report a constant as perfect.  Injection
    # buys per-channel truth only where there are channels to choose between.
    assert task.meta["localisation_truth"] is False
    assert all(a == [0] for a in task.affected_test if a)


def test_censoring_is_refused_on_telemetry(opssat_csv):
    spec = {"source": "opssat", "data_dir": opssat_csv, "cache": False,
            "censor_frac": 0.3}
    with pytest.raises(ValueError, match="no time-to-failure"):
        load_fleets(spec, seed=0)


def test_rul_stage_is_refused_on_telemetry(opssat_csv):
    spec = {"source": "opssat", "data_dir": opssat_csv, "cache": False}
    pair = load_fleets(spec, seed=0)
    with pytest.raises(ValueError, match="no remaining useful life"):
        build_rul_task(pair, spec, seed=0)


# ═══════════════════════════════════════════════════════════════════════════
# Prognostics sources — real RUL all the way to a RULTask
# ═══════════════════════════════════════════════════════════════════════════

def test_phm08_roundtrip(phm08_dir):
    spec = {"source": "phm08", "data_dir": phm08_dir, "window": 6, "stride": 2,
            "rul_stride": 2, "bins": 10, "train_units": 0.6}
    assert dataset_available(spec)[0]
    pair = load_fleets(spec, seed=0)
    assert len(pair.train) == 12
    assert pair.test is None, "PHM08 test RUL is withheld — no official test fleet"
    assert pair.meta["dropped_constant_sensors"] == 1

    rul = build_rul_task(pair, spec, seed=0)
    assert len(rul.X_train) and len(rul.X_test)
    ad = build_ad_task(pair, spec, seed=0)
    assert int(ad.y_test.sum()) > 0


def test_battery_roundtrip_and_genuine_censoring(battery_dir):
    spec = {"source": "battery", "data_dir": battery_dir, "cache": False,
            "window": 5, "stride": 1, "rul_stride": 1, "bins": 8, "cap": 30}
    pair = load_fleets(spec, seed=0)
    assert len(pair.train) == 4
    assert pair.train.n_channels == 8              # BATTERY_FEATURES
    # cells fading 45/50% reach the 30%-fade EOL; those fading 12/15% do not
    # and are genuinely right-censored (not truncated by us)
    assert sorted(pair.train.censored) == [False, False, True, True]
    assert pair.meta["genuine_censoring"] is True
    # capacity must be monotone-ish and RUL must hit 0 exactly at EOL
    failed = pair.train.censored.index(False)
    assert pair.train.rul[failed][-1] == 0

    rul = build_rul_task(pair, spec, seed=0)
    assert int(rul.delta_train.min()) == 0, "censored units must be marked"


def test_ims_bearing_units_and_features(ims_dir):
    spec = {"source": "ims", "data_dir": ims_dir, "cache": False,
            "ims_test": "2nd_test", "window": 5, "stride": 1, "cap": 40}
    pair = load_fleets(spec, seed=0)
    assert len(pair.train) == 4                     # one unit per bearing
    assert pair.train.n_channels == 10              # TIME_FEATURE_NAMES
    # only bearing 1 failed in 2nd_test; the other three are censored survivors
    assert pair.train.censored == [False, True, True, True]
    # the degrading bearing's crest factor must actually rise
    crest = pair.train.series[0][:, 2]
    assert crest[-5:].mean() > crest[:5].mean()

    rig = load_fleets({**spec, "ims_unit": "rig"}, seed=0)
    assert len(rig.train) == 1 and rig.train.n_channels == 40


def test_pcoe_generic_csv_adapter(pcoe_dir):
    spec = {"source": "pcoe", "data_dir": pcoe_dir, "preset": "capacitor",
            "time_col": "time", "window": 5, "stride": 1, "rul_stride": 1,
            "bins": 8, "cap": 40}
    assert dataset_available(spec)[0]
    assert dataset_id(spec) == "pcoe:capacitor"
    pair = load_fleets(spec, seed=0)
    assert len(pair.train) == 6
    assert "time" not in (pair.channel_names or [])
    # 20% capacitance loss = EOL: the cells that fade 30% reach it, the rest
    # are right-censored.
    assert sum(pair.train.censored) == 3
    assert build_rul_task(pair, spec, seed=0).X_train.shape[0] > 0


def test_pcoe_unknown_preset_is_rejected(pcoe_dir):
    with pytest.raises(KeyError, match="unknown PCoE preset"):
        load_fleets({"source": "pcoe", "data_dir": pcoe_dir, "preset": "nope"}, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Invariants that must hold for EVERY source
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fixture_name,spec", [
    ("esa_preprocessed", {"source": "esa", "esa_channels": "all"}),
    ("opssat_csv", {"source": "opssat"}),
    ("smap_msl_dir", {"source": "smapmsl"}),
    ("phm08_dir", {"source": "phm08"}),
    # cap must sit well below the shortest life or the health proxy calls
    # every window degraded; these two fixtures are 40-60 steps long, against
    # a source default sized for real runs
    ("ims_dir", {"source": "ims", "cap": 15}),
    ("pcoe_dir", {"source": "pcoe", "preset": "capacitor", "time_col": "time",
                  "cap": 15}),
])
def test_no_window_leaks_between_train_and_test(fixture_name, spec, request):
    """
    The one invariant that makes every number meaningful: no test window may
    also be a training window.  Overlapping windows from one unit are near
    duplicates, so a leak here inflates every metric silently.
    """
    data_dir = request.getfixturevalue(fixture_name)
    spec = {**spec, "data_dir": data_dir, "cache": False,
            "window": 6, "stride": 3, "min_length": 16}
    task = build_ad_task(load_fleets(spec, seed=0), spec, seed=0)
    tr = {r.tobytes() for r in task.X_train.numpy()}
    te = {r.tobytes() for r in task.X_test.numpy()}
    assert not (tr & te), f"{spec['source']}: test windows appear in training"
    assert task.X_train.shape[1] == task.X_test.shape[1]
    assert len(task.y_test) == len(task.X_test) == len(task.kind_test)
    assert len(task.affected_test) == len(task.X_test)


@pytest.mark.parametrize("fixture_name,spec", [
    ("phm08_dir", {"source": "phm08"}),
    ("ims_dir", {"source": "ims", "cap": 40}),
    ("pcoe_dir", {"source": "pcoe", "preset": "capacitor", "time_col": "time",
                  "cap": 40}),
])
def test_rul_labels_are_consistent(fixture_name, spec, request):
    """RUL must decrease by one per step and end at 0 for uncensored units."""
    data_dir = request.getfixturevalue(fixture_name)
    pair = load_fleets({**spec, "data_dir": data_dir, "cache": False}, seed=0)
    for u in range(len(pair.train)):
        r = pair.train.rul[u]
        assert np.all(np.diff(r) == -1), "RUL must be one cycle per step"
        assert r[-1] == 0
        h = pair.train.health[u]
        assert h.min() >= 0.0 and h.max() <= 1.0
        assert h[-1] == 1.0, "health must reach 1 at the last observed step"


def test_caveats_reach_the_task_meta(smap_msl_dir):
    """A contested dataset must carry its objections into the run log."""
    spec = {"source": "smapmsl", "data_dir": smap_msl_dir, "cache": False,
            "window": 6, "stride": 3}
    pair = load_fleets(spec, seed=0)
    task = build_ad_task(pair, spec, seed=0)
    assert task.meta["caveats"] == pair.meta["caveats"]
    assert task.meta["not_recommended_as_primary_evidence"] is True
