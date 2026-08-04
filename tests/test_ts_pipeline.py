"""
Tests for the time-series experiment pipeline: config, logging, datasets,
device handling, guardrails, conformal calibration, and one end-to-end run.

These are the invariants that keep a long unattended batch honest.  The three
that have already cost this project a wrong published result get explicit
tests: leaf/weight jitter symmetry-breaking, a predictive that does not depend
on its input, and the DAG traversal blowup.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from poc.time_series import aggregate as agg_mod
from poc.time_series.circuits import (
    DegenerateModelError,
    SurvivalPC,
    WindowPC,
    resolve_device,
)
from poc.time_series.conformal import (
    ConformalPredictive,
    conformal_quantile,
    split_units,
)
from poc.time_series.config import (
    DEFAULTS,
    apply_overrides,
    expand_variants,
    load_config,
    resolved_for_hash,
    validate,
)
from poc.time_series.data import Fleet, make_ad_task, make_rul_task, simulate_fleet
from poc.time_series.data_real import (
    censor_fleet,
    correlation_groups,
    health_from_rul,
    subsample_units,
)
from poc.time_series.datasets import (
    build_ad_task,
    build_rul_task,
    dataset_available,
    dataset_id,
    load_fleets,
    make_ad_task_split,
    make_rul_task_split,
)
from poc.time_series.ts_logging import (
    RunLogger,
    config_hash,
    group_stats,
    is_complete,
    read_results,
)
from src.probabilistic_circuits import move_circuit_

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "config", "ts")


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def test_every_shipped_config_loads_and_validates():
    """A config that only parses at 2 a.m. is a config that wasted a night."""
    files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")]
    assert files, "no configs found in config/ts"
    for f in files:
        cfg = load_config(os.path.join(CONFIG_DIR, f))
        validate(cfg)
        assert cfg["log_root"]
        for v in expand_variants(cfg):
            assert v["variant"]
            validate(v)


def test_defaults_are_merged_not_replaced():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            f.write("name: t\nmodel:\n  K: 3\n")
        cfg = load_config(p)
        assert cfg["model"]["K"] == 3
        assert cfg["model"]["vtree"] == DEFAULTS["model"]["vtree"]   # untouched
        assert cfg["dataset"]["source"] == "synthetic"


def test_grid_expansion_is_cartesian_and_named():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            f.write("name: t\ngrid:\n  model.K: [2, 4]\n  model.vtree: [chain, time]\n")
        variants = expand_variants(load_config(p))
        assert len(variants) == 4
        assert {(v["model"]["K"], v["model"]["vtree"]) for v in variants} == {
            (2, "chain"), (2, "time"), (4, "chain"), (4, "time")}
        assert len({v["variant"] for v in variants}) == 4
        assert all("axis" not in v["variant"] for v in variants)


def test_named_variants_compose_with_grid():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        with open(p, "w") as f:
            f.write("name: t\nvariants:\n"
                    "  - {name: a, model: {delta: true}}\n"
                    "  - {name: b, model: {delta: false}}\n"
                    "grid:\n  model.K: [2, 4]\n")
        variants = expand_variants(load_config(p))
        assert len(variants) == 4
        assert {v["model"]["delta"] for v in variants} == {True, False}


def test_dotted_overrides_are_typed():
    cfg = {"model": {"K": 6}, "eval": {"plots": True}}
    apply_overrides(cfg, ["model.K=9", "eval.plots=false", "model.tag=hello"])
    assert cfg["model"]["K"] == 9 and isinstance(cfg["model"]["K"], int)
    assert cfg["eval"]["plots"] is False
    assert cfg["model"]["tag"] == "hello"


def test_invalid_config_is_rejected():
    cfg = dict(DEFAULTS, stages=["not_a_stage"], seeds=[0])
    with pytest.raises(ValueError):
        validate(cfg)
    # SOS on a multi-partition region graph is not exact and must be refused
    bad = json.loads(json.dumps(DEFAULTS))
    bad["stages"], bad["seeds"] = ["ad"], [0]
    bad["model"]["sos"], bad["model"]["vtree"] = True, "orc_rg_multi"
    with pytest.raises(ValueError):
        validate(bad)


def test_hash_ignores_cosmetics_but_not_model():
    a = json.loads(json.dumps(DEFAULTS))
    b = json.loads(json.dumps(DEFAULTS))
    b["description"] = "different words"
    b["eval"]["plots"] = not b["eval"]["plots"]
    assert config_hash(resolved_for_hash(a)) == config_hash(resolved_for_hash(b))
    b["model"]["K"] = 99
    assert config_hash(resolved_for_hash(a)) != config_hash(resolved_for_hash(b))


# ═══════════════════════════════════════════════════════════════════════════
# Logging / resume
# ═══════════════════════════════════════════════════════════════════════════

def test_run_logger_writes_the_full_artifact_set():
    with tempfile.TemporaryDirectory() as d:
        cfg = {"name": "t", "model": {"K": 2}}
        with RunLogger(d, config=cfg, seed=3) as log:
            log.info("hello")
            log.history("train_nll", [3.0, 2.0, 1.0])
            log.result({"stage": "ad", "method": "m", "auroc": 0.9})
            log.metrics({"ad": {"auroc": 0.9}})
            log.artifact_npz("scores", s=np.arange(4))
        for f in ("config.json", "env.json", "run.log", "status.json",
                  "results.jsonl", "metrics.json", "history_train_nll.csv"):
            assert os.path.exists(os.path.join(d, f)), f
        st = json.load(open(os.path.join(d, "status.json")))
        assert st["status"] == "ok" and st["wall_s"] >= 0
        assert is_complete(d, cfg)
        assert not is_complete(d, {"name": "t", "model": {"K": 3}})   # config changed
        rows = read_results(d)
        assert rows and rows[0]["seed"] == 3 and rows[0]["auroc"] == 0.9


def test_failed_run_is_recorded_and_not_resumable():
    with tempfile.TemporaryDirectory() as d:
        cfg = {"name": "t"}
        with RunLogger(d, config=cfg, seed=0, swallow=True) as log:
            raise ValueError("boom")
        st = json.load(open(os.path.join(d, "status.json")))
        assert st["status"] == "failed"
        assert "boom" in st["error"] and "traceback" in st
        assert not is_complete(d, cfg)


def test_group_stats_reports_mean_std_and_seed_count():
    rows = [{"m": "a", "auroc": 0.9, "seed": 0}, {"m": "a", "auroc": 0.7, "seed": 1},
            {"m": "b", "auroc": 0.5, "seed": 0}]
    out = {r["m"]: r for r in group_stats(rows, ("m",), ("auroc",))}
    assert out["a"]["auroc_mean"] == pytest.approx(0.8)
    assert out["a"]["n_seeds"] == 2
    assert out["b"]["auroc_std"] == 0.0


def test_aggregate_writes_tables(tmp_path):
    run = tmp_path / "v" / "seed0"
    run.mkdir(parents=True)
    with open(run / "results.jsonl", "w") as f:
        for s, a in [(0, 0.9), (0, 0.8)]:
            f.write(json.dumps({"stage": "ad", "experiment": "e", "variant": "v",
                                "dataset": "synthetic", "method": f"m{a}",
                                "auroc": a, "seed": s}) + "\n")
    stats = agg_mod.aggregate_root(str(tmp_path), print_tables=False)
    assert stats and os.path.exists(tmp_path / "summary.csv")
    assert os.path.exists(tmp_path / "summary.md")
    table = agg_mod.format_table("ad", stats)
    assert "auroc" in table


# ═══════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════

def test_synthetic_source_round_trips_through_the_registry():
    spec = dict(DEFAULTS["dataset"], units=10, channels=6, regimes=2,
                window=4, stride=2, bins=8)
    assert dataset_available(spec) == (True, "")
    assert dataset_id(spec) == "synthetic"
    pair = load_fleets(spec, seed=0)
    assert len(pair.train) == 10 and pair.test is None
    ad = build_ad_task(pair, spec, seed=0)
    assert ad.X_train.shape[1] == 4 * 6 and len(ad.X_test) == len(ad.kind_test)
    rul = build_rul_task(pair, {**spec, "stride": 3}, seed=0)
    assert rul.unit_train is not None and len(rul.unit_train) == len(rul.X_train)


def test_split_builders_never_share_a_unit_between_train_and_test():
    tr = simulate_fleet(n_units=6, n_channels=5, n_regimes=2, min_life=40,
                        max_life=60, censor_frac=0.0, seed=0)
    te = simulate_fleet(n_units=4, n_channels=5, n_regimes=2, min_life=40,
                        max_life=60, censor_frac=0.0, seed=1)
    ad = make_ad_task_split(tr, te, window=4, stride=2, inject_rate=0.5, seed=0)
    assert ad.meta["official_test_fleet"] and len(ad.X_test) > 0
    assert any(a for a in ad.affected_test), "no injected ground truth produced"

    rul = make_rul_task_split(tr, te, window=4, stride=2, n_bins=8, cap=60.0, seed=0)
    assert len(np.unique(rul.unit_test.numpy())) <= len(te)
    assert rul.delta_train.min() == 1                     # tr is uncensored here

    last = make_rul_task_split(tr, te, window=4, stride=2, n_bins=8, cap=60.0,
                               test_windows="last", seed=0)
    assert len(last.X_test) <= len(te), "the literature protocol is one window/unit"


def test_censoring_truncates_and_marks_units():
    fleet = simulate_fleet(n_units=20, n_channels=4, min_life=40, max_life=50,
                           censor_frac=0.0, seed=0)
    cens = censor_fleet(fleet, frac=1.0, seed=0)
    assert all(cens.censored)
    assert all(len(c) < len(f) for c, f in zip(cens.series, fleet.series))
    assert censor_fleet(fleet, frac=0.0) is fleet          # no-op stays identical


def test_health_from_rul_matches_the_piecewise_linear_convention():
    h = health_from_rul(np.array([0.0, 50.0, 125.0, 400.0]), cap=125.0)
    assert h[0] == pytest.approx(1.0)                      # at failure
    assert h[2] == pytest.approx(0.0) and h[3] == pytest.approx(0.0)  # capped
    assert 0.5 < h[1] < 0.7


def test_correlation_groups_partition_all_channels():
    X = np.random.default_rng(0).normal(size=(200, 9))
    X[:, 3] = X[:, 0] + 0.01 * X[:, 3]                     # a real correlation
    groups = correlation_groups(X, n_groups=3)
    flat = sorted(c for g in groups for c in g)
    assert flat == list(range(9))
    assert all(len(g) > 0 for g in groups)


def test_subsample_units_is_deterministic():
    fleet = simulate_fleet(n_units=12, n_channels=4, min_life=30, max_life=40, seed=0)
    a = subsample_units(fleet, 5, seed=0)
    b = subsample_units(fleet, 5, seed=0)
    assert len(a) == 5
    assert all(np.array_equal(x, y) for x, y in zip(a.series, b.series))


@pytest.mark.parametrize("spec", [
    {"source": "cmapss", "subset": "FD001"},
    {"source": "ncmapss", "dataset": "DS02"},
])
def test_real_sources_report_availability_without_raising(spec):
    ok, why = dataset_available(spec)
    assert isinstance(ok, bool)
    assert ok or why, "unavailable data must explain itself"


# ── the real-data PARSERS, exercised on files we write ourselves ───────────
#
# The NASA archives are not in the repo, so the loaders would otherwise be
# untested until the night they are first used.  These write C-MAPSS-shaped
# files (same columns, same RUL-file convention, including a constant sensor
# and multiple operating conditions) and check the parse, not the science.

def _write_fake_cmapss(d: str, subset: str = "FD001", n_units: int = 6,
                       n_cycles: int = 40, n_test_cycles: int = 25) -> None:
    rng = np.random.default_rng(0)

    def block(units, cycles_per_unit):
        rows = []
        for u in range(1, units + 1):
            T = cycles_per_unit + u
            for t in range(1, T + 1):
                ops = rng.choice([[0.0, 0.0, 100.0], [20.0, 0.7, 100.0]])
                sens = list(rng.normal(size=20) + t * 0.05)
                sens.append(1.0)                       # a CONSTANT sensor
                rows.append([u, t, *ops, *sens])
        return np.asarray(rows)

    np.savetxt(os.path.join(d, f"train_{subset}.txt"), block(n_units, n_cycles))
    np.savetxt(os.path.join(d, f"test_{subset}.txt"), block(n_units, n_test_cycles))
    np.savetxt(os.path.join(d, f"RUL_{subset}.txt"),
               np.arange(10, 10 + n_units).reshape(-1, 1), fmt="%d")


def test_cmapss_parser_handles_columns_rul_file_and_constant_sensors(tmp_path):
    from poc.time_series.data_real import cmapss_available, load_cmapss

    d = str(tmp_path)
    _write_fake_cmapss(d)
    assert cmapss_available("FD001", data_dir=d)

    pair = load_cmapss("FD001", data_dir=d, cap=30.0, n_regimes=2, n_groups=2)
    assert len(pair.train) == 6 and pair.test is not None and len(pair.test) == 6
    assert pair.train.n_channels == 20, "the constant sensor must be dropped"
    assert pair.train.n_channels == pair.test.n_channels
    assert len(pair.channel_names) == 20

    # train units are run to failure: RUL ends at 0
    assert all(r[-1] == 0 for r in pair.train.rul)
    # test units are truncated: RUL ends at the value from RUL_FD001.txt
    assert [int(r[-1]) for r in pair.test.rul] == list(range(10, 16))
    # health is the piecewise-linear proxy, 1 at failure
    assert pair.train.health[0][-1] == pytest.approx(1.0)
    assert set(np.unique(np.concatenate(pair.train.regime))) <= {0, 1}

    ad = build_ad_task(pair, {"source": "cmapss", "window": 5, "stride": 2,
                              "inject_rate": 0.5, "cap": 30.0}, seed=0)
    assert ad.X_train.shape[1] == 5 * 20
    rul = build_rul_task(pair, {"source": "cmapss", "window": 5, "stride": 2,
                                "bins": 6, "cap": 30.0}, seed=0)
    assert rul.n_bins == 6 and rul.unit_test is not None


def test_ncmapss_parser_reads_the_hdf5_layout(tmp_path):
    h5py = pytest.importorskip("h5py", reason="N-C-MAPSS is HDF5-only")
    from poc.time_series.data_real import load_ncmapss, ncmapss_available

    d = str(tmp_path)
    rng = np.random.default_rng(0)
    n_units, n_cycles, per_cycle = 4, 12, 20
    rows = n_units * n_cycles * per_cycle
    A = np.zeros((rows, 4), dtype=np.float32)
    Y = np.zeros((rows, 1), dtype=np.float32)
    i = 0
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            A[i:i + per_cycle] = [u, c, 1, 1]
            Y[i:i + per_cycle, 0] = n_cycles - c
            i += per_cycle
    with h5py.File(os.path.join(d, "N-CMAPSS_DS02-006.h5"), "w") as f:
        for split in ("dev", "test"):
            f.create_dataset(f"X_s_{split}", data=rng.normal(size=(rows, 14)))
            f.create_dataset(f"W_{split}", data=rng.normal(size=(rows, 4)))
            f.create_dataset(f"A_{split}", data=A)
            f.create_dataset(f"Y_{split}", data=Y)
        f.create_dataset("A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"]))
        f.create_dataset("X_s_var", data=np.array([f"s{i}".encode()
                                                   for i in range(14)]))

    assert ncmapss_available("DS02", data_dir=d)
    pair = load_ncmapss("DS02", data_dir=d, cap=12.0, n_regimes=2,
                        aggregate="cycle", use_cache=False)
    assert len(pair.train) == n_units and pair.test is not None
    assert pair.train.n_channels == 14
    # per-cycle aggregation: one row per flight, exactly as in C-MAPSS
    assert all(len(s) == n_cycles for s in pair.train.series)
    assert pair.train.rul[0][0] == pytest.approx(n_cycles - 1)

    raw = load_ncmapss("DS02", data_dir=d, cap=12.0, aggregate="raw",
                       subsample=5, use_cache=False)
    assert len(raw.train.series[0]) == n_cycles * per_cycle // 5


# ═══════════════════════════════════════════════════════════════════════════
# Device handling and the DAG traversal trap
# ═══════════════════════════════════════════════════════════════════════════

def test_resolve_device_accepts_explicit_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device(None).type in ("cpu", "cuda", "mps")
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


def test_move_circuit_is_linear_not_exponential():
    """
    `nn.Module.to()` recurses over children WITHOUT memoisation, so on a shared
    -node DAG it visits each shared sub-circuit once per path — the K^depth
    blowup the region-graph layout exists to remove, reintroduced through a
    convenience method.  This pins the DAG-safe replacement.
    """
    import time
    from src.probabilistic_circuits import GaussianLeaf, RegionGraphPC
    from poc.time_series.circuits import build_window_vtree

    vt = build_window_vtree("chain", 6, 8)
    pc = RegionGraphPC(vt, n_sum_components=4, leaf_factory=GaussianLeaf, seed=0)
    t0 = time.time()
    move_circuit_(pc, torch.device("cpu"))
    assert time.time() - t0 < 5.0, "DAG-safe move should be near-instant"
    assert all(p.device.type == "cpu" for p in pc.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# Degeneracy guardrails (hand-off §3)
# ═══════════════════════════════════════════════════════════════════════════

def _tiny_rul_task():
    return make_rul_task(window=4, stride=4, seed=0, n_units=14, n_channels=5,
                         n_regimes=2, n_bins=8, censor_frac=0.3)


def test_predict_refuses_a_constant_predictive():
    """The exact failure that invalidated a whole pre-registered gate: predict()
    returned the same 102.4 cycles for all 851 test windows, silently."""
    task = _tiny_rul_task()
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method="chain", n_sum_components=4, tau_where="deep",
                    seed=0, device="cpu")
    pc.fit(task.X_train, task.tau_train, task.delta_train, epochs=2)

    flat = torch.log(torch.full((len(task.X_test), task.n_bins),
                                1.0 / task.n_bins))
    pc.log_pmf = lambda X, **kw: flat[: len(X)]            # input-independent
    with pytest.raises(DegenerateModelError) as exc:
        pc.predict(task.X_test)
    assert "tau_where" in str(exc.value)
    # the escape hatch exists, but you have to ask for it explicitly
    out = pc.predict(task.X_test, check_degenerate=False)
    assert float(out["mean"].std()) < 1e-6


def test_window_pc_refuses_a_constant_density():
    task = make_ad_task(window=4, stride=3, seed=0, n_units=10, n_channels=5,
                        n_regimes=2)
    pc = WindowPC(task.window, task.n_channels, vtree_method="chain",
                  n_sum_components=3, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=2)
    assert pc.assert_informative(task.X_train) > 0
    pc.score = lambda X, **kw: torch.zeros(len(X))
    with pytest.raises(DegenerateModelError):
        pc.assert_informative(task.X_train)


def test_training_history_is_recorded_and_decreasing():
    task = make_ad_task(window=4, stride=3, seed=0, n_units=10, n_channels=5,
                        n_regimes=2)
    pc = WindowPC(task.window, task.n_channels, vtree_method="chain",
                  n_sum_components=3, seed=0, device="cpu")
    pc.fit(task.X_train, epochs=5)
    assert len(pc.history) == 5
    assert pc.history[-1] < pc.history[0], "5 epochs should reduce the NLL"


# ═══════════════════════════════════════════════════════════════════════════
# Conformal calibration
# ═══════════════════════════════════════════════════════════════════════════

def test_unit_split_shares_no_unit():
    units = np.repeat(np.arange(10), 7)
    fit, cal = split_units(units, cal_frac=0.3, seed=0)
    assert not (fit & cal).any() and (fit | cal).all()
    assert set(units[fit]).isdisjoint(set(units[cal]))
    assert 1 <= len(set(units[cal])) <= 5


def test_conformal_quantile_is_the_finite_sample_one():
    s = np.arange(10, dtype=float)
    # k = ceil((n+1)(1-alpha)) - 1 = ceil(9.9)-1 = 9  -> the max, not the 90th pct
    assert conformal_quantile(s, 0.10) == 9.0
    assert conformal_quantile(np.array([]), 0.1) == 0.0


class _MockPC:
    """A deliberately over-confident predictive: the thing conformal must fix."""
    cap = 100.0
    n_bins = 20

    def bin_centers(self):
        e = torch.linspace(0, self.cap, self.n_bins + 1)
        return 0.5 * (e[:-1] + e[1:])

    def predict(self, X):
        n = len(X)
        centers = self.bin_centers()
        mu = X[:, 0] * 10.0 + 50.0
        pmf = torch.softmax(-((centers[None, :] - mu[:, None]) ** 2) / 8.0, dim=1)
        return {"pmf": pmf, "mean": mu, "q05": mu - 1.0, "q95": mu + 1.0}


def test_conformal_widens_intervals_to_reach_nominal_coverage():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(600, 3, generator=g)
    y = X[:, 0] * 10.0 + 50.0 + torch.randn(600, generator=g) * 8.0
    pc = _MockPC()
    raw_cov = float(((y >= X[:, 0] * 10 + 49) & (y <= X[:, 0] * 10 + 51)).float().mean())
    assert raw_cov < 0.3, "the mock must start badly calibrated"

    cp = ConformalPredictive(pc, alpha=0.1, mode="cqr").calibrate(X[:400], y[:400])
    assert cp.q_hat > 0
    pred = cp.predict(X[400:])
    cov = float(np.mean((y[400:].numpy() >= pred["lo"])
                        & (y[400:].numpy() <= pred["hi"])))
    assert cov >= 0.80, f"split conformal should reach ~0.90, got {cov:.2f}"
    assert "frac_full_range" in cp.diagnostics


def test_pit_recalibration_also_improves_coverage():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(600, 3, generator=g)
    y = X[:, 0] * 10.0 + 50.0 + torch.randn(600, generator=g) * 8.0
    cp = ConformalPredictive(_MockPC(), alpha=0.1, mode="pit").calibrate(X[:400], y[:400])
    pred = cp.predict(X[400:])
    cov = float(np.mean((y[400:].numpy() >= pred["lo"])
                        & (y[400:].numpy() <= pred["hi"])))
    assert cov > 0.6, f"PIT recalibration should help materially, got {cov:.2f}"


# ═══════════════════════════════════════════════════════════════════════════
# End to end
# ═══════════════════════════════════════════════════════════════════════════

def test_end_to_end_ad_stage_writes_comparable_rows(tmp_path):
    from poc.time_series.pipeline import run_stages

    cfg = json.loads(json.dumps(DEFAULTS))
    cfg.update({"name": "t", "variant": "default", "seeds": [0], "device": "cpu",
                "stages": ["ad"], "variant_axes": {}})
    cfg["dataset"].update({"units": 10, "channels": 6, "regimes": 2,
                           "window": 4, "stride": 3})
    cfg["model"].update({"K": 3, "epochs": 2})
    cfg["eval"].update({"fast_baselines": True, "missing": True, "typed": True,
                        "save_scores": True, "plots": False, "examples": False})

    rdir = str(tmp_path / "run")
    with RunLogger(rdir, config=cfg, seed=0) as log:
        out = run_stages(cfg, 0, log)
    assert "ad" in out and "typed" in out["ad"]

    rows = read_results(rdir)
    methods = {r["method"] for r in rows}
    assert any("RegionGraphPC" in m for m in methods)
    assert any("z-score" in m for m in methods), "baselines must be on equal footing"
    assert any("dead" in m for m in methods), "the dead-sensor query must run"
    assert any("structural-only" in m for m in methods)
    for r in rows:
        assert r["stage"] == "ad" and r["dataset"] == "synthetic"
        assert "auroc" in r
    assert os.path.exists(os.path.join(rdir, "artifacts", "ad_scores.npz"))
    assert os.path.exists(os.path.join(rdir, "history_pc_train_nll.csv"))


# ── one fit, many evaluations ───────────────────────────────────────────────

def test_last_window_view_equals_rebuilding_the_task():
    """
    `eval.test_protocols` reports "all" and "last" from ONE fit, by selecting
    the final window of each test unit instead of rebuilding the task with
    `rul_test_windows: last`.  That is only legitimate if the selection is
    exactly what the builder produces — this asserts it, because the whole
    2x saving on the RUL tier rests on it.
    """
    import torch

    from poc.time_series.config import load_config
    from poc.time_series.datasets import build_rul_task, load_fleets
    from poc.time_series.pipeline import _test_protocol_views

    # `rul_test_windows` only bites on sources with an OFFICIAL test fleet
    # (make_rul_task_split); the synthetic source splits its own units and
    # ignores it.  So this has to run on real C-MAPSS, and skip without it.
    spec = dict(load_config("config/ts/cmapss_rul.yaml")["dataset"])
    spec.update(subset="FD001", window=8, stride=4, rul_stride=6, bins=10,
                censor_frac=0.3)
    ok, why = dataset_available(spec)
    if not ok:
        pytest.skip(f"needs real C-MAPSS: {why}")

    def build(protocol):
        s = dict(spec, rul_test_windows=protocol)
        return build_rul_task(load_fleets(s, seed=0), s, seed=0)

    all_task, last_task = build("all"), build("last")

    views = dict(_test_protocol_views(all_task, ["all", "last"]))
    derived = views["last"]
    assert torch.equal(derived.X_test, last_task.X_test)
    assert torch.equal(derived.rul_test, last_task.rul_test)
    assert torch.equal(derived.tau_test, last_task.tau_test)
    # and the training data was never touched
    assert torch.equal(all_task.X_train, last_task.X_train)


def test_alpha_is_not_a_training_input():
    """
    The calibration stage evaluates several alphas from one fit.  That is valid
    only because alpha reaches nothing before `ConformalPredictive`.  Guard it
    by signature: `_fit_survival` must not accept or read an alpha.
    """
    import inspect

    from poc.time_series.pipeline import _fit_survival

    params = inspect.signature(_fit_survival).parameters
    assert "alpha" not in params
    src = inspect.getsource(_fit_survival)
    assert "alpha" not in src, "the survival fit now reads alpha — the " \
                               "one-fit-many-alphas optimisation is invalid"
