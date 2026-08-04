"""
Tests for the seed-aware logging machinery (src/logging_utils.py) and the
seeded multi-execution flow (src/experiment.run_experiment).

Key properties: logs land in logs/<seed>/, same seed → bit-identical results,
different seeds → independent executions, aggregation reports mean ± std.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment import run_experiment
from src.logging_utils import (
    ExperimentLogger,
    aggregate_results,
    format_summary_table,
    save_summary,
    set_global_seed,
)
from src.probabilistic_circuits import GaussianLeaf
from tests.test_datasets_pipeline import synthetic_tabular


# ─── ExperimentLogger ─────────────────────────────────────────────────────────

def test_logger_creates_seed_folder(tmp_path):
    elog = ExperimentLogger(seed=7, root=str(tmp_path), echo=False)
    elog.info("hello")
    elog.log_config({"latent_dim": 8})
    elog.log_history("train_nll", [3.0, 2.0, 1.5])
    elog.log_result({"dataset": "x", "adaptation": "zero_shot", "auroc": 0.9})
    elog.close()

    seed_dir = tmp_path / "7"
    assert seed_dir.is_dir()
    assert (seed_dir / "run.log").exists()
    assert json.loads((seed_dir / "config.json").read_text())["seed"] == 7
    csv = (seed_dir / "history_train_nll.csv").read_text().strip().splitlines()
    assert csv[0] == "epoch,value" and len(csv) == 4
    results = [json.loads(l) for l in (seed_dir / "results.jsonl").read_text().splitlines()]
    assert results[0]["auroc"] == 0.9 and results[0]["seed"] == 7
    assert "hello" in (seed_dir / "run.log").read_text()


def test_logger_results_append_across_runs(tmp_path):
    for _ in range(2):
        elog = ExperimentLogger(seed=1, root=str(tmp_path), echo=False)
        elog.log_result({"dataset": "x", "auroc": 0.5})
        elog.close()
    elog = ExperimentLogger(seed=1, root=str(tmp_path), echo=False)
    assert len(elog.read_results()) == 2
    elog.close()


# ─── Seeding ──────────────────────────────────────────────────────────────────

def test_set_global_seed_reproducible():
    set_global_seed(3)
    a = (torch.randn(5), __import__("numpy").random.rand(5))
    set_global_seed(3)
    b = (torch.randn(5), __import__("numpy").random.rand(5))
    assert torch.equal(a[0], b[0])
    assert (a[1] == b[1]).all()


def _quick_run(seed, tmp_root):
    src = synthetic_tabular(d=8, seed=100, name="src")
    new = synthetic_tabular(d=10, seed=101, name="new")
    return run_experiment(
        [src], [new], seed=seed, adapt_modes=("zero_shot", "leaves"),
        latent_dim=6, n_sum_components=2, epochs=15,
        log_root=str(tmp_root), leaf_factory=GaussianLeaf,
    )


def test_run_experiment_same_seed_identical(tmp_path):
    r1 = _quick_run(0, tmp_path / "a")
    r2 = _quick_run(0, tmp_path / "b")
    for x, y in zip(r1, r2):
        assert x["dataset"] == y["dataset"] and x["adaptation"] == y["adaptation"]
        assert x["auroc"] == pytest.approx(y["auroc"], abs=1e-12)


def test_run_experiment_writes_seed_logs(tmp_path):
    results = _quick_run(5, tmp_path)
    seed_dir = tmp_path / "5"
    assert (seed_dir / "run.log").exists()
    assert (seed_dir / "history_train_nll.csv").exists()
    logged = [json.loads(l) for l in (seed_dir / "results.jsonl").read_text().splitlines()]
    assert len(logged) == len(results) == 3      # 1 source + 2 adaptation modes
    assert all(r["seed"] == 5 for r in logged)
    roles = {r["role"] for r in logged}
    assert roles == {"source", "held_out"}


def test_run_experiment_multiple_seeds_vary(tmp_path):
    # overlapping anomalies (1.5σ shift), otherwise AUROC saturates at 1.0
    # for every seed and there is no variance left to observe
    src = synthetic_tabular(d=8, seed=100, name="src", anomaly_shift=1.5)
    new = synthetic_tabular(d=10, seed=101, name="new", anomaly_shift=1.5)
    rs = []
    for s in (0, 1, 2):
        r = run_experiment([src], [new], seed=s, adapt_modes=("zero_shot",),
                           latent_dim=6, n_sum_components=2, epochs=15,
                           log_root=str(tmp_path), leaf_factory=GaussianLeaf)
        rs.append(r[1]["auroc"])               # held-out zero-shot result
        assert (tmp_path / str(s)).is_dir()
    assert len({round(r, 12) for r in rs}) > 1, "different seeds should differ"


# ─── Aggregation ──────────────────────────────────────────────────────────────

def test_aggregate_results_mean_std():
    results = [
        {"dataset": "d1", "adaptation": "zero_shot", "auroc": 0.8, "seed": 0},
        {"dataset": "d1", "adaptation": "zero_shot", "auroc": 0.9, "seed": 1},
        {"dataset": "d1", "adaptation": "leaves", "auroc": 0.7, "seed": 0},
    ]
    agg = aggregate_results(results)
    zs = next(a for a in agg if a["adaptation"] == "zero_shot")
    assert zs["auroc_mean"] == pytest.approx(0.85)
    assert zs["auroc_std"] == pytest.approx(0.0707, abs=1e-3)
    assert zs["n_seeds"] == 2 and zs["seeds"] == [0, 1]
    lv = next(a for a in agg if a["adaptation"] == "leaves")
    assert lv["auroc_std"] == 0.0 and lv["n_seeds"] == 1


def test_save_summary_and_table(tmp_path):
    agg = aggregate_results([
        {"dataset": "d1", "adaptation": "zero_shot", "auroc": 0.8, "seed": 0},
        {"dataset": "d1", "adaptation": "zero_shot", "auroc": 0.9, "seed": 1},
    ])
    path = save_summary(agg, root=str(tmp_path))
    assert json.loads(open(path).read())["results"][0]["n_seeds"] == 2
    table = format_summary_table(agg)
    assert "d1" in table and "±" in table


def test_end_to_end_multiseed_aggregation(tmp_path):
    all_results = []
    for s in (0, 1):
        all_results += _quick_run(s, tmp_path)
    agg = aggregate_results(all_results)
    held = [a for a in agg if a["dataset"] == "new"]
    assert {a["adaptation"] for a in held} == {"zero_shot", "leaves"}
    for a in held:
        assert a["n_seeds"] == 2
        assert 0.0 <= a["auroc_mean"] <= 1.0
