"""
Tests for config-file-driven execution (src/experiment.py: load_config,
run_from_config) — fully offline via a pre-seeded fake ADBench cache.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment import DEFAULT_CONFIG, load_config, run_from_config


def make_fake_adbench(tmp_path, names=("faketask", "faketask2"), n=240, d=6):
    data_dir = tmp_path / "adbench"
    data_dir.mkdir()
    with open(data_dir / "_index.json", "w") as f:
        json.dump([f"9{i}_{name}.npz" for i, name in enumerate(names)], f)
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        X = rng.normal(size=(n, d)).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)
        y[-n // 6:] = 1
        X[y == 1] += 6.0
        np.savez(data_dir / f"9{i}_{name}.npz", X=X, y=y)
    return str(data_dir)


def write_yaml(path, text):
    with open(path, "w") as f:
        f.write(text)
    return str(path)


# ─── load_config ──────────────────────────────────────────────────────────────

def test_load_config_merges_defaults(tmp_path):
    p = write_yaml(tmp_path / "c.yaml", """
name: t
train: [adbench:a]
test: [adbench:b]
epochs: 7
""")
    cfg = load_config(p)
    assert cfg["epochs"] == 7
    assert cfg["latent_dim"] == DEFAULT_CONFIG["latent_dim"]   # default kept
    assert cfg["seeds"] == [0]


def test_load_config_rejects_unknown_keys(tmp_path):
    p = write_yaml(tmp_path / "c.yaml", """
train: [adbench:a]
test: [adbench:b]
epochz: 7
""")
    with pytest.raises(KeyError, match="epochz"):
        load_config(p)


def test_load_config_requires_train_and_test(tmp_path):
    p = write_yaml(tmp_path / "c.yaml", "name: t\ntrain: [adbench:a]\ntest: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        load_config(p)


def test_load_config_rejects_bad_adapt_mode(tmp_path):
    p = write_yaml(tmp_path / "c.yaml", """
train: [adbench:a]
test: [adbench:b]
adapt: [zero_shot, full_retrain]
""")
    with pytest.raises(ValueError, match="full_retrain"):
        load_config(p)


def test_load_config_json(tmp_path):
    p = tmp_path / "c.json"
    with open(p, "w") as f:
        json.dump({"train": ["adbench:a"], "test": ["adbench:b"]}, f)
    assert load_config(str(p))["train"] == ["adbench:a"]


def test_load_config_bad_extension(tmp_path):
    p = write_yaml(tmp_path / "c.txt", "train: [a]\ntest: [b]\n")
    with pytest.raises(ValueError, match="yaml"):
        load_config(p)


# ─── run_from_config (offline end-to-end) ────────────────────────────────────

def test_run_from_config_end_to_end(tmp_path):
    data_dir = make_fake_adbench(tmp_path)
    log_root = tmp_path / "logs"
    p = write_yaml(tmp_path / "exp.yaml", f"""
name: cfg_test
train:
  - spec: adbench:faketask
    data_dir: {data_dir}
test:
  - spec: adbench:faketask2
    data_dir: {data_dir}
adapt: [zero_shot, leaves]
seeds: [0, 1]
latent_dim: 4
n_sum_components: 2
epochs: 10
baselines: [iforest, knn]
log_root: {log_root}
""")
    results = run_from_config(p)
    # 2 seeds × (1 source + 2 adaptation modes + 2 baselines)
    assert len(results) == 10
    assert {r["seed"] for r in results} == {0, 1}
    bl_rows = [r for r in results if r["role"] == "baseline"]
    assert {r["adaptation"] for r in bl_rows} == {"iforest", "knn"}
    assert all(r["auroc"] > 0.8 for r in bl_rows)
    # per-seed log folders + named summary written
    assert (log_root / "0" / "results.jsonl").exists()
    assert (log_root / "1" / "run.log").exists()
    summary = json.loads((log_root / "summary_cfg_test.json").read_text())
    assert all(e["n_seeds"] == 2 for e in summary["results"])
    held = [e for e in summary["results"] if e["adaptation"] == "zero_shot"]
    assert held and all(e["auroc_mean"] > 0.9 for e in held)


def test_run_from_config_dict_with_unknown_key():
    with pytest.raises(KeyError, match="Unknown config key"):
        run_from_config({"train": ["x"], "test": ["y"], "nope": 1})


def test_load_config_rejects_bad_vtree_method(tmp_path):
    p = write_yaml(tmp_path / "c.yaml", """
train: [adbench:a]
test: [adbench:b]
vtree_methods: [chow_liu, hyperbolic]
""")
    with pytest.raises(ValueError, match="hyperbolic"):
        load_config(p)


def test_run_from_config_vtree_methods_sweep(tmp_path):
    """The matched-budget vtree ablation: one full run per method, rows
    tagged with the method, NLL aggregated, baselines not re-run."""
    data_dir = make_fake_adbench(tmp_path)
    log_root = tmp_path / "logs"
    p = write_yaml(tmp_path / "abl.yaml", f"""
name: vtree_abl
train:
  - spec: adbench:faketask
    data_dir: {data_dir}
test:
  - spec: adbench:faketask2
    data_dir: {data_dir}
adapt: [zero_shot]
seeds: [0]
latent_dim: 4
n_sum_components: 2
epochs: 6
vtree_methods: [chow_liu, spectral, orc, random]
baselines: [knn]
log_root: {log_root}
""")
    results = run_from_config(p)
    pc_rows = [r for r in results if r["role"] != "baseline"]
    # 4 methods × (1 source + 1 held_out zero_shot)
    assert len(pc_rows) == 8
    assert {r["vtree"] for r in pc_rows} == {"chow_liu", "spectral", "orc", "random"}
    assert all("nll" in r for r in pc_rows)
    # baselines are vtree-independent: run once per seed, not per method
    assert sum(r["role"] == "baseline" for r in results) == 1
    summary = json.loads((log_root / "summary_vtree_abl.json").read_text())
    held = [e for e in summary["results"] if e["role"] == "held_out"]
    assert {e["vtree"] for e in held} == {"chow_liu", "spectral", "orc", "random"}
    assert all("nll_mean" in e for e in held)


# ─── SLURM array hooks: --seed override + aggregate_from_logs ────────────────

def _array_style_config(tmp_path):
    data_dir = make_fake_adbench(tmp_path)
    log_root = tmp_path / "logs"
    return write_yaml(tmp_path / "exp.yaml", f"""
name: array_test
train:
  - spec: adbench:faketask
    data_dir: {data_dir}
test:
  - spec: adbench:faketask2
    data_dir: {data_dir}
adapt: [zero_shot]
seeds: [0, 1, 2]
latent_dim: 4
n_sum_components: 2
epochs: 8
log_root: {log_root}
"""), log_root


def test_seed_override_runs_single_seed(tmp_path):
    p, log_root = _array_style_config(tmp_path)
    results = run_from_config(p, seed_override=1)
    assert {r["seed"] for r in results} == {1}
    assert (log_root / "1" / "results.jsonl").exists()
    assert not (log_root / "0").exists()
    # per-task summary does not clobber the final one
    assert (log_root / "summary_array_test_seed1.json").exists()


def test_aggregate_from_logs_after_array_tasks(tmp_path):
    from src.experiment import aggregate_from_logs

    p, log_root = _array_style_config(tmp_path)
    # simulate three independent array tasks, one seed each
    for s in (0, 1, 2):
        run_from_config(p, seed_override=s)
    results = aggregate_from_logs(p)
    assert {r["seed"] for r in results} == {0, 1, 2}
    summary = json.loads((log_root / "summary_array_test.json").read_text())
    held = [e for e in summary["results"] if e["role"] == "held_out"]
    assert held and all(e["n_seeds"] == 3 for e in held)


def test_aggregate_from_logs_without_runs_raises(tmp_path):
    from src.experiment import aggregate_from_logs

    p, _ = _array_style_config(tmp_path)
    with pytest.raises(RuntimeError, match="No logged results"):
        aggregate_from_logs(p)
