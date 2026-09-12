"""Contracts for the paired comparisons and the one-command report (step 7).

The report exists so nobody has to choose which rows to show. These tests hold
it to that: an incomplete run is refused rather than partially absorbed, a
mismatched pair is an error rather than a silent comparison of two different
experiments, and identical inputs give a delta of exactly zero.
"""
import json
import os

import numpy as np
import pytest

from poc.time_series.compare_diagnosis import compare
from poc.time_series.diagnosis import (cluster_bootstrap, paired_engine_bootstrap,
                                       paired_localization_bootstrap)
from poc.time_series.report_diagnosis import (build_tables, generate,
    quality_cost_frontier, run_inventory)


CHANNELS = 3


def _evidence(rng, n=40, shift=0.0):
    labels = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    units = np.tile(np.repeat(np.arange(5), n // 10), 2)
    attr = rng.normal(size=(n, CHANNELS))
    attr[labels == 1, 0] += shift
    affected = np.zeros((n, CHANNELS), dtype=bool)
    affected[labels == 1, 0] = True
    scores = attr.max(1)
    # Both views, as the stage writes them: the artifact carries every score
    # view and its attribution matrix, and a fixture with only one of them
    # cannot catch a report that reads the wrong one.
    return {"labels": labels, "units": units,
            "kinds": np.array(["normal"] * (n // 2) + ["desync"] * (n // 2)),
            "observed": np.ones((n, CHANNELS), dtype=bool),
            "pair_index": np.tile(np.arange(n // 2), 2),
            "input_windows": rng.normal(size=(n, CHANNELS * 2)),
            "affected": affected,
            "relational_max": scores, "conditional_max": scores,
            "attr_R": attr, "attr_conditional": attr,
            "alarm_relational_max": scores > 0.5,
            "alarm_conditional_max": scores > 0.5}


def _run_dir(tmp_path, name, evidence_list, status="ok", diagnosis="ok"):
    rundir = tmp_path / name / "seed0"
    (rundir / "artifacts").mkdir(parents=True)
    (rundir / "status.json").write_text(json.dumps(
        {"status": status, "stages": {"diagnosis": diagnosis}, "seed": 0, "wall_s": 1.0}))
    (rundir / "config.json").write_text(json.dumps({"variant": name}))
    rows = []
    for i, (tag, evidence) in enumerate(evidence_list):
        np.savez_compressed(rundir / "artifacts" / f"{tag}.npz", **evidence)
        rows += [
            {"seed": 0, "stage": "diagnosis", "variant": name,
             "method": f"query full | {'circuit' if 'gaussian' not in tag else 'gaussian_fitted'}",
             "method_name": "circuit" if "gaussian" not in tag else "gaussian_fitted",
             "mask": "full", "query_s": 0.01 * (i + 1), "warmup_s": 0.001,
             "cost_passes": 2, "cost_node_evaluations": 100, "n_observed": 3.0},
            {"seed": 0, "stage": "diagnosis", "variant": name,
             "method": f"relational_max | full | {'circuit' if 'gaussian' not in tag else 'gaussian_fitted'}",
             "method_name": "circuit" if "gaussian" not in tag else "gaussian_fitted",
             "mask": "full", "score": "relational_max", "auroc": 0.7 + 0.05 * i,
             "loc_ap": 0.6, "end_to_end_unique_top1": 0.5, "alpha": 0.1,
             "numerically_flat": False, "threshold_infinite": bool(i)},
            {"seed": 0, "stage": "diagnosis", "variant": name, "method": "model",
             "parameters": 60, "fit_s": 1.0, "optimizer_steps": 12, "best_epoch": 3},
        ]
    with open(rundir / "results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (rundir / "history_pc_selection_loss.csv").write_text(
        "epoch,value\n0,3.0\n1,2.5\n2,2.4\n")
    return rundir


# ── paired statistics ───────────────────────────────────────────────────────

def test_identical_inputs_give_exactly_zero_on_every_endpoint():
    rng = np.random.default_rng(0)
    e = _evidence(rng, shift=1.5)
    auc = paired_engine_bootstrap(e["relational_max"], e["relational_max"],
                                  e["labels"], e["units"], reps=25)
    assert auc["auroc_delta"] == auc["ci_low"] == auc["ci_high"] == 0
    for metric in ("loc_ap", "end_to_end_unique_top1"):
        out = paired_localization_bootstrap(
            e["attr_R"], e["attr_R"], e["affected"], e["observed"],
            e["alarm_relational_max"], e["alarm_relational_max"], e["units"],
            metric=metric, reps=25)
        assert out[f"{metric}_delta"] == 0
        assert out["ci_low"] == out["ci_high"] == 0


def test_a_genuinely_better_localiser_shows_a_positive_interval():
    rng = np.random.default_rng(1)
    e = _evidence(rng, shift=4.0)
    worse = e["attr_R"][:, ::-1].copy()                 # points at the wrong channel
    out = paired_localization_bootstrap(
        e["attr_R"], worse, e["affected"], e["observed"],
        e["alarm_relational_max"], e["alarm_relational_max"], e["units"],
        metric="loc_ap", reps=200, seed=0)
    assert out["loc_ap_delta"] > 0
    assert out["ci_low"] > 0


def test_the_bootstrap_resamples_engines_not_windows():
    units = np.repeat(np.arange(4), 10)
    seen = []
    cluster_bootstrap(lambda idx: seen.append(len(idx)) or 1.0, units, reps=5)
    assert set(seen) == {40}                            # whole engines, every time
    with pytest.raises(ValueError, match=">=2 units"):
        cluster_bootstrap(lambda idx: 1.0, np.zeros(10), reps=5)


# ── pairing refusals ────────────────────────────────────────────────────────

def test_mismatched_and_incomplete_artifacts_are_refused(tmp_path):
    rng = np.random.default_rng(2)
    a = _evidence(rng, shift=2.0)
    b = dict(a)
    ok = _run_dir(tmp_path, "ok", [("diagnosis_mask0", a),
                                   ("diagnosis_gaussian_fitted_mask0", b)])
    art = ok / "artifacts"
    assert compare(art / "diagnosis_mask0.npz",
                   art / "diagnosis_gaussian_fitted_mask0.npz",
                   reps=20)["auroc_delta"] == 0
    different = dict(a)
    different["labels"] = 1 - a["labels"]
    np.savez_compressed(art / "diagnosis_other_mask0.npz", **different)
    with pytest.raises(ValueError, match="unpaired artifacts"):
        compare(art / "diagnosis_mask0.npz", art / "diagnosis_other_mask0.npz", reps=20)
    broken = _run_dir(tmp_path, "broken", [("diagnosis_mask0", a)],
                      status="failed", diagnosis="failed")
    with pytest.raises(ValueError, match="completed diagnosis stage"):
        compare(broken / "artifacts" / "diagnosis_mask0.npz",
                broken / "artifacts" / "diagnosis_mask0.npz", reps=20)


def test_localization_comparison_refuses_artifacts_without_the_matrices(tmp_path):
    rng = np.random.default_rng(3)
    e = _evidence(rng)
    thin = {k: v for k, v in e.items() if not k.startswith(("attr_", "alarm_"))}
    run = _run_dir(tmp_path, "thin", [("diagnosis_mask0", thin),
                                      ("diagnosis_gaussian_fitted_mask0", thin)])
    with pytest.raises(ValueError, match="attr_conditional is absent"):
        compare(run / "artifacts" / "diagnosis_mask0.npz",
                run / "artifacts" / "diagnosis_gaussian_fitted_mask0.npz",
                metric="loc_ap", reps=20)


# ── the report ──────────────────────────────────────────────────────────────

def test_report_names_every_refused_run_and_uses_none_of_their_rows(tmp_path):
    rng = np.random.default_rng(4)
    e = _evidence(rng, shift=2.0)
    _run_dir(tmp_path, "good", [("diagnosis_mask0", e),
                                ("diagnosis_gaussian_fitted_mask0", dict(e))])
    _run_dir(tmp_path, "crashed", [("diagnosis_mask0", e)],
             status="failed", diagnosis="failed")
    inventory = run_inventory(str(tmp_path))
    assert inventory["n_runs"] == 2 and inventory["n_usable"] == 1
    assert inventory["refused"] == ["crashed/seed0"]
    report = generate(str(tmp_path), reps=20, plots=False)
    assert report["inventory"]["n_refused"] == 1
    assert "crashed/seed0" in report["markdown"]
    assert {r["variant"] for r in report["tables"]["detection"]} == {"good"}
    for name in ("report.md", "report.json", "detection.csv", "paired.csv"):
        assert os.path.exists(os.path.join(report["out_dir"], name))
    # the paired section is generated exhaustively, not requested row by row
    assert {p["metric"] for p in report["paired"]} == {
        "auroc", "loc_ap", "end_to_end_unique_top1"}


def test_report_refuses_a_root_with_nothing_complete(tmp_path):
    rng = np.random.default_rng(5)
    _run_dir(tmp_path, "crashed", [("diagnosis_mask0", _evidence(rng))],
             status="failed", diagnosis="failed")
    with pytest.raises(ValueError, match="no run finished"):
        generate(str(tmp_path), reps=10, plots=False)
    with pytest.raises(FileNotFoundError):
        generate(str(tmp_path / "nowhere"))


def test_flag_fractions_survive_aggregation(tmp_path):
    """`threshold_infinite` is a bool, so plain numeric aggregation drops it —
    and dropping it is how an infinite threshold disappears from a table."""
    rng = np.random.default_rng(6)
    e = _evidence(rng)
    run = _run_dir(tmp_path, "good", [("diagnosis_mask0", e),
                                      ("diagnosis_gaussian_fitted_mask0", dict(e))])
    from poc.time_series.ts_logging import read_results
    tables = build_tables(read_results(str(tmp_path)))
    assert any("threshold_infinite_frac" in row for row in tables["detection"])


def test_quality_cost_points_join_within_one_run(tmp_path):
    rng = np.random.default_rng(7)
    e = _evidence(rng)
    _run_dir(tmp_path, "good", [("diagnosis_mask0", e),
                                ("diagnosis_gaussian_fitted_mask0", dict(e))])
    from poc.time_series.ts_logging import read_results
    points = quality_cost_frontier(read_results(str(tmp_path)))
    assert points and all("auroc" in p and "query_s" in p for p in points)
    assert {p["method_name"] for p in points} == {"circuit", "gaussian_fitted"}
