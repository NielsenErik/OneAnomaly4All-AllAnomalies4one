"""Contracts for the frozen confirmation (roadmap step 8) and the oracle arm.

The confirmation machinery has one job that the rest of the package deliberately
does not have: to make it mechanically impossible to choose the endpoint after
seeing the numbers.  Every test here is about that, not about accuracy.
"""
import json
import os

import numpy as np
import pytest
import torch
import yaml

from poc.time_series import confirm_diagnosis as cd
from poc.time_series.confirm_diagnosis import ProtocolError
from poc.time_series.diagnosis import OracleMethod
from poc.time_series.diagnosis_controls import (control_covariance,
                                                gaussian_oracle_map)


# ═══════════════════════════════════════════════════════════════════════════
# The oracle as a scored method
# ═══════════════════════════════════════════════════════════════════════════

def test_oracle_method_reproduces_the_reference_map_and_reports_a_cost():
    cov = control_covariance(3, 3, cross=.7, temporal=.5)
    x = torch.randn(16, 9, generator=torch.Generator().manual_seed(0)).double()
    method = OracleMethod(cov, 3)
    mask = torch.tensor([True, True, False])
    out = method.diagnosis_map(x, mask)
    truth = gaussian_oracle_map(x, cov, 3, mask)
    for key in ("log_px", "marginal", "conditional", "R"):
        assert torch.allclose(out[key], truth[key], equal_nan=True)
    cost = out["cost"].as_dict()
    assert cost["passes"] == 3 and cost["n_masks"] == 1


def test_oracle_method_refuses_a_per_row_mask_rather_than_inventing_a_law():
    cov = control_covariance(2, 3, cross=.7, temporal=.5)
    x = torch.randn(8, 6, generator=torch.Generator().manual_seed(1)).double()
    per_row = torch.ones(8, 3, dtype=torch.bool)
    per_row[0, 0] = False
    with pytest.raises(ValueError, match="shared channel pattern"):
        OracleMethod(cov, 3).diagnosis_map(x, per_row)


def test_oracle_singleton_mask_has_exactly_zero_relational_score():
    cov = control_covariance(3, 3, cross=.8, temporal=.4)
    x = torch.randn(12, 9, generator=torch.Generator().manual_seed(2)).double()
    out = OracleMethod(cov, 3).diagnosis_map(x, torch.tensor([True, False, False]))
    observed = out["mask"]
    assert float(out["R"][observed].abs().max()) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Freezing
# ═══════════════════════════════════════════════════════════════════════════

def _protocol(tmp_path, **overrides):
    block = {"design": "noninferiority", "candidate": "blocked",
             "comparator": "gaussian_fitted", "primary_metric": "loc_ap",
             "score": "relational_max", "margin": 0.02, "alpha": 0.05,
             "sampling_unit": "engine", "mask_workload": ["full"],
             "primary_mask": "full", "speed_factor": 2.0,
             "fault_mechanisms": ["desync"], "n_units_planned": 10,
             "pilot_sd": 0.05, "bootstrap_reps": 50,
             "resource_ceiling": {"max_query_s": 1.0}, "tie_rule": "ties count as failures",
             "preprocessing": "per-channel standardisation on training units",
             "dataset": "FD003"}
    block.update(overrides)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(
        {"name": "test_confirmation", "log_root": str(tmp_path / "logs"),
         "confirmation": block}, sort_keys=False))
    return str(path)


def test_freeze_stamps_a_digest_and_is_idempotent(tmp_path):
    path = _protocol(tmp_path)
    first = cd.freeze(path)
    assert first["changed"] and len(first["digest"]) == 64
    second = cd.freeze(path)
    assert not second["changed"] and second["digest"] == first["digest"]


def test_editing_a_frozen_protocol_voids_it(tmp_path):
    path = _protocol(tmp_path)
    cd.freeze(path)
    cfg = yaml.safe_load(open(path))
    cfg["confirmation"]["margin"] = 0.10          # the classic post-hoc edit
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    with pytest.raises(ProtocolError, match="changed after freezing"):
        cd.check_frozen(cd.load_protocol(path), path)
    with pytest.raises(ProtocolError, match="already frozen"):
        cd.freeze(path)
    info = cd.freeze(path, refreeze=True)
    assert info["version"] == 2
    block = yaml.safe_load(open(path))["confirmation"]
    assert block["history"][0]["version"] == 1      # the old digest is kept


def test_an_unfrozen_protocol_cannot_be_evaluated(tmp_path):
    path = _protocol(tmp_path)
    with pytest.raises(ProtocolError, match="never frozen"):
        cd.check_frozen(cd.load_protocol(path), path)


def test_an_incomplete_frozen_block_is_refused(tmp_path):
    path = _protocol(tmp_path)
    cfg = yaml.safe_load(open(path))
    del cfg["confirmation"]["margin"]
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    with pytest.raises(ProtocolError, match="incomplete"):
        cd.load_protocol(path)


def test_digest_ignores_the_stamp_but_not_the_content(tmp_path):
    path = _protocol(tmp_path)
    block = yaml.safe_load(open(path))["confirmation"]
    base = cd.digest(block)
    assert cd.digest({**block, "frozen_at": "later", "version": 9}) == base
    assert cd.digest({**block, "alpha": 0.20}) != base


# ═══════════════════════════════════════════════════════════════════════════
# The decision rule
# ═══════════════════════════════════════════════════════════════════════════

def _evidence(delta, low, high, mask="full"):
    return {"metric": "loc_ap", "score": "relational_max", "candidate": "blocked",
            "comparator": "gaussian_fitted", "bootstrap_reps": 50, "n_pairs_used": 1,
            "n_refused": 0, "aggregation": "frozen primary mask",
            "primary": {"mask": mask, "delta": delta, "ci_low": low,
                        "ci_high": high, "n_units": 10},
            "all_pairs": [], "refusals": []}


def _latency(factor=3.0, query_s=0.01):
    return {"per_mask": [{"mask": "full", "circuit_query_s": query_s,
                          "comparator_query_s": query_s * factor,
                          "speed_factor": factor, "n_circuit": 1, "n_comparator": 1}],
            "worst_speed_factor": factor, "note": ""}


def _operational(q95=0.03, alpha=0.05):
    return {"alpha": alpha, "n_rows": 1, "trial_fpr_q95_max": q95,
            "trial_power_mean": 0.5, "infinite_threshold_frac_max": 0.0,
            "observed_fpr_max": 0.02, "measured": True, "note": ""}


def test_noninferiority_needs_both_the_margin_and_the_speed(tmp_path):
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    good = cd.decide(block, _evidence(-0.005, -0.015, 0.01), _latency(3.0), _operational())
    assert good["verdict"] == "CONFIRMED"
    slow = cd.decide(block, _evidence(-0.005, -0.015, 0.01), _latency(1.2), _operational())
    assert slow["verdict"] == "NOT MET"
    wide = cd.decide(block, _evidence(-0.005, -0.030, 0.01), _latency(3.0), _operational())
    assert wide["verdict"] == "NOT MET"


def test_a_dominating_comparator_is_reported_as_a_negative_result(tmp_path):
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    out = cd.decide(block, _evidence(-0.08, -0.12, -0.04), _latency(3.0), _operational())
    assert out["verdict"] == "NEGATIVE"
    assert "dominates" in out["reading"]


def test_the_false_alarm_condition_can_sink_a_good_localiser(tmp_path):
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    out = cd.decide(block, _evidence(0.05, 0.01, 0.09), _latency(4.0),
                    _operational(q95=0.30))
    assert out["verdict"] == "NOT MET"
    failed = [c["name"] for c in out["conditions"] if not c["met"]]
    assert failed == ["operational_false_alarm"]


def test_an_unmeasured_false_alarm_rate_is_not_a_pass(tmp_path):
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    op = {**_operational(), "trial_fpr_q95_max": None, "measured": False}
    out = cd.decide(block, _evidence(-0.001, -0.01, 0.01), _latency(3.0), op)
    assert out["verdict"] == "NOT MET"


def test_superiority_requires_the_point_the_interval_and_the_ceiling(tmp_path):
    block = yaml.safe_load(open(_protocol(tmp_path, design="superiority", margin=0.03)))["confirmation"]
    ok = cd.decide(block, _evidence(0.05, 0.01, 0.09), _latency(1.0, 0.5), _operational())
    assert ok["verdict"] == "CONFIRMED"
    touching_zero = cd.decide(block, _evidence(0.05, -0.001, 0.09), _latency(1.0, 0.5),
                              _operational())
    assert touching_zero["verdict"] == "NOT MET"
    small = cd.decide(block, _evidence(0.01, 0.001, 0.02), _latency(1.0, 0.5), _operational())
    assert small["verdict"] == "NOT MET"
    over_budget = cd.decide(block, _evidence(0.05, 0.01, 0.09), _latency(1.0, 5.0),
                            _operational())
    assert over_budget["verdict"] == "NOT MET"


# ═══════════════════════════════════════════════════════════════════════════
# Evidence collection
# ═══════════════════════════════════════════════════════════════════════════

def _write_run(root, variant, seed, masks, comparator="gaussian_fitted",
               status="ok", delta=0.0):
    """A minimal completed run: status, results and paired mask artifacts."""
    run = os.path.join(root, variant, f"seed{seed}")
    art = os.path.join(run, "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(run, "status.json"), "w") as f:
        json.dump({"status": status, "stages": {"diagnosis": status},
                   "attempt": "t0"}, f)
    rng = np.random.default_rng(seed)
    rows = []
    for index, name in enumerate(masks):
        n, c = 40, 3
        units = np.repeat(np.arange(10), 4)
        labels = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
        shared = {"mask_name": np.asarray(name), "mask_index": np.asarray(index),
                  "mask_mechanism": np.asarray("externally_fixed_shared"),
                  "labels": labels, "units": units,
                  "kinds": np.asarray(["clean"] * n),
                  "observed": np.ones((n, c), dtype=bool),
                  "pair_index": np.arange(n), "input_windows": rng.normal(size=(n, c * 2)),
                  "affected": np.tile(np.array([True, False, False]), (n, 1))}
        base = labels + rng.normal(scale=.1, size=n)
        attr = rng.normal(size=(n, c))
        for who, path, bump in (("circuit", f"diagnosis_mask{index}.npz", delta),
                                (comparator, f"diagnosis_{comparator}_mask{index}.npz", 0.0)):
            scores = base + bump
            this_attr = attr.copy()
            this_attr[:, 0] += bump * 5
            np.savez_compressed(
                os.path.join(art, path), **shared,
                relational_max=scores, conditional_max=scores, marginal_max=scores,
                joint=scores, independent=scores,
                attr_R=this_attr, attr_conditional=this_attr, attr_marginal=this_attr,
                alarm_relational_max=scores > .5, alarm_conditional_max=scores > .5,
                alarm_marginal_max=scores > .5, alarm_joint=scores > .5,
                alarm_independent=scores > .5)
            rows.append({"stage": "diagnosis", "variant": variant, "seed": seed,
                         "method": f"query {name} | {who}", "method_name": who,
                         "mask": name, "query_s": 0.01 if who == "circuit" else 0.05})
            rows.append({"stage": "diagnosis", "variant": variant, "seed": seed,
                         "method": f"relational_max | {name} | {who}",
                         "method_name": who, "mask": name, "score": "relational_max",
                         "fpr": 0.02, "trial_fpr_q95": 0.04, "trial_power_mean": 0.6,
                         "trial_infinite_threshold_frac": 0.0})
    with open(os.path.join(run, "results.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return run


def test_evidence_uses_only_the_frozen_workload_and_records_refusals(tmp_path):
    root = str(tmp_path / "logs")
    _write_run(root, "blocked", 11, ["full", "random1:0"], delta=0.3)
    path = _protocol(tmp_path)
    cd.freeze(path)
    block = yaml.safe_load(open(path))["confirmation"]
    evidence = cd.primary_evidence(root, block, reps=20)
    assert evidence["n_pairs_used"] == 1                  # random1:0 is outside it
    assert evidence["primary"]["mask"] == "full"
    assert evidence["primary"]["delta"] >= 0


def test_a_failed_run_is_a_refusal_not_a_silent_drop(tmp_path):
    root = str(tmp_path / "logs")
    _write_run(root, "blocked", 11, ["full"], delta=0.3)
    _write_run(root, "blocked", 12, ["full"], status="failed")
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    evidence = cd.primary_evidence(root, block, reps=20)
    assert evidence["n_pairs_used"] == 1 and evidence["n_refused"] == 1
    assert "status=failed" in json.dumps(evidence["refusals"])


def test_a_missing_comparator_artifact_is_refused(tmp_path):
    root = str(tmp_path / "logs")
    run = _write_run(root, "blocked", 11, ["full"])
    os.remove(os.path.join(run, "artifacts", "diagnosis_gaussian_fitted_mask0.npz"))
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    with pytest.raises(ProtocolError, match="no paired"):
        cd.primary_evidence(root, block, reps=20)


def test_an_artifact_without_a_mask_name_cannot_pass_the_workload_check(tmp_path):
    root = str(tmp_path / "logs")
    run = _write_run(root, "blocked", 11, ["full"])
    path = os.path.join(run, "artifacts", "diagnosis_mask0.npz")
    with np.load(path, allow_pickle=False) as z:
        kept = {k: z[k] for k in z.files if k != "mask_name"}
    np.savez_compressed(path, **kept)
    block = yaml.safe_load(open(_protocol(tmp_path)))["confirmation"]
    with pytest.raises(ProtocolError, match="no paired"):
        cd.primary_evidence(root, block, reps=20)


def test_worst_mask_aggregation_cannot_be_improved_by_adding_an_easy_mask(tmp_path):
    root = str(tmp_path / "logs")
    _write_run(root, "blocked", 11, ["full", "random1:0"], delta=0.3)
    block = yaml.safe_load(open(_protocol(
        tmp_path, mask_workload=["full", "random1:0"], primary_mask=None)))["confirmation"]
    block.pop("primary_mask")
    evidence = cd.primary_evidence(root, block, reps=20)
    lows = [e["ci_low"] for e in evidence["all_pairs"] if not e.get("refused")]
    assert evidence["primary"]["ci_low"] == min(lows)


def test_evaluate_writes_a_release_bundle_with_the_frozen_protocol(tmp_path):
    root = str(tmp_path / "logs")
    _write_run(root, "blocked", 11, ["full"], delta=0.3)
    path = _protocol(tmp_path)
    cd.freeze(path)
    report = cd.evaluate(path, out_dir=str(tmp_path / "release"), reps=20)
    out = report["out_dir"]
    assert os.path.exists(os.path.join(out, "frozen_protocol.yaml"))
    assert os.path.exists(os.path.join(out, "confirmation.json"))
    text = open(os.path.join(out, "confirmation.md")).read()
    assert report["decision"]["verdict"] in text
    assert report["protocol"]["digest"][:16] in text
    # Provenance and refusal counts travel with the verdict, not in a footnote.
    assert "Run inventory" in text and "Operational false alarm" in text


# ═══════════════════════════════════════════════════════════════════════════
# The oracle appears exactly once per (mask, score)
# ═══════════════════════════════════════════════════════════════════════════

def test_the_scored_oracle_does_not_double_count_with_the_legacy_reference(tmp_path):
    """One method, one row per (mask, score).

    The stage has emitted known-law detection rows since before the oracle was
    a method.  With both alive, every oracle AUROC entered an aggregate twice
    and half the duplicates carried no threshold, localisation or cost column —
    a mean over that column silently averages a different set of runs than the
    mean beside it.
    """
    import copy
    from poc.time_series.config import DEFAULTS
    from poc.time_series.diagnosis import stage_diagnosis
    from poc.time_series.ts_logging import RunLogger

    cfg = copy.deepcopy(DEFAULTS)
    cfg.update({"name": "oracle_once", "variant": "blocked", "device": "cpu",
                "stages": ["diagnosis"]})
    cfg["dataset"].update({"source": "synthetic", "window": 3, "channels": 3})
    cfg["model"].update({"vtree": "channel_blocked", "K": 2, "leaf_components": 2,
                         "epochs": 2, "min_epochs": 1, "patience": 1, "batch_size": 64})
    cfg["eval"].update({"diagnosis_control": True, "control_train": 128,
                        "control_val": 96, "diagnosis_max_windows": 32,
                        "mask_ks": [1], "masks_per_k": 1, "oracle_check_windows": 4,
                        "plots": False})
    out = stage_diagnosis(cfg, 5, RunLogger(str(tmp_path / "run"), "oracle_once"))
    assert "known_law_oracle" in out["methods"]

    rows = [json.loads(line) for line in
            open(os.path.join(str(tmp_path / "run"), "results.jsonl"))]
    scored = [r for r in rows if r.get("method_name") == "known_law_oracle"
              and r.get("score") and r.get("mask")]
    assert scored, "the oracle produced no scored rows"
    keys = [(r["mask"], r["score"]) for r in scored]
    assert len(keys) == len(set(keys)), f"duplicated oracle rows: {keys}"
    # And the surviving row is the rich one, not the detection-only stub.
    for r in scored:
        assert r.get("score_range") is not None and r.get("localization_score")
