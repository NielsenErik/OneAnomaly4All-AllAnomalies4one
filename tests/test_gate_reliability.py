"""The FD001 reliability gate: it has to fail the run that motivated it.

A gate written after the fact can always be made to pass. This one is checked
against the recorded shape of `diagnosis_fd001_candidate` — five seeds of one
architecture at 20.2-59.1 nats, two collapsed and two at the epoch ceiling —
and against the shape that would count as fixed.
"""
import json

import pytest

from poc.time_series.gate_reliability import gate


def _study(tmp_path, variant, rows):
    for i, row in enumerate(rows):
        run = tmp_path / variant / f"seed{i}"
        (run).mkdir(parents=True)
        (run / "status.json").write_text(json.dumps(
            {"status": "ok", "stages": {"diagnosis": "ok"}, "seed": i}))
        line = {"stage": "diagnosis", "method": "model", "variant": variant,
                "epochs_budget": 600, "min_epochs": 100, "patience": 60,
                **row}
        (run / "results.jsonl").write_text(json.dumps(line) + "\n")
    return tmp_path


def _row(nll, best_epoch, stopped_early=True):
    return {"checkpoint_nll": nll, "best_epoch": best_epoch,
            "stopped_early": stopped_early, "restarts_run": 1}


def test_the_recorded_candidate_fails_on_spread_collapse_and_ceiling(tmp_path):
    # The five FD001 seeds as they were actually logged.
    r = _study(tmp_path, "chow_liu_leaf6", [
        _row(25.70, 435), _row(59.06, 80), _row(20.18, 596, stopped_early=False),
        _row(27.23, 599, stopped_early=False), _row(35.97, 129)])
    out = gate(str(r))["variants"]["chow_liu_leaf6"]
    assert not out["passes"]
    assert out["spread"] == pytest.approx(38.88, abs=.01)
    assert len(out["collapsed"]) == 2      # best epoch inside min_epochs+patience
    assert len(out["at_ceiling"]) == 2     # ended by the budget, not by patience
    assert gate(str(r))["passes"] is False


def test_a_reliable_study_passes_and_the_spread_threshold_binds(tmp_path):
    r = _study(tmp_path, "restarts4_cosine",
               [_row(21.1, 300), _row(22.4, 355), _row(23.0, 412)])
    assert gate(str(r))["passes"]
    assert not gate(str(r), spread_max=1.0)["passes"]


def test_tail_improvement_is_diagnostic_and_never_moves_the_verdict(tmp_path):
    # A run that keeps descending under a decayed learning rate looks exactly
    # like a truncated one to the ceiling test, which is why the tail is
    # reported beside it — and why it must not change the verdict.
    r = _study(tmp_path, "restarts4_cosine",
               [_row(21.0, 599, stopped_early=False)] * 3)
    curve = "\n".join(f"{i},{100 - i * 0.1:.4f}" for i in range(600))
    for seed in range(3):
        (r / "restarts4_cosine" / f"seed{seed}" / "history_pc_val_nll.csv"
         ).write_text("epoch,value\n" + curve + "\n")
    out = gate(str(r))["variants"]["restarts4_cosine"]
    assert all(t is not None and t < 0 for t in out["tail_improvement_nats"])
    assert not out["passes"] and len(out["at_ceiling"]) == 3
    # No curve on disk is reported as unknown, not as zero.
    bare = _study(tmp_path / "bare", "restarts4", [_row(21.0, 300)])
    assert gate(str(bare))["variants"]["restarts4"]["tail_improvement_nats"] == [None]


def test_unfinished_runs_are_not_evidence(tmp_path):
    r = _study(tmp_path, "restarts4", [_row(21.0, 300), _row(22.0, 310)])
    (r / "restarts4" / "seed1" / "status.json").write_text(json.dumps(
        {"status": "failed", "stages": {"diagnosis": "failed"}, "seed": 1}))
    assert gate(str(r))["variants"]["restarts4"]["n"] == 1
    with pytest.raises(SystemExit):
        gate(str(r), variant="does_not_exist")
