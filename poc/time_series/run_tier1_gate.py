"""
Tier 1.2 — read the kill gate and take the decision.

    PYTHONPATH=. python -m poc.time_series.runner config/ts/tier1_kill_gate.yaml
    PYTHONPATH=. python -m poc.time_series.run_tier1_gate logs/ts/tier1_kill_gate

The runner produces one `structure_gate` row per (structure, seed).  This
applies the PRE-REGISTERED budget — read from the config the runs recorded, not
from the command line — to the SEED AVERAGE, which is the number the decision is
actually taken on.  One seed is not a decision, and a budget typed at the
prompt after seeing the rows is not a pre-registration.

It prints, in this order:

  1. provenance — how many rows were refused because their stage did not
     complete (Tier 0.2).  A gate read off a partial batch is not a gate.
  2. the matched-parameter table, mean ± sd over seeds.
  3. the verdict, with the rule stated in the same breath as the numbers.

Exit status: 0 on PASS, 2 on FAIL, 3 on VOID (every arm factorised, so
there was nothing to compare), 4 on NO_COMPARATOR (every arm turned out to be
channel-blocked), 1 when the gate could not be read at all.
A FAIL is not an error — it is the answer, and the plan says what to do with
it: write up the negative result plus the benchmark, and do not build Tier
1.3-1.6 on top.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from .pipeline import gate_verdict
from .ts_logging import group_stats, provenance_report, read_results

VALUE_KEYS = ("auroc", "ap", "val_nll", "params", "K", "fit_s",
              "best_epoch", "dependence_nats")


def _config_of(root: str) -> Dict[str, Any]:
    """The resolved eval block the runs actually used (any one of them: the
    gate keys are not on the grid, so they are identical across runs)."""
    for dirpath, _, files in os.walk(root):
        if "config.json" in files:
            with open(os.path.join(dirpath, "config.json")) as f:
                return json.load(f)
    raise FileNotFoundError(f"no config.json under {root!r}: was the batch run?")


def read_gate(root: str) -> Dict[str, Any]:
    rows = [r for r in read_results(root, require_ok=True)
            if r.get("stage") == "structure_gate" and r.get("vtree")]
    if not rows:
        raise FileNotFoundError(
            f"no completed structure_gate rows under {root!r}. Run\n"
            "  PYTHONPATH=. python -m poc.time_series.runner "
            "config/ts/tier1_kill_gate.yaml")
    cfg = _config_of(root)
    ev = cfg["eval"]
    # `blocked` is a bool, so `group_stats` (numeric only) drops it — and the
    # rule needs it: the comparison set is the arms that are NOT blocked.  A
    # structure counts as blocked only if it was blocked in EVERY seed; a
    # learner whose vtree happens to be channel-contiguous for some seeds is
    # not a reliable blocked arm and belongs in the comparison set.
    blocked: Dict[str, bool] = {}
    for r in rows:
        v = r["vtree"]
        blocked[v] = blocked.get(v, True) and bool(r.get("blocked", False))
    stats = group_stats(rows, ["vtree"], VALUE_KEYS)
    stats.sort(key=lambda s: -s.get("auroc_mean", -np.inf))
    mean_rows = [{"vtree": s["vtree"], "auroc": s.get("auroc_mean", float("nan")),
                  "val_nll": s.get("val_nll_mean", float("nan")),
                  "params": s.get("params_mean", float("nan")),
                  "dependence_nats": s.get("dependence_nats_mean", float("nan")),
                  "blocked": blocked.get(s["vtree"], False),
                  "n_seeds": s["n_seeds"]} for s in stats]
    verdict = gate_verdict(mean_rows, ev)
    verdict["n_seeds"] = sorted({r.get("seed") for r in rows})
    verdict["pre_registered"] = {
        "gate_max_auroc_loss": ev["gate_max_auroc_loss"],
        "gate_max_nll_loss_frac": ev["gate_max_nll_loss_frac"],
        "gate_candidate": ev["gate_candidate"],
        "gate_reference": ev["gate_reference"],
        # the runs record the resolved config, not the path it came from, so
        # name the experiment when the path is absent — the point is that the
        # budget is READ BACK from what ran, never retyped here
        "config_path": cfg.get("config_path") or f"config/ts/{cfg.get('name')}.yaml",
    }
    return {"stats": stats, "verdict": verdict, "blocked": blocked,
            "provenance": provenance_report(root)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="logs/ts/tier1_kill_gate")
    ap.add_argument("--json", action="store_true", help="machine-readable only")
    args = ap.parse_args(argv)

    try:
        out = read_gate(args.root)
    except (FileNotFoundError, KeyError) as exc:
        print(f"cannot read the gate: {exc}")
        return 1

    if args.json:
        print(json.dumps(out, indent=2, default=float))
    else:
        p = out["provenance"]
        print(f"\nprovenance: {p['rows_used']}/{p['rows_total']} rows used, "
              f"{p['rows_dropped']} refused; runs {p['runs_by_status']}")
        if p["incomplete_runs"]:
            print(f"  incomplete: {p['incomplete_runs']}")
        blocked = out["blocked"]
        print(f"\n{'structure':<27}{'K':>4}{'params':>10}{'val NLL':>14}"
              f"{'val AUROC':>16}{'depend.':>10}{'seeds':>7}")
        print("-" * 88)
        for s in out["stats"]:
            mark = " ·blocked" if blocked.get(s["vtree"]) else ""
            print(f"{s['vtree'] + mark:<27}{s.get('K_mean', float('nan')):>4.0f}"
                  f"{s.get('params_mean', float('nan')):>10,.0f}"
                  f"{s.get('val_nll_mean', float('nan')):>9.3f}"
                  f" ±{s.get('val_nll_std', 0.0):<4.2f}"
                  f"{s.get('auroc_mean', float('nan')):>11.4f}"
                  f" ±{s.get('auroc_std', 0.0):<4.3f}"
                  f"{s.get('dependence_nats_mean', float('nan')):>10.3f}"
                  f"{s['n_seeds']:>7}")
        v = out["verdict"]
        print("\nrecomputed from the arm rows above, on the seed average; the "
              "per-seed VERDICT rows\nin results.jsonl are progress "
              "indicators, not the gate.")
        print(f"\nrule (pre-registered in {v['pre_registered']['config_path']}):"
              f"\n  {v['rule']}")
        print(f"\nGATE: {v['verdict']} — {v['reason']}")
        if v["verdict"] == "NO_COMPARATOR":
            print("\n  Every arm came out channel-blocked, so there is nothing "
                  "to measure the cost of\n  blocking against. Informative in "
                  "its own right — the learners chose the boundary —\n  but not "
                  "a PASS. Add an arm that provably splits a channel "
                  "(`time`) and re-run.")
        if v["verdict"] == "VOID":
            print("\n  Every arm is a product of per-channel marginals, so the "
                  "structure comparison is empty:\n  no vtree can matter to a "
                  "model that carries no cross-channel dependence, and every\n  "
                  "relational quantity downstream of it is identically zero. "
                  "Train longer (the\n  crossover is in gradient steps, ~20-40 "
                  "epochs on C-MAPSS) and re-run the gate.")
        if v["verdict"] == "FAIL":
            failed = ", ".join(v.get("failed_on") or ["the budget"])
            print(f"\n  Failed on: {failed}.")
            if v.get("nll_ok") and not v.get("auroc_ok"):
                # Say what actually happened.  "Does not hold density" would be
                # the opposite of the measurement in this case, and a stale
                # sentence quoted out of a log is how a wrong claim starts.
                print("  The blocked structure HOLDS density — it is better on "
                      "held-out NLL — and loses\n  DETECTION. Those point in "
                      "opposite directions, so read the dependence column "
                      "before\n  deciding: if the blocked arms also carry less "
                      "cross-channel dependence, the boundary\n  is costing "
                      "the model the very quantity the relational statistic is "
                      "computed from,\n  which is a deeper problem than the "
                      "AUROC gap.")
            elif v.get("auroc_ok"):
                print("  The blocked structure holds detection and loses "
                      "density.")
            else:
                print("  The blocked structure loses on both axes.")
            print("  Plan §1.2: the cheap-query story has nothing to sit on as "
                  "it stands — write up\n  the negative result plus the "
                  "benchmark, and do NOT read Tier 1.3-1.6 as findings on this\n"
                  "  structure. Any repair (a different blocked structure, a "
                  "different candidate) is a NEW\n  pre-registration on fresh "
                  "seeds, not a re-read of this one.")

    dest = os.path.join(args.root, "gate_verdict.json")
    try:
        with open(dest, "w") as f:
            json.dump(out, f, indent=2, default=float)
        if not args.json:
            print(f"\nwritten: {dest}")
    except OSError:
        pass
    # VOID is not a pass: the comparison never happened.  Distinct codes so a
    # launcher can tell "the structure lost" from "there was nothing to compare".
    return {"PASS": 0, "FAIL": 2, "VOID": 3,
            "NO_COMPARATOR": 4}.get(out["verdict"]["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
