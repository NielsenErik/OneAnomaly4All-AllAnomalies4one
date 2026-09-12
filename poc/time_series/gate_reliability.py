"""Pre-registered reliability gate for the FD001 restart study.

Reads the model rows of a finished study and answers one question: is the
candidate's fit still decided by which initialisation it happened to draw?
Written before the run, and printed with every number it used, so a failure
cannot be re-read as a success afterwards.

    python -m poc.time_series.gate_reliability LOG_ROOT [--variant NAME]

Rows written before 2026-09-12 carry no `patience`/`min_epochs`, so on those
the collapse test cannot fire and the ceiling test falls back to "selected the
last budgeted epoch" — a legacy study reads as no worse than it was, never
better.

Criteria (from config/ts/diagnosis_fd001_reliability.yaml):
  spread    max - min checkpoint NLL over seeds  <  SPREAD (default 5 nats)
  collapse  no seed selects within `patience` of `min_epochs`
  ceiling   no seed selects its last budgeted epoch
"""
import argparse
import json
import os
import sys
from typing import Dict, List


def model_rows(root: str) -> List[Dict]:
    rows = []
    for dirpath, _, files in os.walk(root):
        if "results.jsonl" not in files:
            continue
        status = os.path.join(dirpath, "status.json")
        if os.path.exists(status):
            with open(status) as f:
                st = json.load(f)
            if st.get("stages", {}).get("diagnosis") != "ok":
                continue            # an unfinished run is not evidence
        with open(os.path.join(dirpath, "results.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("method") == "model" and "checkpoint_nll" in r:
                    r["run_dir"] = os.path.relpath(dirpath, root)
                    rows.append(r)
    return rows


def gate(root: str, variant=None, spread_max=5.0) -> Dict:
    rows = [r for r in model_rows(root)
            if variant is None or r.get("variant") == variant]
    if not rows:
        raise SystemExit(f"{root}: no completed model rows"
                         + (f" for variant {variant!r}" if variant else ""))
    by_variant: Dict[str, List[Dict]] = {}
    for r in rows:
        by_variant.setdefault(str(r.get("variant")), []).append(r)
    report = {"root": root, "spread_max": spread_max, "variants": {}}
    for name, rs in sorted(by_variant.items()):
        nlls = [float(r["checkpoint_nll"]) for r in rs]
        budget = [int(r.get("epochs_budget", 0)) for r in rs]
        best = [int(r.get("best_epoch", -1)) for r in rs]
        # A run that never triggered early stopping was ended by the BUDGET,
        # whatever epoch it selected: its fit is unfinished, and reading its
        # NLL as the architecture's is reading the ceiling instead.
        ceiling = [i for i, r in enumerate(rs)
                   if (int(r.get("patience", 0)) and not r.get("stopped_early"))
                   or (not int(r.get("patience", 0)) and budget[i]
                       and best[i] >= budget[i] - 1)]
        # A checkpoint selected inside the first `min_epochs + patience`
        # epochs is the collapse mode seen on FD001 (seeds 22 and 25): the run
        # stops improving almost immediately and early stopping then ends it.
        collapsed = [i for i, r in enumerate(rs)
                     if best[i] >= 0 and best[i] <= int(r.get("min_epochs", 1))
                     + int(r.get("patience", 0))]
        spread = max(nlls) - min(nlls)
        report["variants"][name] = {
            "n": len(rs), "checkpoint_nll": nlls, "spread": spread,
            "best_epochs": best, "epochs_budget": budget,
            "restarts_run": [int(r.get("restarts_run", 1)) for r in rs],
            "restarts_abandoned": [int(r.get("restarts_abandoned", 0)) for r in rs],
            "at_ceiling": [rs[i]["run_dir"] for i in ceiling],
            "collapsed": [rs[i]["run_dir"] for i in collapsed],
            "passes": spread < spread_max and not ceiling and not collapsed}
    report["passes"] = any(v["passes"] for v in report["variants"].values())
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--spread", type=float, default=5.0,
                    help="maximum checkpoint-NLL spread over seeds, in nats")
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args(argv)
    report = gate(args.root, args.variant, args.spread)
    for name, v in report["variants"].items():
        print(f"{name:24s} n={v['n']:2d}  spread {v['spread']:6.2f} nats  "
              f"(NLL {min(v['checkpoint_nll']):.2f}-{max(v['checkpoint_nll']):.2f})  "
              f"ceiling {len(v['at_ceiling'])}  collapsed {len(v['collapsed'])}  "
              f"-> {'PASS' if v['passes'] else 'FAIL'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    print("gate:", "PASS" if report["passes"] else "FAIL")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
