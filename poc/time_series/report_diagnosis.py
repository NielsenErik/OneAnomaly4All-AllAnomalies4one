"""
One command, one report — roadmap step 7.

    PYTHONPATH=. python -m poc.time_series.report_diagnosis logs/ts/diagnosis_controls
    PYTHONPATH=. python -m poc.time_series.report_diagnosis logs/ts/diagnosis_fd001 \
        --out report/fd001 --reps 2000

It consumes COMPLETED artifacts and emits the tables, intervals, quality–cost
curves, convergence plots and failure counts. What it deliberately does not do
is let anyone choose a favourable row: the run inventory comes first, every
incomplete run is named, every method/mask/score cell that exists is printed,
and the paired comparisons are generated exhaustively from the artifacts rather
than requested one at a time.

Sections, and the question each one answers:

  inventory      what finished, what failed, what was refused, and why
  models         optimiser updates, selected epoch, wall time, parameters —
                 step 3's undertrained-versus-weak distinction
  detection      AUROC / AP per (method, mask, score), mean ± sd over seeds
  localisation   AP, ambiguity, abstention and end-to-end success in the same
                 table, so a localisation number is never read without the
                 ambiguity count beside it
  operational    calibrated false-alarm rate, its spread over repeated
                 calibration draws, power, and how often the honest threshold
                 was infinite
  accuracy       error against the known-law oracle, where a control provides one
  cost           measured query seconds, node-row work, cache bytes, output
                 size and parameters, per method and mask
  paired         circuit versus every fitted comparator on detection,
                 localisation AP and end-to-end success, engine-resampled

A tie, an undefined endpoint and an excluded run are three different things and
each is counted separately. Nothing here converts an exploratory table into a
confirmatory claim; the header of every generated file says so.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ts_logging import group_stats, provenance_report, read_results, run_status

HEADER = ("EXPLORATORY. Generated from completed validation-split runs. "
          "No PASS/FAIL, no confirmatory claim, and no endpoint here was "
          "frozen before the data were seen.")

DETECTION_COLUMNS = ["auroc", "ap", "fpr", "power", "score_range"]
LOCALIZATION_COLUMNS = ["loc_ap", "loc_ap_n", "loc_unique_top1_accuracy",
                        "end_to_end_unique_top1", "loc_n_ambiguous",
                        "loc_n_abstained_no_alarm", "loc_n_no_observed_target",
                        "loc_n_all_observed_are_targets"]
OPERATIONAL_COLUMNS = ["alpha", "fpr", "power", "threshold",
                       "trial_fpr_mean", "trial_fpr_q05", "trial_fpr_q95",
                       "trial_fpr_exceeds_alpha_frac", "trial_power_mean",
                       "trial_infinite_threshold_frac", "trial_calibration_objects",
                       "trajectory_auroc", "trajectory_fpr", "trajectory_power"]
COST_COLUMNS = ["query_s", "warmup_s", "cost_passes", "cost_node_evaluations",
                "cost_n_masks", "cost_output_elements", "cost_peak_boundary_bytes",
                "n_observed", "oracle_max_error"]
MODEL_COLUMNS = ["parameters", "nodes", "fit_s", "optimizer_steps", "epochs_run",
                 "best_epoch", "checkpoint_nll", "final_train_nll", "log_partition",
                 "lr", "epochs_budget"]
ACCURACY_COLUMNS = ["oracle_conditional_mae", "oracle_conditional_bias",
                    "oracle_conditional_corr", "oracle_marginal_mae",
                    "oracle_R_mae", "oracle_R_corr", "oracle_log_px_mae"]


# ═══════════════════════════════════════════════════════════════════════════
# Inventory
# ═══════════════════════════════════════════════════════════════════════════

def run_inventory(root: str) -> Dict[str, Any]:
    """Every run directory under `root`, with the reason it is in or out.

    A run that crashed after writing some rows is NOT partially usable: its
    checkpoint selection, calibration and evaluation may all be half-done, and
    a table that silently absorbs those rows is the failure this function
    exists to prevent.
    """
    runs: List[Dict[str, Any]] = []
    for dirpath, _, files in os.walk(root):
        if "status.json" not in files:
            continue
        status = run_status(dirpath) or {}
        stages = status.get("stages", {})
        usable = status.get("status") == "ok" and stages.get("diagnosis") == "ok"
        runs.append({"run_dir": os.path.relpath(dirpath, root),
                     "status": status.get("status", "missing"),
                     "diagnosis": stages.get("diagnosis", "absent"),
                     "seed": status.get("seed"), "wall_s": status.get("wall_s"),
                     "usable": usable,
                     "reason": "" if usable else
                               f"status={status.get('status')} diagnosis={stages.get('diagnosis', 'absent')}",
                     "error": str(status.get("error", ""))[:200]})
    runs.sort(key=lambda r: r["run_dir"])
    return {"runs": runs, "n_runs": len(runs),
            "n_usable": sum(r["usable"] for r in runs),
            "n_refused": sum(not r["usable"] for r in runs),
            "refused": [r["run_dir"] for r in runs if not r["usable"]],
            "provenance": provenance_report(root)}


# ═══════════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════════

def _select(rows: Iterable[dict], predicate) -> List[dict]:
    return [r for r in rows if r.get("stage") == "diagnosis" and predicate(r)]


# Boolean columns that must never be dropped from a table: `group_stats`
# excludes bools (a mean of True/False is not a mean of a measurement), but the
# FRACTION of runs in which a statistic was numerically flat or a threshold was
# infinite is exactly what stops a reader from taking the AUROC beside it at
# face value. They are aggregated separately and merged back in.
FLAG_COLUMNS = ("numerically_flat", "threshold_infinite", "mask_shared",
                "target_visible", "identifiable", "depends_on_values",
                "depends_on_fault", "trajectory_threshold_infinite")


def _flag_fractions(rows: List[dict], keys: Sequence[str]) -> Dict[tuple, dict]:
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out: Dict[tuple, dict] = {}
    for key, rs in groups.items():
        entry: Dict[str, Any] = {}
        for flag in FLAG_COLUMNS:
            values = [bool(r[flag]) for r in rs if isinstance(r.get(flag), bool)]
            if values:
                entry[f"{flag}_frac"] = float(np.mean(values))
        out[key] = entry
    return out


def _table(rows: List[dict], keys: Sequence[str], columns: Sequence[str]
           ) -> List[dict]:
    if not rows:
        return []
    out = group_stats(rows, list(keys), list(columns))
    flags = _flag_fractions(rows, keys)
    for entry in out:
        entry.update(flags.get(tuple(entry.get(k) for k in keys), {}))
    return sorted(out, key=lambda r: tuple(str(r.get(k)) for k in keys))


def build_tables(rows: List[dict]) -> Dict[str, List[dict]]:
    scored = _select(rows, lambda r: r.get("score") and r.get("mask"))
    queries = _select(rows, lambda r: str(r.get("method", "")).startswith("query "))
    models = _select(rows, lambda r: r.get("method") == "model"
                     or str(r.get("method", "")).startswith("model "))
    accuracy = _select(rows, lambda r: str(r.get("method", "")).startswith("known-law error"))
    shortcut = _select(rows, lambda r: r.get("method") == "shortcut audit")
    keys = ("variant", "method_name", "mask", "score")
    return {
        "models": _table(models, ("variant", "method"), MODEL_COLUMNS),
        "detection": _table(scored, keys, DETECTION_COLUMNS),
        "localization": _table(scored, keys, LOCALIZATION_COLUMNS),
        "operational": _table(scored, keys, OPERATIONAL_COLUMNS),
        "cost": _table(queries, ("variant", "method_name", "mask"), COST_COLUMNS),
        "accuracy": _table(accuracy, ("variant", "method_name", "mask"), ACCURACY_COLUMNS),
        "shortcut": _table(shortcut, ("variant",), ["shortcut_auroc", "shortcut_n_dev",
                                                    "shortcut_n_eval"]),
    }


def quality_cost_frontier(rows: List[dict], quality: str = "auroc",
                          cost: str = "query_s") -> List[dict]:
    """Join each (variant, method, mask) score row to the query that produced it.

    A quality–cost point is only meaningful when both numbers came from the
    same run and the same mask, so the join is on the run directory, not on a
    variant name that two different batches might share.
    """
    costs: Dict[Tuple, dict] = {}
    for r in _select(rows, lambda r: str(r.get("method", "")).startswith("query ")):
        costs[(r.get("run_dir"), r.get("method_name"), r.get("mask"))] = r
    out = []
    for r in _select(rows, lambda r: r.get("score") and r.get("mask")):
        c = costs.get((r.get("run_dir"), r.get("method_name"), r.get("mask")))
        if c is None or r.get(quality) is None or c.get(cost) is None:
            continue
        out.append({"variant": r.get("variant"), "method_name": r.get("method_name"),
                    "mask": r.get("mask"), "score": r.get("score"),
                    "seed": r.get("seed"), quality: r.get(quality), cost: c.get(cost),
                    "cost_node_evaluations": c.get("cost_node_evaluations"),
                    "parameters": c.get("parameters")})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Paired comparisons, generated exhaustively
# ═══════════════════════════════════════════════════════════════════════════

def paired_comparisons(root: str, inventory: Dict[str, Any], score: str,
                       reps: int, seed: int, metrics: Sequence[str]
                       ) -> List[Dict[str, Any]]:
    """Circuit versus every fitted comparator, within each completed run.

    Comparing inside a run is what makes the pair exact: identical windows,
    labels, corruptions and masks by construction. A pair whose artifacts do
    not match is reported as a refusal, never dropped silently.
    """
    from .compare_diagnosis import compare
    out: List[Dict[str, Any]] = []
    for run in inventory["runs"]:
        if not run["usable"]:
            continue
        art = os.path.join(root, run["run_dir"], "artifacts")
        if not os.path.isdir(art):
            continue
        circuits = sorted(f for f in os.listdir(art)
                          if f.startswith("diagnosis_mask") and f.endswith(".npz"))
        for name in circuits:
            index = name[len("diagnosis_mask"):-len(".npz")]
            others = sorted(f for f in os.listdir(art)
                            if f.endswith(f"_mask{index}.npz") and f != name)
            for other in others:
                for metric in metrics:
                    entry = {"run_dir": run["run_dir"], "seed": run["seed"],
                             "mask_index": index, "a": name, "b": other,
                             "score": score, "metric": metric}
                    try:
                        entry.update(compare(os.path.join(art, name),
                                             os.path.join(art, other),
                                             score=score, reps=reps, seed=seed,
                                             metric=metric))
                    except Exception as exc:                # refusal, recorded
                        entry.update({"refused": True, "reason": str(exc)[:200]})
                    out.append(entry)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Convergence curves and plots
# ═══════════════════════════════════════════════════════════════════════════

def read_histories(root: str, inventory: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for run in inventory["runs"]:
        if not run["usable"]:
            continue
        rundir = os.path.join(root, run["run_dir"])
        curves: Dict[str, List[float]] = {}
        for fname in sorted(os.listdir(rundir)):
            if not (fname.startswith("history_") and fname.endswith(".csv")):
                continue
            with open(os.path.join(rundir, fname)) as f:
                values = [float(r["value"]) for r in csv.DictReader(f)]
            if values:
                curves[fname[len("history_"):-len(".csv")]] = values
        if curves:
            out[run["run_dir"]] = curves
    return out


def write_plots(out_dir: str, histories, frontier) -> List[str]:
    """Convergence curves and the quality–cost scatter. Missing matplotlib is
    not a failure: the tables are the report, the plots are a convenience."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written = []
    if histories:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for run, curves in sorted(histories.items()):
            values = curves.get("pc_selection_loss") or curves.get("pc_val_nll")
            if values:
                ax.plot(range(len(values)), values, lw=1, label=run)
        ax.set_xlabel("epoch"); ax.set_ylabel("selection loss on checkpoint units")
        ax.set_title("convergence (checkpoint objective)")
        if len(histories) <= 12:
            ax.legend(fontsize=6)
        fig.tight_layout()
        path = os.path.join(out_dir, "convergence.png")
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)
    if frontier:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        methods = sorted({p["method_name"] for p in frontier})
        for method in methods:
            pts = [p for p in frontier if p["method_name"] == method]
            ax.scatter([p["query_s"] for p in pts], [p["auroc"] for p in pts],
                       s=14, alpha=.65, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("query seconds (measured, this implementation)")
        ax.set_ylabel("detection AUROC")
        ax.set_title("quality vs cost — one point per run/mask/score")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(out_dir, "quality_cost.png")
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)
    return written


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)


def render_markdown(name: str, tables, inventory, paired, frontier) -> str:
    lines = [f"# Diagnosis report — {name}", "", HEADER, "",
             "## Run inventory", "",
             f"- runs found: {inventory['n_runs']}",
             f"- usable (run ok and diagnosis ok): {inventory['n_usable']}",
             f"- refused: {inventory['n_refused']}"]
    for run in inventory["runs"]:
        if not run["usable"]:
            lines.append(f"  - `{run['run_dir']}` — {run['reason']} {run['error']}".rstrip())
    prov = inventory["provenance"]
    lines += ["", f"- result rows used: {prov['rows_used']} of {prov['rows_total']} "
                  f"({prov['rows_dropped']} dropped by provenance gating)", ""]
    for title, key in (("Models and optimisation", "models"),
                       ("Detection", "detection"),
                       ("Localisation", "localization"),
                       ("Operational alarms", "operational"),
                       ("Accuracy against the known law", "accuracy"),
                       ("Cost", "cost"),
                       ("Marginal-shortcut audit", "shortcut")):
        rows = tables.get(key) or []
        lines += [f"## {title}", ""]
        if not rows:
            lines += ["_no rows_", ""]
            continue
        columns = [c for c in rows[0] if c not in ("seeds",)]
        present = [c for c in columns if any(r.get(c) is not None for r in rows)]
        lines.append("| " + " | ".join(present) + " |")
        lines.append("|" + "|".join(["---"] * len(present)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(_fmt(r.get(c)) for c in present) + " |")
        lines.append("")
    lines += ["## Paired comparisons (circuit − comparator)", ""]
    if not paired:
        lines += ["_no paired artifacts found_", ""]
    else:
        lines.append("| run | mask | comparator | metric | delta | ci_low | ci_high | reps |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in paired:
            if p.get("refused"):
                lines.append(f"| {p['run_dir']} | {p['mask_index']} | {p['b']} | "
                             f"{p['metric']} | REFUSED | {p.get('reason', '')} | | |")
                continue
            delta = next((v for k, v in p.items() if k.endswith("_delta")), None)
            lines.append(f"| {p['run_dir']} | {p['mask_index']} | {p['b']} | {p['metric']} | "
                         f"{_fmt(delta)} | {_fmt(p.get('ci_low'))} | "
                         f"{_fmt(p.get('ci_high'))} | {p.get('bootstrap_reps_used')} |")
        lines.append("")
    lines += ["## Reading notes", "",
              "- Intervals resample ENGINES; repeated training seeds are a separate axis "
              "and are reported as `n_seeds` / `_std`, never folded into an interval.",
              "- `loc_ap` excludes windows where every observed channel is a target "
              "(`loc_n_all_observed_are_targets`) — average precision is undefined there — "
              "and windows with no observed target (`loc_n_no_observed_target`).",
              "- `loc_n_ambiguous` counts tied top scores; no winner is forced, so a large "
              "ambiguity count with a high `loc_unique_top1_accuracy` is a small sample, "
              "not a good method.",
              "- An infinite threshold is the honest answer to too few independent "
              "calibration objects. `trial_infinite_threshold_frac` says how often that "
              "happened; it is never clipped to a finite value.",
              "- Query seconds compare THIS implementation of each method; node-row "
              "evaluations compare the algorithms and are a work proxy, not FLOPs.",
              "- Masks whose `mechanism` is not `externally_fixed_shared` are stress "
              "tests: the conservative threshold's exchangeability argument does not "
              "extend to them.", ""]
    return "\n".join(lines)


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    columns: List[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in columns})


def generate(root: str, out_dir: Optional[str] = None, score: str = "relational_max",
             reps: int = 1000, seed: int = 0, metrics: Sequence[str] = (),
             plots: bool = True, paired: bool = True) -> Dict[str, Any]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no such log root: {root}")
    metrics = tuple(metrics) or ("auroc", "loc_ap", "end_to_end_unique_top1")
    out_dir = out_dir or os.path.join(root, "report")
    os.makedirs(out_dir, exist_ok=True)
    inventory = run_inventory(root)
    if not inventory["n_usable"]:
        raise ValueError(
            f"{root}: no run finished with a completed diagnosis stage. "
            f"{inventory['n_refused']} refused: {inventory['refused'][:5]}")
    rows = read_results(root, require_ok=True)
    tables = build_tables(rows)
    frontier = quality_cost_frontier(rows)
    # Paired comparisons cost one bootstrap per (run x mask x comparator x
    # metric), which is the bulk of this command.  A study with no comparator
    # arm -- the optimisation grid, where every run is the same method -- has
    # nothing to pair, so generating them there buys an empty table at the
    # price of the whole report.
    paired_requested = bool(paired)
    paired = (paired_comparisons(root, inventory, score, reps, seed, metrics)
              if paired_requested else [])
    histories = read_histories(root, inventory)
    written = write_plots(out_dir, histories, frontier) if plots else []

    report = {"header": HEADER, "root": root, "score": score,
              "paired_requested": paired_requested,
              "bootstrap_reps": reps, "metrics": list(metrics),
              "inventory": inventory, "tables": tables, "paired": paired,
              "quality_cost": frontier, "plots": written}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    for key, table in tables.items():
        write_csv(os.path.join(out_dir, f"{key}.csv"), table)
    write_csv(os.path.join(out_dir, "quality_cost.csv"), frontier)
    write_csv(os.path.join(out_dir, "paired.csv"), paired)
    markdown = render_markdown(os.path.basename(os.path.normpath(root)),
                               tables, inventory, paired, frontier)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(markdown + "\n")
    report["out_dir"] = out_dir
    report["markdown"] = markdown
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a log root containing completed runs")
    ap.add_argument("--out", default=None, help="output directory (default: <root>/report)")
    ap.add_argument("--score", default="relational_max",
                    help="score view used for the paired comparisons")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", dest="metrics", action="append", default=[],
                    choices=["auroc", "loc_ap", "end_to_end_unique_top1"],
                    help="repeatable; default is all three")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-paired", action="store_true",
                    help="skip the paired comparisons (a study with one method "
                         "per run has nothing to pair)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    report = generate(args.root, args.out, args.score, args.reps, args.seed,
                      args.metrics, plots=not args.no_plots,
                      paired=not args.no_paired)
    if not args.quiet:
        print(report["markdown"])
    print(f"\nwritten: {report['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
