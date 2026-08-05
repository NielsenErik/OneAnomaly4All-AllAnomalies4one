"""
Cross-run aggregation: many run directories in, one report out.

    PYTHONPATH=. python -m poc.time_series.aggregate logs/ts/cmapss_ad
    PYTHONPATH=. python -m poc.time_series.aggregate logs/ts --recursive
    PYTHONPATH=. python -m poc.time_series.aggregate logs/ts/cmapss_ad --stage rul

Reads every `results.jsonl` under the root, groups by
(experiment, variant, dataset, stage, method) and reports mean ± std across
seeds, then writes three files next to the runs:

    summary.json   the grouped records
    summary.csv    long format, one row per group — the thing to load in pandas
    summary.md     the tables as printed, for pasting into a write-up

Two conventions that keep the tables honest:
  * the seed count is always shown, because a 0.9 ± 0.0 over one seed and over
    five seeds are different claims;
  * nothing is sorted by a metric the config was tuned on — sorting is by the
    stage's primary metric only, with the method name as tiebreak.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .ts_logging import group_stats, read_results

# Metric shown first per stage, and the direction that counts as "better".
PRIMARY = {
    "ad": ("auroc", "desc"),
    "explain": ("loc_auroc", "desc"),
    "rul": ("crps", "asc"),
    "rul_partial": ("crps_exact_marginal", "asc"),
    "calibration": ("picp", "desc"),
    "scaling": ("d", "asc"),
}

PREFERRED_COLUMNS = {
    "ad": ["auroc", "ap", "train_nll", "fit_s", "params"],
    "explain": ["loc_auroc", "prec_at_k", "deletion_auc",
                "max_residual_nats", "mean_residual_nats"],
    # picp/picp_edge and pit_var sit next to each other on purpose: the three
    # together say whether an under-coverage number is the density or the
    # endpoint convention, and no one of them says it alone (hand-off §B.2).
    "rul": ["rmse", "mae", "nasa", "crps", "interval_score", "picp", "mpiw",
            "picp_edge", "mpiw_edge", "pit_mean", "pit_var",
            "calib_err", "pred_sd", "fit_s"],
    "rul_partial": ["crps_full", "crps_exact_marginal", "crps_imputed",
                    "rmse_full", "rmse_exact_marginal", "rmse_imputed",
                    "picp_exact_marginal", "picp_exact_marginal_edge",
                    "picp_imputed", "picp_imputed_edge", "n_dead"],
    "calibration": ["picp", "mpiw", "picp_edge", "mpiw_edge", "pit_mean",
                    "pit_var", "interval_score", "crps", "rmse", "mae"],
    "scaling": ["d", "dag_leaves", "dag_params", "dag_build_s", "dag_fwd_s",
                "tree_leaves", "tree_params", "tree_build_s"],
}

GROUP_KEYS = ("experiment", "variant", "dataset", "stage", "method")


def _numeric_keys(rows: Iterable[dict]) -> List[str]:
    keys: List[str] = []
    for r in rows:
        for k, v in r.items():
            if k in GROUP_KEYS or k in ("seed", "run_dir") or k.startswith("axis:"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if k not in keys:
                keys.append(k)
    return keys


def _order_columns(stage: str, keys: Sequence[str]) -> List[str]:
    pref = [k for k in PREFERRED_COLUMNS.get(stage, []) if k in keys]
    per_kind = sorted(k for k in keys if k.startswith(("auroc[", "loc_auroc[")))
    rest = [k for k in keys if k not in pref and k not in per_kind]
    return pref + per_kind + sorted(rest)


def aggregate_rows(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    stages = sorted({r.get("stage") for r in rows if r.get("stage")})
    for stage in stages:
        sub = [r for r in rows if r.get("stage") == stage]
        stats = group_stats(sub, GROUP_KEYS, _numeric_keys(sub))
        # carry the variant axes through so the CSV can be pivoted on them
        axes_by_group = {}
        for r in sub:
            key = tuple(r.get(k) for k in GROUP_KEYS)
            axes_by_group.setdefault(key, {k: v for k, v in r.items()
                                           if k.startswith("axis:")})
        for s in stats:
            s.update(axes_by_group.get(tuple(s.get(k) for k in GROUP_KEYS), {}))
        out.extend(stats)
    return out


def format_table(stage: str, stats: List[dict], max_cols: int = 8) -> str:
    metric, direction = PRIMARY.get(stage, ("", "desc"))
    key = f"{metric}_mean"
    rows = [s for s in stats if s.get("stage") == stage]
    if not rows:
        return ""
    rows.sort(key=lambda s: (-(s.get(key) if s.get(key) is not None else -1e18)
                             if direction == "desc"
                             else (s.get(key) if s.get(key) is not None else 1e18),
                             str(s.get("method"))))
    cols = _order_columns(stage, [k[:-5] for k in
                                  {c for s in rows for c in s if c.endswith("_mean")}])
    cols = cols[:max_cols]

    datasets = sorted({str(s.get("dataset")) for s in rows})
    variants = sorted({str(s.get("variant")) for s in rows})
    w_method = max(28, min(46, max(len(str(s.get("method"))) for s in rows) + 1))
    lines: List[str] = []
    head = f"{'method':<{w_method}}"
    if len(datasets) > 1:
        head += f" {'dataset':<16}"
    if len(variants) > 1:
        head += f" {'variant':<24}"
    head += "".join(f" {c:>18}" for c in cols) + f" {'seeds':>6}"
    lines.append(head)
    lines.append("-" * len(head))
    for s in rows:
        line = f"{str(s.get('method')):<{w_method}}"
        if len(datasets) > 1:
            line += f" {str(s.get('dataset')):<16}"
        if len(variants) > 1:
            line += f" {str(s.get('variant')):<24}"
        for c in cols:
            m, sd = s.get(f"{c}_mean"), s.get(f"{c}_std")
            if m is None:
                line += f" {'—':>18}"
            elif abs(m) >= 1e5 or (m != 0 and abs(m) < 1e-3):
                line += f" {m:>11.3e}±{sd or 0:<6.0e}"
            else:
                line += f" {m:>11.4f}±{sd or 0:<6.4f}"
        line += f" {s.get('n_seeds', 0):>6}"
        lines.append(line)
    return "\n".join(lines)


def aggregate_root(root: str, print_tables: bool = True,
                   stage_filter: Optional[str] = None) -> List[dict]:
    rows = read_results(root)
    if stage_filter:
        rows = [r for r in rows if r.get("stage") == stage_filter]
    if not rows:
        print(f"no results.jsonl found under {root}")
        return []
    stats = aggregate_rows(rows)

    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump({"root": root, "n_rows": len(rows), "results": stats}, f, indent=2)

    cols: List[str] = []
    for s in stats:
        for k in s:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in stats:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in s.items()})

    blocks: List[str] = []
    for stage in sorted({s.get("stage") for s in stats}):
        t = format_table(stage, stats)
        if not t:
            continue
        blocks.append(f"\n### stage: {stage}\n\n```\n{t}\n```\n")
        if print_tables:
            print(f"\n=== {stage} (mean ± sd over seeds) ===\n")
            print(t)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write(f"# Summary — {root}\n\n{len(rows)} result rows.\n")
        f.write("".join(blocks))

    if print_tables:
        print(f"\nwrote {os.path.join(root, 'summary.csv')}, summary.json, summary.md\n")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="logs/ts")
    ap.add_argument("--stage", default=None, help="aggregate one stage only")
    ap.add_argument("--recursive", action="store_true",
                    help="aggregate each immediate sub-directory separately too")
    args = ap.parse_args(argv)

    aggregate_root(args.root, print_tables=True, stage_filter=args.stage)
    if args.recursive:
        for name in sorted(os.listdir(args.root)):
            sub = os.path.join(args.root, name)
            if os.path.isdir(sub):
                print(f"\n{'#' * 70}\n# {sub}\n{'#' * 70}")
                aggregate_root(sub, print_tables=True, stage_filter=args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
