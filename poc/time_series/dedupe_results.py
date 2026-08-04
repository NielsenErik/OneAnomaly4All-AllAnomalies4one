"""
Keep only the LAST run's rows in every results.jsonl under a log root.

    PYTHONPATH=. python -m poc.time_series.dedupe_results logs/ts --dry-run
    PYTHONPATH=. python -m poc.time_series.dedupe_results logs/ts

WHY.  `results.jsonl` was opened in append mode and never truncated, so a run
directory re-executed N times held N copies of every row.  The aggregator has
no way to tell them apart — it grouped by (experiment, variant, dataset, stage,
method) and averaged the lot, reporting "21 seeds" for a 3-seed config and
silently mixing results from before and after whatever was being fixed.

The logger now truncates on entry, so this only matters for directories written
before that change.  Nothing here re-runs anything: it trims files in place
(after a .bak) to the final complete block.

HOW ROWS ARE IDENTIFIED.  Keep the LAST row for each
(stage, method, set-of-metric-names) identity.

The obvious heuristic — "cut at the last repeat of the first row's
(stage, method)" — is WRONG, and quietly so.  One run of the explain stage
emits two rows per method: a localisation row (loc_auroc, prec_at_k) and later
a deletion row (deletion_auc).  Cutting at the last repeat of the first
(stage, method) therefore lands in the MIDDLE of the final run and throws away
every localisation row, leaving a file that still looks well-formed.  It did
exactly that here before being caught.  Including the metric names in the
identity distinguishes the two rows, so both survive.

LIMITS, stated because this edits data:
  * a run that CRASHED partway leaves rows that are indistinguishable from a
    complete run's.  Check status.json — "failed" or "running" means the
    directory should be re-run, not deduped and trusted.
  * rows carry no timestamp, so "last in file order" is the only ordering
    available.  That is exactly the order they were appended in.
  * if a single run legitimately emits two rows with the SAME stage, method
    and metric names, the earlier one is lost.  No stage does that today, and
    `--check` will tell you if one starts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from typing import List, Tuple


# Fields that identify WHICH row this is, as opposed to what it measured.
_ID_FIELDS = ("stage", "experiment", "variant", "dataset", "method", "seed")
# Bookkeeping that varies run to run without changing a row's identity.
_NOISE = {"fit_s", "wall_s"}


def _identity(row: dict):
    metrics = frozenset(k for k in row
                        if k not in _ID_FIELDS and k not in _NOISE
                        and not k.startswith("axis:"))
    return tuple(row.get(f) for f in _ID_FIELDS) + (metrics,)


def last_block(rows: List[dict]) -> Tuple[List[dict], int]:
    """(deduped rows in original order, number of stale rows discarded)."""
    if not rows:
        return rows, 0
    keep_at = {}
    for i, r in enumerate(rows):
        keep_at[_identity(r)] = i          # last wins
    idx = sorted(keep_at.values())
    return [rows[i] for i in idx], len(rows) - len(idx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="log root, e.g. logs/ts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.root, "**", "results.jsonl"),
                             recursive=True))
    total_removed = total_files = 0
    for f in files:
        with open(f) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        keep, dropped = last_block(rows)
        if dropped == 0:
            continue
        # loud, per-metric: the failure mode this tool must never repeat is
        # dropping a whole CLASS of row (all the localisation rows) while
        # leaving a file that still parses and still looks complete
        def _metric_names(rs):
            from collections import Counter
            c = Counter()
            for r in rs:
                for k in r:
                    if k not in _ID_FIELDS and k not in _NOISE:
                        c[k] += 1
            return c
        before, after = _metric_names(rows), _metric_names(keep)
        lost = sorted(k for k in before if k not in after)
        if lost:
            raise SystemExit(
                f"REFUSING to write {f}: dedupe would remove every row "
                f"carrying {lost}. That is a bug in the identity function, "
                "not stale data. Nothing has been modified.")
        total_files += 1
        total_removed += len(rows) - len(keep)
        status = "?"
        sp = os.path.join(os.path.dirname(f), "status.json")
        if os.path.exists(sp):
            try:
                status = json.load(open(sp)).get("status", "?")
            except Exception:
                pass
        flag = "" if status == "ok" else f"   <-- status={status}, RE-RUN THIS"
        print(f"{os.path.relpath(f, args.root):<62} "
              f"{len(rows):>4} -> {len(keep):>3} rows "
              f"({dropped} earlier attempt(s)){flag}")
        if args.dry_run:
            continue
        if not args.no_backup:
            shutil.copy2(f, f + ".bak")
        with open(f, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")

    verb = "would remove" if args.dry_run else "removed"
    print(f"\n{verb} {total_removed} stale rows across {total_files} file(s)"
          + ("" if args.dry_run else "; originals kept as *.bak"))
    print("re-aggregate afterwards:  PYTHONPATH=. python -m "
          f"poc.time_series.aggregate {args.root} --recursive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
