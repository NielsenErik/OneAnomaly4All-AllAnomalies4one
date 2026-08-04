"""
Experiment runner — the single entry point for the whole time-series pipeline.

    PYTHONPATH=. python -m poc.time_series.runner config/ts/cmapss_ad.yaml
    PYTHONPATH=. python -m poc.time_series.runner config/ts/smoke.yaml --device cpu
    PYTHONPATH=. python -m poc.time_series.runner config/ts/cmapss_rul.yaml \
        --seeds 0 1 2 3 4 --set model.K=10 --only vtree-chain

One config expands into `variants × seeds` runs; each run gets its own
directory, its own log, and its own status file.  Three properties matter for
an unattended overnight batch on a workstation and are all deliberate:

  RESUMABLE   a finished run is skipped when its config hash still matches, so
              re-launching after a crash, a reboot or a Ctrl-C costs nothing.
              `--force` overrides.
  ISOLATED    one variant blowing up (missing optional dependency, degenerate
              model, OOM) is recorded as a failed run and the batch continues.
              A twelve-hour batch must not die in hour two.
  SELF-DESCRIBING  every run stores its resolved config, the git commit, the
              GPU it ran on and the full console log next to its numbers.

Exit status is 0 when every run finished, 1 when any failed — so a shell
launcher can react without parsing output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .config import (
    apply_overrides,
    expand_variants,
    load_config,
    resolved_for_hash,
    run_dir_for,
)
from .datasets import dataset_available, dataset_id
from .pipeline import run_stages
from .ts_logging import RunLogger, config_hash, is_complete, run_status


def _print_plan(variants: List[Dict[str, Any]], seeds: List[int]) -> None:
    print(f"\n{len(variants)} variant(s) × {len(seeds)} seed(s) = "
          f"{len(variants) * len(seeds)} run(s)\n")
    for v in variants:
        print(f"  {v['variant']:<38} dataset={dataset_id(v['dataset']):<16} "
              f"stages={','.join(v['stages'])}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a YAML config (see config/ts/)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override the config's seed list")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:0 | mps")
    ap.add_argument("--evaluator", default=None, choices=["layered", "recursive"],
                    help="circuit evaluator; 'layered' (default) compiles the DAG "
                         "into a topological layer schedule, 'recursive' is the "
                         "per-node reference (same numbers, much slower)")
    # `extend`, not the default `store`: with plain nargs="+" a second --set on
    # the same command line silently REPLACES the first, which is a quiet way to
    # run the wrong experiment.
    ap.add_argument("--set", dest="overrides", nargs="+", default=[],
                    action="extend", metavar="KEY=VALUE",
                    help="dotted config overrides, e.g. model.K=8 eval.plots=false "
                         "(repeatable)")
    ap.add_argument("--only", default=None,
                    help="run only variants whose name contains this substring")
    ap.add_argument("--log-root", default=None, help="override the output root")
    ap.add_argument("--tag", default=None,
                    help="append a suffix to the log root (e.g. a date stamp)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even when a matching completed run exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--list", action="store_true", help="alias for --dry-run")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort the batch on the first failure (default: continue)")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="skip the summary pass at the end")
    ap.add_argument("--quiet", action="store_true",
                    help="run logs go to file only, not to the console")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    if args.seeds:
        cfg["seeds"] = list(args.seeds)
    if args.device:
        cfg["device"] = args.device
    if args.evaluator:
        cfg["evaluator"] = args.evaluator
    if args.log_root:
        cfg["log_root"] = args.log_root
    if args.tag:
        cfg["log_root"] = f"{cfg['log_root']}_{args.tag}"

    seeds = list(cfg["seeds"])
    variants = expand_variants(cfg)
    if args.only:
        variants = [v for v in variants if args.only in v["variant"]]
        if not variants:
            print(f"no variant matches --only {args.only!r}")
            return 1

    print("=" * 78)
    print(f" experiment: {cfg['name']}   ({cfg.get('config_path')})")
    if cfg.get("description"):
        print(f" {cfg['description'].strip().splitlines()[0]}")
    print(f" output:     {cfg['log_root']}")
    print("=" * 78)
    _print_plan(variants, seeds)

    if args.dry_run or args.list:
        return 0

    # data availability is checked ONCE, up front: a real-data config on a
    # machine without the files should say so immediately, not twelve variants
    # into a batch.
    missing = []
    for v in variants:
        ok, why = dataset_available(v["dataset"])
        if not ok:
            missing.append((v["variant"], why))
    if missing:
        for name, why in missing:
            print(f"  [data missing] {name}: {why}")
        if cfg.get("skip_if_missing_data", True):
            print("\nskipping this config (skip_if_missing_data: true).  Run\n"
                  "  python -m poc.time_series.check_data\n"
                  "for download instructions.\n")
            return 0
        return 1

    os.makedirs(cfg["log_root"], exist_ok=True)
    index_path = os.path.join(cfg["log_root"], "index.jsonl")
    n_ok = n_skip = n_fail = 0
    t_batch = time.time()

    total_runs = len(variants) * len(seeds)
    done_runs = 0
    run_times: List[float] = []

    for v in variants:
        for seed in seeds:
            rdir = run_dir_for(v, seed)
            resolved = resolved_for_hash(v)
            done_runs += 1
            if not args.force and is_complete(rdir, resolved):
                st = run_status(rdir) or {}
                print(f"  [skip] {v['variant']} seed {seed} "
                      f"(done in {st.get('wall_s', '?')}s)")
                n_skip += 1
                continue

            os.makedirs(rdir, exist_ok=True)
            # Batch-level ETA from the runs already finished IN THIS BATCH.
            # Crude (variants differ in cost) but it answers the only question
            # you have at 3 a.m.: is this thing going to be done by morning?
            eta = ""
            if run_times:
                mean = sum(run_times) / len(run_times)
                left = mean * (total_runs - done_runs + 1)
                eta = (f" · mean {mean / 60:.1f} min/run · "
                       f"ETA {left / 60:.0f} min for the remaining "
                       f"{total_runs - done_runs + 1}")
            print(f"\n>>> [{done_runs}/{total_runs}] {v['variant']} · seed {seed} "
                  f"· {rdir}{eta}")
            t0 = time.time()
            run_cfg = dict(v)
            run_cfg["seed"] = seed
            with RunLogger(rdir, config=resolved, seed=seed,
                           echo=not args.quiet,
                           swallow=not args.stop_on_error) as log:
                # config.json holds the hashed subset; keep the readable one too
                log.artifact_json("config_full", v)
                run_stages(v, seed, log)
            ok = not log.failed
            n_ok += int(ok)
            n_fail += int(not ok)
            run_times.append(time.time() - t0)
            with open(index_path, "a") as f:
                f.write(json.dumps({
                    "experiment": cfg["name"], "variant": v["variant"],
                    "seed": seed, "run_dir": os.path.relpath(rdir, cfg["log_root"]),
                    "status": "ok" if ok else "failed",
                    "wall_s": round(time.time() - t0, 1),
                    "config_hash": config_hash(resolved),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }) + "\n")
            if not ok:
                print(f"  [FAILED] see {os.path.join(rdir, 'run.log')}")
                if args.stop_on_error:
                    return 1

    dt = time.time() - t_batch
    print(f"\n{'=' * 78}\n batch finished in {dt / 60:.1f} min — "
          f"{n_ok} ok, {n_skip} skipped, {n_fail} failed\n{'=' * 78}")

    if not args.no_aggregate:
        from .aggregate import aggregate_root
        try:
            aggregate_root(cfg["log_root"], print_tables=True)
        except Exception as exc:                      # never fail a batch on a table
            print(f"(aggregation failed: {exc})")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
