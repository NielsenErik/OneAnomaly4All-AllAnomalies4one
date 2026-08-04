"""
PoC part 2 — anomaly detection on windowed multivariate sensor data.

What this establishes (and what it deliberately does not):

  1. The DAG circuit trains and scores on windows that the tree layout cannot
     represent at all, and it is competitive with the standard detector suite.
  2. Two queries no baseline in the table can express, both from the SAME
     trained circuit and with no retraining:
       --missing  score with whole sensors DEAD, by exact marginalisation,
                  versus baselines that must impute first;
       --typed    exact per-channel split into marginal vs. structural
                  surprise, which says WHETHER an anomaly is univariate.
  3. The simple tier is reported first and on equal footing.  If z-score wins,
     that is the result — on data whose anomalies are univariate it SHOULD win,
     and that is exactly the critique (arXiv:2606.02670) this design confronts
     rather than hides.

Run:
    PYTHONPATH=. python -m poc.time_series.run_ad
    PYTHONPATH=. python -m poc.time_series.run_ad --missing --typed
    PYTHONPATH=. python -m poc.time_series.run_ad --vtree-ablation
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from .baselines import detection_baselines
from .circuits import WindowPC
from .data import make_ad_task
from .metrics import detection_report


def _fmt(v) -> str:
    return "  n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def run_once(args, seed: int) -> Dict[str, dict]:
    task = make_ad_task(
        window=args.window, stride=args.stride, seed=seed,
        n_units=args.units, n_channels=args.channels, n_regimes=args.regimes,
        inject_rate=args.inject_rate, strength=args.strength)
    print(f"  {task}  (window={task.window}, C={task.n_channels}, "
          f"d={task.X_train.shape[1]})")

    results: Dict[str, dict] = {}

    # ── the circuit ──────────────────────────────────────────────────────
    t0 = time.time()
    pc = WindowPC(task.window, task.n_channels, vtree_method=args.vtree,
                  n_sum_components=args.K, leaf_components=args.leaf_components,
                  channel_groups=task.channel_groups, use_sos=args.sos,
                  delta=args.delta, seed=seed, device=args.device)
    pc.fit(task.X_train, epochs=args.epochs, lr=args.lr, verbose=args.verbose)
    fit_s = time.time() - t0
    s = pc.score(task.X_test)
    kind = "SquaredPC/SOS" if args.sos else "RegionGraphPC"
    kind += " +delta" if args.delta else ""
    name = f"{kind} [{args.vtree} vtree, K={args.K}]"
    results[name] = {**detection_report(s, task.y_test, task.kind_test),
                     "fit_s": fit_s, "params": pc.size()["parameters"]}

    # ── baselines ────────────────────────────────────────────────────────
    for b in detection_baselines(task.window, task.n_channels, seed=seed,
                                 include_slow=not args.fast,
                                 device=args.device):
        t0 = time.time()
        try:
            b.fit(task.X_train)
            sb = b.score(task.X_test)
            results[b.name] = {**detection_report(sb, task.y_test, task.kind_test),
                               "fit_s": time.time() - t0}
        except Exception as exc:                       # a broken optional dep
            results[b.name] = {"auroc": float("nan"), "error": str(exc)[:80]}

    # ── query 1: dead sensors, exact marginalisation vs. imputation ──────
    if args.missing:
        dead = list(range(0, task.n_channels, max(task.n_channels // 4, 1)))[:args.n_dead]
        Xte = task.X_test.clone().reshape(-1, task.window, task.n_channels)
        Xte_imp = Xte.clone()
        Xte_imp[:, :, dead] = 0.0                      # mean-imputed (data is standardised)
        Xte_imp = Xte_imp.reshape(len(Xte), -1)

        exact = pc.score_with_missing(task.X_test, dead)
        imputed = pc.score(Xte_imp)
        results[f"  ↳ PC, {len(dead)} dead sensors (exact marginal)"] = \
            detection_report(exact, task.y_test, task.kind_test)
        results[f"  ↳ PC, {len(dead)} dead sensors (imputed)"] = \
            detection_report(imputed, task.y_test, task.kind_test)
        for b in detection_baselines(task.window, task.n_channels, seed=seed,
                                     include_slow=False,
                                     device=args.device)[:3]:
            b.fit(task.X_train)
            results[f"  ↳ {b.name}, {len(dead)} dead (imputed)"] = \
                detection_report(b.score(Xte_imp), task.y_test, task.kind_test)

    # ── query 2: exact typed decomposition ───────────────────────────────
    typed = None
    if args.typed:
        td = pc.typed_scores(task.X_test)
        y = task.y_test.numpy()
        kinds = np.asarray(task.kind_test)
        typed = {}
        for kind in ["normal"] + sorted(set(kinds[y == 1])):
            sel = kinds == kind
            if not sel.any():
                continue
            typed[kind] = {
                "marginal": float(td["marginal"][sel].max(1).values.mean()),
                "conditional": float(td["conditional"][sel].max(1).values.mean()),
                "structural": float(td["structural"][sel].max(1).values.mean()),
                "n": int(sel.sum()),
            }
        results["  ↳ PC, structural-only score"] = detection_report(
            td["structural"].max(1).values, task.y_test, task.kind_test)

    return {"scores": results, "typed": typed}


def vtree_ablation(args, seed: int) -> Dict[str, dict]:
    """
    T4 in miniature: same budget, same data, only the vtree changes.  Every
    method yields a valid vtree, so all four properties hold throughout and the
    comparison is purely about structure QUALITY.  `time` and `channel` are
    hand-built from the known layout — the honest adversaries for the learners.
    """
    task = make_ad_task(window=args.window, stride=args.stride, seed=seed,
                        n_units=args.units, n_channels=args.channels,
                        n_regimes=args.regimes, inject_rate=args.inject_rate,
                        strength=args.strength)
    out = {}
    for method in (args.ablation_methods or
                   ["random", "time", "channel", "chow_liu", "spectral",
                    "orc_rg", "forman_rg", "chain"]):
        t0 = time.time()
        pc = WindowPC(task.window, task.n_channels, vtree_method=method,
                      n_sum_components=args.K, leaf_components=args.leaf_components,
                      channel_groups=task.channel_groups, use_sos=args.sos,
                      delta=args.delta, seed=seed, device=args.device)
        pc.fit(task.X_train, epochs=args.epochs, lr=args.lr)
        with torch.no_grad():
            held_nll = float(-pc.pc.log_prob(task.X_train[:512]).mean())
        rep = detection_report(pc.score(task.X_test), task.y_test, task.kind_test)
        out[method] = {**rep, "train_nll": held_nll, "fit_s": time.time() - t0}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channels", type=int, default=14)
    ap.add_argument("--units", type=int, default=60)
    ap.add_argument("--regimes", type=int, default=3)
    ap.add_argument("--inject-rate", type=float, default=0.12)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--vtree", default="time")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--leaf-components", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-dead", type=int, default=3)
    ap.add_argument("--missing", action="store_true", help="dead-sensor query")
    ap.add_argument("--typed", action="store_true", help="typed decomposition")
    ap.add_argument("--vtree-ablation", action="store_true")
    ap.add_argument("--ablation-methods", nargs="+", default=None)
    ap.add_argument("--delta", action="store_true",
                    help="first-difference reparameterisation (unit Jacobian, "
                         "makes the window order-sensitive)")
    ap.add_argument("--sos", action="store_true",
                    help="use the squared (SOS) circuit instead of the monotone one")
    ap.add_argument("--fast", action="store_true", help="skip slow baselines")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="logs/poc_ts_ad.json")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:0 | mps; with the compiled "
                         "(layered) evaluator the GPU wins above batch ~128 — "
                         "measure with `python -m poc.time_series.bench_device`")
    args = ap.parse_args(argv)

    if args.vtree_ablation:
        print("\n=== vtree ablation (matched budget; structure quality only) ===\n")
        per_seed = [vtree_ablation(args, s) for s in args.seeds]
        methods = list(per_seed[0])
        print(f"{'vtree':>16} {'AUROC':>16} {'AP':>16} {'train NLL':>12}")
        print("-" * 64)
        for m in methods:
            a = [r[m]["auroc"] for r in per_seed]
            p = [r[m]["ap"] for r in per_seed]
            n = [r[m]["train_nll"] for r in per_seed]
            print(f"{m:>16} {np.mean(a):>8.4f}±{np.std(a):<6.4f} "
                  f"{np.mean(p):>8.4f}±{np.std(p):<6.4f} {np.mean(n):>12.2f}")
        print()
        return

    all_runs = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        all_runs.append(run_once(args, seed))

    # aggregate
    keys = list(all_runs[0]["scores"])
    metrics = sorted({m for r in all_runs for v in r["scores"].values() for m in v
                      if m.startswith("auroc") or m == "ap"})
    print(f"\n\n=== anomaly detection, mean ± sd over {len(args.seeds)} seeds ===\n")
    head = f"{'detector':>42} " + " ".join(f"{m:>15}" for m in ["auroc", "ap"])
    extra = [m for m in metrics if m.startswith("auroc[")]
    head += " " + " ".join(f"{m.replace('auroc', ''):>10}" for m in extra)
    print(head)
    print("-" * len(head))
    summary = {}
    for k in keys:
        row = {}
        for m in ["auroc", "ap"] + extra:
            vals = [r["scores"][k].get(m) for r in all_runs if k in r["scores"]]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            row[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else None
        summary[k] = row
        cells = []
        for m in ["auroc", "ap"]:
            cells.append(f"{row[m][0]:>7.4f}±{row[m][1]:<6.4f}" if row[m] else f"{'n/a':>15}")
        for m in extra:
            cells.append(f"{row[m][0]:>10.4f}" if row[m] else f"{'n/a':>10}")
        print(f"{k:>42} " + " ".join(cells))

    if all_runs[0]["typed"]:
        print("\n=== exact typed decomposition (mean worst-channel surprise) ===")
        print("structural = −log p(x_c | x_-c) + log p(x_c): dependence-breaking mass")
        print(f"\n{'window kind':>12} {'n':>6} {'marginal':>12} {'conditional':>12} "
              f"{'structural':>12}")
        print("-" * 58)
        for kind in all_runs[0]["typed"]:
            m = np.mean([r["typed"][kind]["marginal"] for r in all_runs])
            c = np.mean([r["typed"][kind]["conditional"] for r in all_runs])
            s = np.mean([r["typed"][kind]["structural"] for r in all_runs])
            n = all_runs[0]["typed"][kind]["n"]
            print(f"{kind:>12} {n:>6} {m:>12.3f} {c:>12.3f} {s:>12.3f}")
        print("\nA large structural term means the channel is only anomalous GIVEN "
              "the others —\ni.e. a genuinely multivariate anomaly.  Near-zero "
              "structural = univariate,\nwhich per-channel scoring already catches.\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "runs": [r["scores"] for r in all_runs]}, f, indent=2, default=str)
    print(f"\nwrote {args.out}\n")


if __name__ == "__main__":
    main()
