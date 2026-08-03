"""
PoC part 4 — driver for the explanation experiments (library: `explain.py`).

Detection is a three-way tie (conv-AE ~ chain PC ~ Mahalanobis), so the
question that decides whether the circuit earns its place is not "does it score
better" but "can it say WHY, correctly and completely, when the methods tying
it cannot".  Three claims, three measurements:

  1. CORRECTNESS   channel localisation vs the generator's ground truth
  2. COMPLETENESS  do attributions sum to the score with no residual
  3. FAITHFULNESS  deletion curves (model-agnostic, so nobody wins by
                   self-consistency)

plus `--examples`, which prints and plots individual windows, because a number
that says "0.87 AUROC" does not tell you whether the explanation reads like
something an engineer would act on.

Run:
    PYTHONPATH=. python -m poc.time_series.run_explain
    PYTHONPATH=. python -m poc.time_series.run_explain --examples --plots
    PYTHONPATH=. python -m poc.time_series.run_explain --seeds 0 1 2 --shapley 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from .baselines import ChannelZScore, ConvAutoencoder
from .circuits import WindowPC
from .data import make_ad_task
from .explain import (
    GaussianConditional,
    ae_channel_error,
    completeness_error,
    deletion_curve,
    explain_window,
    format_explanation,
    localization_report,
    pc_attributions,
    plot_case_study,
    plot_deletion_curves,
    plot_localization,
    sampling_shap,
    zscore_channel,
)


def build_all(args, seed):
    task = make_ad_task(window=args.window, stride=args.stride, seed=seed,
                        n_units=args.units, n_channels=args.channels,
                        n_regimes=args.regimes, inject_rate=args.inject_rate,
                        strength=args.strength)
    pc = WindowPC(task.window, task.n_channels, vtree_method=args.vtree,
                  n_sum_components=args.K, channel_groups=task.channel_groups,
                  delta=args.delta, seed=seed)
    pc.fit(task.X_train, epochs=args.epochs, lr=args.lr)
    ae = ConvAutoencoder(task.window, task.n_channels, seed=seed).fit(task.X_train)
    gc = GaussianConditional(task.window, task.n_channels).fit(task.X_train)
    zs = ChannelZScore(task.window, task.n_channels).fit(task.X_train)
    return task, pc, ae, gc, zs


def attributions_for(args, task, pc, ae, gc, zs, seed):
    attrs, timing = {}, {}
    t0 = time.time()
    attrs.update(pc_attributions(pc, task.X_test, shapley_orders=args.shapley))
    timing["pc_exact_s"] = time.time() - t0
    attrs[gc.name] = gc.attribute(task.X_test)
    attrs["AE reconstruction (per channel)"] = ae_channel_error(ae, task.X_test)
    t0 = time.time()
    attrs[f"AE sampling-SHAP ({args.shap_samples}/ch)"] = sampling_shap(
        ae, task.X_test, task.X_train, n_samples=args.shap_samples, seed=seed)
    timing["sampling_shap_s"] = time.time() - t0
    attrs["z-score (per channel)"] = zscore_channel(zs, task.X_test)
    return attrs, timing


def run_once(args, seed: int) -> Dict:
    task, pc, ae, gc, zs = build_all(args, seed)
    print(f"  {task}")
    attrs, timing = attributions_for(args, task, pc, ae, gc, zs, seed)

    loc = {n: localization_report(a, task.affected_test, task.kind_test, args.kinds)
           for n, a in attrs.items()}
    # per-kind breakdown: the typed views are specialised, so a pooled number
    # hides exactly the effect they exist to produce
    per_kind = {n: {k: localization_report(a, task.affected_test,
                                           task.kind_test, [k])["auroc"]
                    for k in args.kinds}
                for n, a in attrs.items()}
    comp = completeness_error(pc, task.X_test[:args.n_complete])

    curves, dele = {}, {}
    if args.deletion:
        score_fn = lambda X: pc.score(X)         # one shared scorer for fairness
        sel = np.array([k in args.kinds and bool(a) for k, a
                        in zip(task.kind_test, task.affected_test)])
        Xs = task.X_test[torch.from_numpy(sel)]
        for n, a in attrs.items():
            c, auc = deletion_curve(score_fn, Xs, a[sel], task.window,
                                    task.n_channels, reference=task.X_train)
            curves[n] = c
            dele[n] = auc
    return {"loc": loc, "per_kind": per_kind, "completeness": comp, "deletion": dele,
            "curves": {k: v.tolist() for k, v in curves.items()},
            "timing": timing, "_task": task, "_pc": pc}


def show_examples(args, task, pc) -> None:
    """Print and (optionally) plot one window of each injected kind."""
    print("\n\n=== worked examples — one window per anomaly kind ===")
    print("    'out of range'        = large marginal, small structural")
    print("    'broken relationship' = small marginal, large structural\n")
    os.makedirs(args.fig_dir, exist_ok=True)
    kinds = ["normal"] + list(args.kinds)
    for kind in kinds:
        idxs = [i for i, k in enumerate(task.kind_test) if k == kind]
        if not idxs:
            continue
        # pick the median-scoring instance, not the most extreme: cherry-picking
        # the worst case would flatter every method equally and prove nothing
        s = pc.score(task.X_test[idxs]).numpy()
        i = idxs[int(np.argsort(s)[len(s) // 2])]
        x = task.X_test[i]
        exp = explain_window(pc, x, top=args.top)
        truth = task.affected_test[i] if task.affected_test[i] else None
        print(f"  [{kind}]")
        print(format_explanation(exp, truth=truth, kind=kind))
        print()
        if args.plots:
            p = os.path.join(args.fig_dir, f"case_{kind}.png")
            plot_case_study(exp, x, task.window, task.n_channels, truth, kind, p)
            print(f"    figure -> {p}\n")


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
    ap.add_argument("--vtree", default="chain")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--shapley", type=int, default=0,
                    help="random orderings for exact-conditional Shapley (0=off)")
    ap.add_argument("--shap-samples", type=int, default=32)
    ap.add_argument("--n-complete", type=int, default=64)
    ap.add_argument("--deletion", action="store_true", default=True)
    ap.add_argument("--no-deletion", dest="deletion", action="store_false")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--fig-dir", default="logs/figs")
    ap.add_argument("--kinds", nargs="+",
                    default=["spike", "offset", "drift", "decouple", "desync"])
    ap.add_argument("--out", default="logs/poc_ts_explain.json")
    args = ap.parse_args(argv)

    runs = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        runs.append(run_once(args, seed))

    names = list(runs[0]["loc"])
    print(f"\n\n=== 1. CORRECTNESS: channel localisation vs ground truth "
          f"({len(args.seeds)} seed(s)) ===")
    print(f"    scored on {args.kinds}; normal/organic excluded "
          "(no localised truth)\n")
    head = f"{'attribution method':>36} {'AUROC':>17} {'prec@k':>17}"
    print(head); print("-" * len(head))
    summary = {}
    order = sorted(names, key=lambda n: -np.mean([r["loc"][n]["auroc"] for r in runs]))
    for n in order:
        a = [r["loc"][n]["auroc"] for r in runs]
        p = [r["loc"][n]["prec_at_k"] for r in runs]
        summary[n] = {"auroc": float(np.mean(a)), "auroc_sd": float(np.std(a)),
                      "prec_at_k": float(np.mean(p)), "prec_sd": float(np.std(p))}
        print(f"{n:>36} {np.mean(a):>8.4f}±{np.std(a):<8.4f} "
              f"{np.mean(p):>8.4f}±{np.std(p):<8.4f}")

    print(f"\n=== 1b. localisation AUROC BY ANOMALY KIND ===\n")
    kh = f"{'attribution method':>36} " + " ".join(f"{k:>10}" for k in args.kinds)
    print(kh); print("-" * len(kh))
    for n in order:
        cells = " ".join(
            f"{np.nanmean([r['per_kind'][n][k] for r in runs]):>10.4f}"
            for k in args.kinds)
        print(f"{n:>36} {cells}")
    print("\n    'decouple'/'desync' leave every per-channel marginal in "
          "distribution,\n    so they are the columns where a joint/conditional "
          "view must win.\n")

    print("\n=== 2. COMPLETENESS: do attributions sum to the score? ===\n")
    mx = np.mean([r["completeness"]["max_residual_nats"] for r in runs])
    mn = np.mean([r["completeness"]["mean_residual_nats"] for r in runs])
    print(f"    PC chain-rule attribution: max residual {mx:.2e} nats, "
          f"mean {mn:.2e} nats")
    print("    (float32 round-off — completeness is a theorem here, not a target.")
    print("     SHAP satisfies it only up to estimation error, and its")
    print("     conditional variant needs conditionals the AE cannot provide.)\n")

    if runs[0]["deletion"]:
        print("=== 3. FAITHFULNESS: deletion curves (lower AUC = more faithful) ===\n")
        dh = f"{'attribution method':>36} {'deletion AUC':>14}"
        print(dh); print("-" * len(dh))
        dorder = sorted(names, key=lambda n: np.mean([r["deletion"][n] for r in runs]))
        for n in dorder:
            v = np.mean([r["deletion"][n] for r in runs])
            summary.setdefault(n, {})["deletion_auc"] = float(v)
            print(f"{n:>36} {v:>14.4f}")
        print("\n    Neutralise the most-blamed channel first and re-score; a")
        print("    faithful explanation makes the score collapse quickly.\n")

    t = runs[0]["timing"]
    print(f"  cost: PC exact attribution {t['pc_exact_s']:.1f}s for all views; "
          f"AE sampling-SHAP\n        {t['sampling_shap_s']:.1f}s for ONE "
          f"approximate view ({args.shap_samples} samples/channel).\n")

    if args.plots:
        os.makedirs(args.fig_dir, exist_ok=True)
        plot_localization(summary, os.path.join(args.fig_dir, "localization.png"))
        if runs[0]["curves"]:
            plot_deletion_curves({k: np.asarray(v) for k, v in runs[0]["curves"].items()},
                                 os.path.join(args.fig_dir, "deletion.png"))
        print(f"  figures -> {args.fig_dir}/\n")

    if args.examples:
        show_examples(args, runs[0]["_task"], runs[0]["_pc"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "completeness": {"max_residual_nats": mx, "mean_residual_nats": mn},
                   "runs": [{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in runs]}, f, indent=2, default=str)
    print(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
