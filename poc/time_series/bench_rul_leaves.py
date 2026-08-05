"""
Is the leaf deficit CAPACITY, or is it INITIALISATION?

    PYTHONPATH=. python -m poc.time_series.bench_rul_leaves --device cuda
    PYTHONPATH=. python -m poc.time_series.bench_rul_leaves --components 1 3 --quick

WHAT THIS FOLLOWS UP.  `bench_rul_capacity` swept four knobs against the
ridge line on C-MAPSS FD001 and only one of them moved: mixture leaves.  Over
3 seeds, `leaf_components=3` took RMSE-last 25.45 -> 20.96 (ridge 15.96),
CRPS 12.86 -> 10.36 and TRAINING NLL 453 -> 60.  Epochs, rul_K and bins were
all flat, and rul_K bought 36 nats of density for zero RMSE.  So the binding
constraint was the single-Gaussian leaf, not the coupling capacity.

But that one row cannot distinguish three explanations, and they imply
completely different next moves:

  CAPACITY       three components genuinely fit the sensor marginals better.
                 Then RMSE keeps falling with more components, and the fix is
                 "use mixture leaves", full stop.
  INITIALISATION `GaussianLeaf.fit` seeds sigma at max(MAD, 0.01*std, 1e-3);
                 `GaussianMixtureLeaf.fit` seeds it at std/n_components, i.e.
                 3x SHARPER at n=3, with no relative floor.  A sharper start
                 alone raises train density.  Then the gain appears at 2
                 components and flattens immediately.
  COLLAPSE       the mixture leaf's only runtime guard is softplus + 1e-5.  A
                 component that walks onto a duplicated value drives sigma to
                 the floor and density to +inf.  Four silent degeneracies have
                 already produced confident wrong numbers in this project;
                 a 10x NLL drop has exactly this shape too.

The seed spread is what raised the question: RMSE-last for the SAME config was
18.40 on seed 0 and 20.96 averaged over seeds 0-2, while every non-leaf row in
that table sat at 25.4-25.6 regardless of seed.  Something about the mixture
leaf is seed-sensitive, and mean-only reporting hides it.

SO THIS BENCH REPORTS, per component count:  per-seed RMSE and the spread, not
just the mean;  a HELD-OUT joint NLL (the capacity bench's NLL column is
`history[-1]`, a training loss, and must never be read as a test likelihood);
and sigma diagnostics — the smallest fitted sigma, the 1st percentile, and how
many components sit at the floor.

READ IT AS:
  * RMSE falls monotonically with components and sigma_min stays off the floor
        -> CAPACITY.  Mixture leaves are the fix; pick the knee.
  * RMSE jumps at 2 then flattens
        -> INITIALISATION.  Port the relative floor / MAD seeding from
           GaussianLeaf.fit into GaussianMixtureLeaf.fit and re-run; the
           "capacity" win should then mostly reproduce at 1 component.
  * NLL_test improves while sigma_min collapses toward 1e-5, or the seed
        spread grows with components -> COLLAPSE.  The number is an artefact.

Rows are appended to --out as they complete, so an interrupt keeps the work
already paid for.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .bench_rul_capacity import last_idx, make_task, ridge_reference
from .circuits import DegenerateModelError, SurvivalPC
from .metrics import crps_from_pmf, rmse
from .progress import track
from src.probabilistic_circuits import GaussianLeaf, GaussianMixtureLeaf

# Held fixed at the capacity bench's baseline point, so a component count is
# never confounded with a knob that table already showed to be flat.
BASE = dict(K=12, bins=25, tau_where="deep", epochs=60)


def leaf_sigmas(pc: SurvivalPC) -> np.ndarray:
    """Every fitted sigma in the circuit, window leaves only (tau is
    Categorical).  `modules()` memoises, so the DAG is walked once."""
    out: List[np.ndarray] = []
    for m in pc.pc.modules():
        if isinstance(m, GaussianMixtureLeaf):
            out.append(m.sigmas.detach().cpu().numpy().ravel())
        elif isinstance(m, GaussianLeaf):
            out.append(np.atleast_1d(m.sigma.detach().cpu().numpy()))
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def heldout_nll(pc: SurvivalPC, X: torch.Tensor, tau: torch.Tensor,
                batch_size: int = 512) -> float:
    """Mean exact joint -log p(window, tau) on TEST data.  Same variable set
    and same bin count across rows, so it is comparable down the column —
    which `history[-1]` is not."""
    tot, n = 0.0, 0
    for s in range(0, len(X), batch_size):
        xb = X[s:s + batch_size]
        tb = tau[s:s + batch_size].to(pc.device).float()
        lp = pc.pc.log_prob(pc._augment(xb, tb))
        tot += float(lp.sum()); n += len(xb)
    return -tot / max(n, 1)


def run_one(task, seed: int, comps: int, epochs: int, K: int,
            tau_where: str, device: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    pc = SurvivalPC(task.window, task.n_channels, task.n_bins, task.cap,
                    vtree_method="chain", n_sum_components=K,
                    leaf_components=comps, tau_where=tau_where,
                    channel_groups=task.channel_groups, seed=seed, device=device)
    pc.fit(task.X_train, task.tau_train, task.delta_train, epochs=epochs,
           lr=0.05, batch_size=256, use_censored=True)
    sig = leaf_sigmas(pc)
    pred = pc.predict(task.X_test)
    i = last_idx(task)
    bw = task.cap / task.n_bins
    return {
        "rmse_last": rmse(np.asarray(pred["mean"])[i], task.rul_test[i]),
        "rmse_all": rmse(pred["mean"], task.rul_test),
        "crps": crps_from_pmf(torch.as_tensor(pred["pmf"]), task.tau_test, bw),
        "nll_train": float(pc.history[-1]),
        "nll_test": heldout_nll(pc, task.X_test, task.tau_test),
        "sigma_min": float(sig.min()) if len(sig) else float("nan"),
        "sigma_p1": float(np.percentile(sig, 1)) if len(sig) else float("nan"),
        "n_floor": int((sig < 1e-3).sum()),      # below GaussianLeaf's init floor
        "n_sigma": int(len(sig)),
        "params": pc.size()["parameters"],
        "secs": time.time() - t0,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--K", type=int, default=BASE["K"])
    ap.add_argument("--epochs", type=int, default=BASE["epochs"])
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    ap.add_argument("--quick", action="store_true",
                    help="one seed, 15 epochs — wiring check, not a result")
    ap.add_argument("--out", default="logs/rul_leaves.json",
                    help="rows appended here as they finish")
    ap.add_argument("--floor", choices=("relative", "legacy"), default="relative",
                    help="which RUNTIME floor holds during training: "
                         "'relative' = sigma >= max(0.01*std, 1e-3) per "
                         "feature; 'legacy' = the absolute 1e-5 epsilon, i.e. "
                         "the pre-change behaviour that produced the collapsed "
                         "3-component row. BOTH modes initialise from the same "
                         "relative rule (see GaussianLeaf.fit), so the "
                         "difference between the two runs is the runtime floor "
                         "alone. Run BOTH: that difference is the whole "
                         "question.")
    args = ap.parse_args(argv)
    seeds = [args.seeds[0]] if args.quick else args.seeds
    epochs = 15 if args.quick else args.epochs
    GaussianLeaf.use_relative_floor = args.floor == "relative"
    GaussianMixtureLeaf.use_relative_floor = args.floor == "relative"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []

    print(f"\nC-MAPSS {args.subset} · leaf-component sweep at the capacity "
          f"bench's baseline point\nK={args.K}, bins={BASE['bins']}, "
          f"tau_where={BASE['tau_where']}, epochs={epochs}, seeds {seeds}, "
          f"sigma floor = {args.floor}\n")

    tasks = {s: make_task(args.subset, BASE["bins"], s) for s in seeds}
    ref = [ridge_reference(tasks[s]) for s in seeds]
    r_last = float(np.mean([r["rmse_last"] for r in ref]))
    print(f"{'RIDGE REFERENCE (same split)':<24}{r_last:>10.2f}\n")

    hdr = (f"{'leaf comps':<24}{'RMSE last':>10}{'(per seed)':>22}"
           f"{'CRPS':>8}{'NLL test':>10}{'sigma min':>11}{'@floor':>8}"
           f"{'vs ridge':>10}{'fit s':>7}")
    print(hdr); print("-" * len(hdr))

    for c in args.components:
        got: List[Dict[str, Any]] = []
        for s in track(seeds, f"leaf_components={c}", total=len(seeds)):
            try:
                got.append(run_one(tasks[s], s, c, epochs, args.K,
                                   BASE["tau_where"], args.device))
            except DegenerateModelError as exc:
                print(f"{f'{c} comp.':<24}   DEGENERATE — {str(exc)[:56]}")
                got = []
                break
        if not got:
            continue
        f = lambda k: float(np.mean([g[k] for g in got]))
        per = " ".join(f"{g['rmse_last']:.1f}" for g in got)
        row = {"components": c, "seeds": seeds, "epochs": epochs, "K": args.K,
               "floor": args.floor, "ridge_rmse_last": r_last, "runs": got,
               "rmse_last_mean": f("rmse_last"),
               "rmse_last_std": float(np.std([g["rmse_last"] for g in got]))}
        written.append(row)
        out.write_text(json.dumps(written, indent=2))     # after EVERY row
        print(f"{f'{c} comp.':<24}{f('rmse_last'):>10.2f}"
              f"{('[' + per + ']'):>22}{f('crps'):>8.2f}{f('nll_test'):>10.1f}"
              f"{f('sigma_min'):>11.2e}"
              f"{f('n_floor'):>8.0f}{f('rmse_last') - r_last:>+10.2f}"
              f"{f('secs'):>7.0f}")

    print(f"\nrows -> {out}")
    print("Interpretation:")
    print("  * monotone RMSE fall + sigma_min off the floor -> CAPACITY;")
    print("  * jump at 2 then flat -> INITIALISATION (port GaussianLeaf.fit's")
    print("    relative sigma floor into GaussianMixtureLeaf.fit and re-run);")
    print("  * sigma_min -> 1e-5, @floor climbing, or seed spread growing with")
    print("    components -> COLLAPSE, and the density gain is an artefact.\n")


if __name__ == "__main__":
    main()
