"""
PoC part 1 — the blocker and the fix, measured.

DensityPC._build recurses independently per mixture component, so a leaf
position is instantiated K^depth times and the circuit is O(d·K^depth).
RegionGraphPC builds each region ONCE and shares it across the K parents, so it
is O(d·K²).  This script measures both on the same vtrees and shows where the
tree layout stops being representable.

Run:
    PYTHONPATH=. python -m poc.time_series.bench_scaling
"""
from __future__ import annotations

import argparse
import time

import torch

from src.probabilistic_circuits import (
    DensityPC,
    GaussianLeaf,
    RegionGraphPC,
    circuit_size,
    random_balanced_vtree,
    time_channel_vtree,
)

# Above this many leaf modules we do not even attempt the tree layout: building
# it would take longer than the rest of the PoC and the point is already made.
TREE_BUDGET = 300_000


def predicted_tree_leaves(d: int, K: int) -> int:
    """d · K^ceil(log2 d) — the closed form for the balanced tree layout."""
    import math
    return d * K ** math.ceil(math.log2(max(d, 2)))


def bench(d: int, K: int, batch: int = 32, try_tree: bool = True) -> dict:
    vt = random_balanced_vtree(list(range(d)), seed=0)
    x = torch.randn(batch, d)
    row = {"d": d, "K": K}

    t0 = time.time()
    dag = RegionGraphPC(vt, n_sum_components=K, leaf_factory=GaussianLeaf)
    row["dag_build_s"] = time.time() - t0
    sz = circuit_size(dag.root)
    row["dag_leaves"] = sz["leaf"]
    row["dag_nodes"] = sz["nodes"]
    row["dag_params"] = sz["parameters"]
    t0 = time.time()
    lp = dag.log_prob(x)
    row["dag_fwd_s"] = time.time() - t0
    row["dag_logp"] = float(lp.mean().detach())

    est = predicted_tree_leaves(d, K)
    row["tree_leaves_predicted"] = est
    if try_tree and est <= TREE_BUDGET:
        t0 = time.time()
        tree = DensityPC(vt, n_sum_components=K, leaf_factory=GaussianLeaf)
        row["tree_build_s"] = time.time() - t0
        tsz = circuit_size(tree.root)
        row["tree_leaves"] = tsz["leaf"]
        row["tree_nodes"] = tsz["nodes"]
        row["tree_params"] = tsz["parameters"]
        t0 = time.time()
        tree.log_prob(x)
        row["tree_fwd_s"] = time.time() - t0
    else:
        row["tree_leaves"] = None
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=4, help="mixture components")
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[8, 16, 32, 64, 112, 256, 1024])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    print(f"\nTree (DensityPC) vs DAG (RegionGraphPC), K = {args.K}\n")
    hdr = (f"{'d':>6} {'tree leaves':>14} {'DAG leaves':>11} {'tree params':>12} "
           f"{'DAG params':>11} {'tree build':>11} {'DAG build':>10} {'DAG fwd':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in args.dims:
        r = bench(d, args.K, batch=args.batch)
        rows.append(r)
        tl = (f"{r['tree_leaves']:,}" if r["tree_leaves"] is not None
              else f"~{r['tree_leaves_predicted']:.1e} (skipped)")
        tp = f"{r['tree_params']:,}" if r.get("tree_params") else "—"
        tb = f"{r['tree_build_s']:.2f}s" if r.get("tree_build_s") else "—"
        print(f"{d:>6} {tl:>14} {r['dag_leaves']:>11,} {tp:>12} "
              f"{r['dag_params']:>11,} {tb:>11} "
              f"{r['dag_build_s']:>9.2f}s {r['dag_fwd_s']:>8.3f}s")

    print("\nSame model, two layouts: the DAG shares each region across the K "
          "parents that\nmix it, so size grows as O(d·K²) instead of O(d·K^depth). "
          "Both are exactly\nnormalised and both satisfy all four circuit "
          "properties — this is a layout\nchange, not a modelling change.\n")

    # the windowed time-series case that motivated the rebuild
    for w, C in [(8, 14), (16, 25), (32, 25)]:
        d = w * C
        vt = time_channel_vtree(w, C, mode="time")
        pc = RegionGraphPC(vt, n_sum_components=args.K, leaf_factory=GaussianLeaf)
        pc.validate()
        sz = circuit_size(pc.root)
        print(f"  window {w:>2}×{C:<3} (d={d:>4}): DAG {sz['leaf']:>7,} leaves, "
              f"{sz['parameters']:>8,} params, all 4 properties OK   |   "
              f"tree would need ~{predicted_tree_leaves(d, args.K):.1e} leaves")
    print()


if __name__ == "__main__":
    main()
