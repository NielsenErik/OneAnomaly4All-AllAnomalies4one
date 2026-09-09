"""
Tier 1.4 — the measured cost curve, in C and in the number of masks.

    PYTHONPATH=. python -m poc.time_series.bench_relational
    PYTHONPATH=. python -m poc.time_series.bench_relational --channels 4 8 16 32

The map reuses one lower-circuit evaluation and two upper traversals. Circuit
size and output grow with channels; scoring M masks for each window multiplies
upper arithmetic by M even within one vectorized traversal. Report traversals,
node-times-row evaluations (a work proxy, not FLOPs), boundary memory, output
size and wall time separately. The reference re-queries the same circuit; it
is an inference oracle, not a reproduction of another published method.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import torch

from .circuits import WindowPC
from .data import make_ad_task
from .diagnosis import _sync


def _timed(query, device, repeats=3):
    """Warm caches, synchronize accelerators, report median measured latency."""
    query()
    times = []
    result = None
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = query()
        _sync(device)
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def _fit(window: int, channels: int, K: int, epochs: int, seed: int = 0) -> WindowPC:
    task = make_ad_task(window=window, stride=3, seed=seed, n_units=12,
                        n_channels=channels, n_regimes=2)
    pc = WindowPC(window, channels, vtree_method="channel_blocked",
                  n_sum_components=K, leaf_components=1, seed=seed, device="cpu")
    pc.fit(task.X_train, epochs=epochs, lr=0.05, X_val=task.X_val)
    return pc, task


def sweep_channels(channels: List[int], window: int = 8, K: int = 4,
                   epochs: int = 3, n: int = 64) -> List[Dict]:
    """Relational map for all channels: 3·C oracle queries vs two passes."""
    rows = []
    for C in channels:
        pc, task = _fit(window, C, K, epochs)
        X = task.X_test[:n]
        _, oracle_s = _timed(lambda: pc.typed_scores(X), pc.device)
        r, two_s = _timed(lambda: pc.relational_map(X), pc.device)
        size = pc.size()
        rows.append({
            "channels": C, "window": window, "K": K,
            "nodes": size["nodes"], "params": size["parameters"],
            "oracle_queries": 3 * C, "oracle_node_visits": 3 * C * size["nodes"],
            "oracle_s": round(oracle_s, 4),
            "two_pass_passes": r["cost"].passes,
            "two_pass_node_visits": r["cost"].node_visits,
            "two_pass_node_evaluations": r["cost"].node_evaluations,
            "oracle_node_evaluations": 3 * C * size["nodes"] * len(X),
            "two_pass_s": round(two_s, 4),
            "visit_ratio": round(3 * C * size["nodes"]
                                 / max(r["cost"].node_visits, 1), 2),
            "wall_ratio": round(oracle_s / max(two_s, 1e-9), 2),
        })
        print(f"  C={C:>3}  nodes {size['nodes']:>7,}  "
              f"oracle {3*C:>3} queries / {oracle_s:6.3f}s   "
              f"two-pass {r['cost'].passes} passes / {two_s:6.3f}s   "
              f"visits ×{rows[-1]['visit_ratio']:>6.1f}  "
              f"wall ×{rows[-1]['wall_ratio']:>5.2f}")
    return rows


def sweep_masks(counts: List[int], window: int = 8, channels: int = 12,
                K: int = 4, epochs: int = 3, n: int = 64) -> List[Dict]:
    """M missing-sensor patterns: one boundary reuse vs one query per mask."""
    pc, task = _fit(window, channels, K, epochs)
    X = task.X_test[:n]
    rng = np.random.default_rng(0)
    rows = []
    for M in counts:
        pats = torch.ones(M, channels, dtype=torch.bool)
        for i in range(M):
            dead = rng.choice(channels, size=rng.integers(1, max(channels // 3, 2)),
                              replace=False)
            pats[i, dead] = False
        _, fast_s = _timed(lambda: pc.score_with_masks(X, pats), pc.device)
        fast = pc.last_query_cost
        _, ref_s = _timed(lambda: [pc.score_with_missing(
            X, [c for c in range(channels) if not pats[i, c]]) for i in range(M)], pc.device)
        rows.append({
            "masks": M, "channels": channels,
            "reuse_passes": fast.passes, "reuse_node_visits": fast.node_visits,
            "reuse_node_evaluations": fast.node_evaluations,
            "boundary_elements": fast.boundary_elements,
            "peak_boundary_bytes": fast.peak_boundary_bytes,
            "output_elements": fast.output_elements,
            "distinct_masks": len(torch.unique(pats, dim=0)),
            "reuse_s": round(fast_s, 4),
            "per_mask_queries": M, "per_mask_s": round(ref_s, 4),
            "wall_ratio": round(ref_s / max(fast_s, 1e-9), 2),
        })
        print(f"  masks={M:>3}  reuse {fast.passes:>3} passes / "
              f"{fast.node_evaluations:>9,} node-row evaluations / {fast_s:6.3f}s   "
              f"per-mask {M:>3} queries / {ref_s:6.3f}s   "
              f"wall ×{rows[-1]['wall_ratio']:>5.2f}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, nargs="+", default=[4, 8, 16, 24])
    ap.add_argument("--masks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--out", default=None, help="write the rows as JSON")
    args = ap.parse_args(argv)

    print("\nrelational map — cost in C (all channels, one query)")
    print("  the oracle spends 3·C circuit queries; the two-pass map spends "
          "two partial passes\n")
    ch = sweep_channels(args.channels, args.window, args.K, args.epochs,
                        args.windows)
    print("\nmissing sensors — cost in the number of distinct masks")
    print("  the reference re-queries the circuit per mask; the reuse path "
          "computes the boundary once\n")
    mk = sweep_masks(args.masks, args.window, 12, args.K, args.epochs,
                     args.windows)
    print("\nnote: node visits compare the ALGORITHMS; seconds compare THIS "
          "implementation\n      of them — the oracle runs on the compiled "
          "layer-parallel evaluator and the\n      two-pass map on the "
          "per-node Python one, so wall clock flatters the oracle\n      at "
          "small C and the gap closes as C grows.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"channels": ch, "masks": mk}, f, indent=2)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
