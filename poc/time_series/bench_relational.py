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
from typing import Dict, List, Sequence

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


# ═══════════════════════════════════════════════════════════════════════════
# Cross-family end-to-end benchmark (roadmap step 7)
# ═══════════════════════════════════════════════════════════════════════════
#
# The circuit's cost claim is only a claim if the comparator is a real model
# answering the same query on the same data at the same precision on the same
# device.  Everything below is measured, not derived, and each number is kept
# separate because they answer different questions and get conflated otherwise:
#
#   fit_s              building the model at all
#   cold_query_s       the FIRST query, with every cache empty — what a fresh
#                      process or a new mask pattern actually pays
#   warm_query_s       the median of synchronised repeats afterwards
#   cache_build_s      cold minus warm: the per-mask construction the fitted
#                      Gaussian family pays and the circuit's boundary reuse
#                      does not
#   cache_bytes        what that construction costs in memory
#   single_window_s    latency for ONE window, which is not throughput/N
#   throughput_win_s   windows per second at the benchmark batch, which is not
#                      latency
#   end_to_end_s       from raw numpy in, including tensor construction and
#                      device transfer, because that is what a caller waits for
#
# No row here is allowed to stand on the favourable workload alone: the sweeps
# vary channels, window, batch size and the number of distinct masks, and the
# writer of any speed claim has to report the sweep, not a point.

def _peak_rss_gb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 ** 3) if peak > 2 ** 32 else peak / (1024 ** 2)
    except Exception:
        return float("nan")


def _peak_gpu_gb(device) -> float:
    if getattr(device, "type", None) != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def _clear_cache(method) -> None:
    """Put a method back in its cold state without refitting it.

    Only the fitted diagnosers hold a persistent query cache. The circuit's
    reuse happens INSIDE one call — the boundary values are computed once per
    query and thrown away — so there is nothing to clear, and a cold circuit
    query is a warm one. That asymmetry is a result, not an omission: it is why
    `cache_bytes` below is zero for the circuit and megabytes for the Gaussian
    family, and why the two families move in opposite directions as the number
    of distinct masks grows.
    """
    model = getattr(method, "model", None)
    if model is not None and hasattr(model, "_cache"):
        model._cache = {}


def _cache_bytes(method) -> int:
    """Bytes of PERSISTENT cross-call cache; transient per-query working memory
    is reported separately as `peak_boundary_bytes` from the cost ledger."""
    model = getattr(method, "model", None)
    if model is not None and hasattr(model, "cache_bytes"):
        return int(model.cache_bytes)
    return 0


def bench_methods(channels: int = 8, window: int = 8, K: int = 4, epochs: int = 5,
                  n: int = 256, masks: int = 4, baselines=("gaussian", "lowrank", "gmm"),
                  repeats: int = 3, seed: int = 0) -> List[Dict]:
    """One workload, every method, identical data / precision / device."""
    from .diagnosis import CircuitMethod, BaselineMethod
    from .diagnosis_baselines import build_baselines

    pc, task = _fit(window, channels, K, epochs, seed)
    X = task.X_test[:n]
    raw = X.cpu().numpy()
    rng = np.random.default_rng(seed)
    patterns = torch.ones(max(masks, 1), channels, dtype=torch.bool)
    for i in range(1, len(patterns)):
        patterns[i, rng.choice(channels, size=max(1, channels // 4), replace=False)] = False
    # One shared mask per row, cycled, so every method sees the same workload.
    per_row = patterns[torch.arange(len(X)) % len(patterns)]

    methods = [CircuitMethod(pc, "fast" if pc.is_channel_blocked else "oracle")]
    for model in build_baselines(list(baselines), window, channels, seed=seed):
        model.fit(task.X_train, task.X_val)
        methods.append(BaselineMethod(model))

    rows: List[Dict] = []
    for method in methods:
        device = method.device
        _clear_cache(method)
        _sync(device)
        t0 = time.perf_counter()
        result = method.diagnosis_map(X, per_row)
        _sync(device)
        cold_s = time.perf_counter() - t0
        _, warm_s = _timed(lambda: method.diagnosis_map(X, per_row), device, repeats)
        _, single_s = _timed(lambda: method.diagnosis_map(X[:1], per_row[:1]), device, repeats)
        _sync(device)
        t0 = time.perf_counter()
        method.diagnosis_map(torch.from_numpy(raw), per_row)
        _sync(device)
        end_to_end_s = time.perf_counter() - t0
        cost = result["cost"].as_dict()
        size = method.size()
        rows.append({
            "method": method.name, "channels": channels, "window": window,
            "batch": len(X), "distinct_masks": int(len(torch.unique(per_row, dim=0))),
            "parameters": size.get("parameters"), "nodes": size.get("nodes"),
            "fit_s": round(float(getattr(getattr(method, "model", None), "fit_seconds",
                                         getattr(pc, "fit_seconds", float("nan")))), 4),
            "cold_query_s": round(cold_s, 5), "warm_query_s": round(warm_s, 5),
            "cache_build_s": round(max(cold_s - warm_s, 0.0), 5),
            "cache_bytes": _cache_bytes(method),
            "single_window_s": round(single_s, 6),
            "throughput_win_s": round(len(X) / max(warm_s, 1e-9), 1),
            "end_to_end_s": round(end_to_end_s, 5),
            "peak_rss_gb": round(_peak_rss_gb(), 3),
            "peak_gpu_gb": round(_peak_gpu_gb(device), 3),
            "device": str(device), "dtype": str(result["log_px"].dtype),
            "passes": cost["passes"], "node_evaluations": cost["node_evaluations"],
            "output_elements": cost["output_elements"],
            "peak_boundary_bytes": cost["peak_boundary_bytes"],
        })
        print(f"  {method.name:<18} cold {cold_s:7.3f}s  warm {warm_s:7.3f}s  "
              f"cache {rows[-1]['cache_build_s']:6.3f}s / {rows[-1]['cache_bytes']:>9,}B  "
              f"1-window {single_s * 1e3:7.2f}ms  {rows[-1]['throughput_win_s']:>9,.0f} win/s")
    return rows


def sweep_methods(channel_grid, window_grid, batch_grid, mask_grid, K: int = 4,
                  epochs: int = 5, baselines=("gaussian", "lowrank", "gmm"),
                  repeats: int = 3, seed: int = 0) -> List[Dict]:
    """Vary one axis at a time around a fixed centre; never one point."""
    rows: List[Dict] = []
    centre = dict(channels=channel_grid[0], window=window_grid[0],
                  n=batch_grid[0], masks=mask_grid[0])
    for axis, grid in (("channels", channel_grid), ("window", window_grid),
                       ("n", batch_grid), ("masks", mask_grid)):
        for value in grid:
            spec = {**centre, axis: value}
            print(f"\n  workload: {spec}")
            for row in bench_methods(K=K, epochs=epochs, baselines=baselines,
                                     repeats=repeats, seed=seed, **spec):
                rows.append({"axis": axis, **row})
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
    ap.add_argument("--methods", action="store_true",
                    help="also run the cross-family benchmark against the fitted "
                         "Gaussian / low-rank / GMM diagnosers")
    ap.add_argument("--method-channels", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--method-windows", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--method-batches", type=int, nargs="+", default=[64, 256, 512])
    ap.add_argument("--method-masks", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--baselines", nargs="+", default=["gaussian", "lowrank", "gmm"])
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
    methods = []
    if args.methods:
        print("\ncross-family end-to-end benchmark — circuit vs fitted comparators")
        print("  same data, same precision, same device; cold, warm, cache and "
              "end-to-end costs kept apart\n")
        methods = sweep_methods(args.method_channels, args.method_windows,
                                args.method_batches, args.method_masks,
                                K=args.K, epochs=args.epochs,
                                baselines=tuple(args.baselines))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"channels": ch, "masks": mk, "methods": methods}, f, indent=2)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
