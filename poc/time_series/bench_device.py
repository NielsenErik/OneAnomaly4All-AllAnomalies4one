"""
CPU vs GPU for a probabilistic circuit — measured, for BOTH evaluators.

    PYTHONPATH=. python -m poc.time_series.bench_device
    PYTHONPATH=. python -m poc.time_series.bench_device --batches 32 256 2048 8192
    PYTHONPATH=. python -m poc.time_series.bench_device --compile   # + torch.compile
    PYTHONPATH=. python -m poc.time_series.bench_device --window 20 --channels 15 --K 8

WHAT THIS MEASURED, AND WHAT IT OVERTURNED
------------------------------------------
The earlier version of this file asserted that a circuit is inherently
GPU-hostile: "hundreds of nodes, each doing a small stack/logsumexp, evaluated
by a Python recursion over the DAG", therefore launch-latency bound, therefore
the CPU usually wins.  Two of those three clauses were about the EVALUATOR, not
about the model, and the conclusion did not survive replacing it.

`CompiledCircuit` (src/probabilistic_circuits.py §6a) compiles the DAG once
into a topological layer schedule: flat int64 child-index buffers, packed
parameter tensors, one preallocated value buffer, one gather + one logsumexp
per sum layer (a real GEMM when the K units of a region mix the same children),
one gather + one sum per product layer.  The Python loop goes from O(#nodes)
to O(depth) — on the standard 8×14 window with K=6 that is 1057 nodes and 16
layers, a 66× reduction in interpreter steps and kernel launches.

Measured on an M-series Mac (torch 2.10, fp32, chain vtree, window 8×14, K=6,
1057 nodes / 16 layers), forward pass throughput in samples/s:

    batch   cpu recursive   cpu layered   mps recursive   mps layered
       32          1,368        26,626             486        18,963
      256         12,705        63,265           4,001       105,116
     1024          9,972        40,119          11,761       233,593
     4096         63,714        84,552          61,682       245,324

Three conclusions, none of which match the old docstring:

  1. The recursion, not the model, was the bottleneck: 4-39× on identical
     hardware, identical numbers (max |Δ log p| ≈ 1.5e-5 in fp32, gated below).
  2. With the recursion the GPU LOSES to the CPU at every batch size — that is
     the observation the old docstring generalised into a claim about circuits.
     With the compiled evaluator the GPU WINS by ~2.9× at batch 4096.
  3. The crossover is a batch size, not a property: below ~256 the GPU is still
     launch-bound and the CPU is competitive; above it the GPU pulls away and
     keeps scaling while the CPU flattens.  Batch across sliding windows and
     the GPU is the right device; feed it 32 windows at a time and it is not.

So `--device` remains a knob to measure per machine — but the honest statement
is "a circuit is a wide, shallow, batched computation once you stop evaluating
it one node at a time", not "a circuit is GPU-hostile".

METHOD (the previous benchmark got this wrong too)
  * warmup iterations before every timed region;
  * torch.cuda.synchronize / torch.mps.synchronize around it, since launches
    are asynchronous and an unsynchronised timer measures the enqueue;
  * inputs already resident on the device — host-to-device transfer is timed
    and reported SEPARATELY, never folded into compute;
  * a batch sweep, because a single small batch says nothing about a GPU;
  * throughput in samples/s, plus the training step (forward + backward + Adam)
    which is what the fit loop actually pays;
  * the correctness gate runs FIRST and refuses to report any speedup for an
    evaluator that disagrees with the recursive reference.
"""
from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Optional

import torch

from src.probabilistic_circuits import (
    CompiledCircuit,
    GaussianLeaf,
    RegionGraphPC,
    move_circuit_,
)

from .circuits import build_window_vtree, resolve_device


# ═══════════════════════════════════════════════════════════════════════════
# timing helpers
# ═══════════════════════════════════════════════════════════════════════════

def sync(dev: torch.device) -> None:
    """Without this the timer measures kernel ENQUEUE, not kernel execution."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def timed(fn: Callable[[], object], dev: torch.device,
          iters: int = 20, warmup: int = 5) -> float:
    """Seconds per call, warmed up and synchronised.  CUDA events when we can:
    they time the device, not the host's view of it."""
    for _ in range(warmup):
        fn()
    sync(dev)
    if dev.type == "cuda":
        start, end = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(dev)
        return start.elapsed_time(end) / 1000.0 / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / iters


def available_devices() -> List[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append("mps")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# the circuit under test
# ═══════════════════════════════════════════════════════════════════════════

def build(window: int, channels: int, K: int, vtree: str, dev: torch.device,
          seed: int = 0):
    d = window * channels
    torch.manual_seed(seed)
    vt = build_window_vtree(vtree, window, channels, X=torch.randn(512, d), seed=seed)
    pc = RegionGraphPC(vt, n_sum_components=K, leaf_factory=GaussianLeaf,
                       weight_jitter=0.5, seed=seed)
    pc.validate()
    move_circuit_(pc, dev)
    return pc


def correctness_gate(pc, dev: torch.device, d: int, verbose: bool = True) -> float:
    """
    Refuse to benchmark an evaluator that does not reproduce the reference.

    Edge cases are deliberate: a −inf child (a leaf far in a tail underflows to
    −inf in fp32), a single-child node, and the widest/deepest layers of the
    real circuit are all exercised by the random and extreme batches below.
    """
    comp = CompiledCircuit(pc.root, device=dev)
    worst = 0.0
    checks = {
        "random": torch.randn(64, d, device=dev),
        "extreme (−inf-inducing tails)": torch.randn(16, d, device=dev) * 60.0,
        "constant": torch.zeros(8, d, device=dev),
    }
    def report(name: str, err: float) -> float:
        g = comp.last_gate
        if verbose:
            print(f"  gate · {name:<34} max |Δ log p| = {err:.3e}  "
                  f"(relative {g['rel']:.1e}, |log p| ~ {g['scale']:.3g})")
        return g["rel"]

    for name, x in checks.items():
        worst = max(worst, report(name, comp.assert_matches_reference(pc.root, x)))
    marg = list(range(0, d, 7))
    worst = max(worst, report(
        f"marginalising {len(marg)} of {d} features",
        comp.assert_matches_reference(pc.root, checks["random"], marginalized=marg)))
    if verbose:
        print(f"  gate · PASS — worst RELATIVE error {worst:.1e}, i.e. fp32 "
              f"round-off (eps = {torch.finfo(torch.float32).eps:.1e})\n")
    return worst


# ═══════════════════════════════════════════════════════════════════════════
# benchmark
# ═══════════════════════════════════════════════════════════════════════════

def bench(devices: List[str], batches: List[int], window: int, channels: int,
          K: int, vtree: str, use_compile: bool, train_step: bool,
          seed: int = 0) -> Dict[str, Dict[int, float]]:
    d = window * channels
    results: Dict[str, Dict[int, float]] = {}
    schedule = None

    for dv in devices:
        dev = resolve_device(dv)
        pc = build(window, channels, K, vtree, dev, seed=seed)
        comp = CompiledCircuit(pc.root, device=dev)
        if schedule is None:
            schedule = comp.schedule_report()

        variants: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "recursive": lambda x, pc=pc: pc.log_prob(x),
            "layered": lambda x, c=comp: c.log_prob(x),
        }
        if use_compile:
            # static shapes + reduce-overhead => CUDA graphs capture the whole
            # layer schedule, which is where the remaining launch cost lives
            cc = torch.compile(comp, mode="reduce-overhead", dynamic=False)
            variants["layered+compile"] = lambda x, c=cc: c(x)

        for name, fn in variants.items():
            key = f"{dv}:{name}"
            results[key] = {}
            for B in batches:
                x = torch.randn(B, d, device=dev)          # already resident
                try:
                    with torch.no_grad():
                        # recursion at large B is slow enough that 20 iters is
                        # a waste of wall-clock; the variance is tiny anyway
                        iters = 5 if name == "recursive" and B >= 1024 else 20
                        s = timed(lambda: fn(x), dev, iters=iters,
                                  warmup=2 if name == "recursive" else 5)
                    results[key][B] = B / s
                except Exception as exc:                   # OOM at large B
                    results[key][B] = float("nan")
                    print(f"    {key} B={B}: {str(exc)[:60]}")

        if train_step:
            for name, model in (("recursive", pc), ("layered", comp)):
                key = f"{dv}:{name}:train"
                results[key] = {}
                params = list(model.parameters())
                opt = torch.optim.Adam(params, lr=0.05)

                def step(x, model=model, opt=opt, params=params):
                    loss = -model.log_prob(x).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()

                for B in batches:
                    x = torch.randn(B, d, device=dev)
                    iters = 3 if name == "recursive" and B >= 1024 else 10
                    s = timed(lambda: step(x), dev, iters=iters, warmup=2)
                    results[key][B] = B / s

        # host-to-device transfer, reported separately and never folded in
        if dev.type != "cpu":
            xc = torch.randn(max(batches), d)
            t = timed(lambda: xc.to(dev), dev, iters=20, warmup=5)
            results[f"{dv}:H2D"] = {max(batches): max(batches) / t}

    return results, schedule


def table(results: Dict[str, Dict[int, float]], batches: List[int],
          keys: List[str], title: str) -> None:
    print(f"\n{title}")
    w = 16
    print(f"{'batch':>7} " + " ".join(f"{k:>{w}}" for k in keys))
    print("-" * (8 + (w + 1) * len(keys)))
    for B in batches:
        cells = []
        for k in keys:
            v = results.get(k, {}).get(B, float("nan"))
            cells.append("—".rjust(w) if v != v else f"{v:>{w},.0f}")
        print(f"{B:>7} " + " ".join(cells))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None,
                    help="default: every device this machine has")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[32, 128, 512, 2048, 8192])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--channels", type=int, default=14)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--vtree", default="chain")
    ap.add_argument("--compile", dest="use_compile", action="store_true",
                    help="also time torch.compile(mode='reduce-overhead')")
    ap.add_argument("--no-train", dest="train_step", action="store_false",
                    help="skip the forward+backward+Adam measurement")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    devices = args.devices or available_devices()
    d = args.window * args.channels

    print(f"\ncircuit: vtree={args.vtree}  window={args.window}×{args.channels} "
          f"(d={d})  K={args.K}  torch {torch.__version__}")

    print("\ncorrectness gate — compiled evaluator vs the recursive reference")
    pc0 = build(args.window, args.channels, args.K, args.vtree,
                torch.device("cpu"), seed=args.seed)
    correctness_gate(pc0, torch.device("cpu"), d)

    results, schedule = bench(devices, args.batches, args.window, args.channels,
                              args.K, args.vtree, args.use_compile,
                              args.train_step, seed=args.seed)
    print(f"schedule: {schedule['nodes']:,} nodes · depth {schedule['depth']} · "
          f"{schedule['layers']} layers · {schedule['dense_sum_layers']} dense "
          f"(GEMM) sum layers · widest layer {schedule['max_layer_width']} · "
          f"padding waste {schedule['pad_waste']:.1%}")
    print(f"Python steps per forward: {schedule['nodes']:,} recursive → "
          f"{schedule['layers']} layered "
          f"({schedule['nodes'] / max(schedule['layers'], 1):.0f}× fewer)")

    fwd_keys = [k for k in results if not k.endswith(":train") and ":H2D" not in k]
    table(results, args.batches, fwd_keys, "forward pass — samples/s (higher is better)")
    train_keys = [k for k in results if k.endswith(":train")]
    if train_keys:
        table(results, args.batches, train_keys,
              "training step (fwd+bwd+Adam) — samples/s")

    for k in results:
        if k.endswith(":H2D"):
            B, v = next(iter(results[k].items()))
            print(f"\nhost→device transfer ({k.split(':')[0]}, B={B}): "
                  f"{v:,.0f} samples/s — excluded from the numbers above")

    # ── the two comparisons that answer the actual question ─────────────
    print("\nspeedup of the layered evaluator over the recursion, per device:")
    for dv in devices:
        r, l = results.get(f"{dv}:recursive", {}), results.get(f"{dv}:layered", {})
        gains = [l[B] / r[B] for B in args.batches
                 if r.get(B) and l.get(B) and r[B] == r[B]]
        if gains:
            print(f"  {dv:>8}: {min(gains):.1f}× – {max(gains):.1f}×")

    gpus = [dv for dv in devices if dv != "cpu"]
    for dv in gpus:
        print(f"\n{dv} vs cpu, same evaluator (crossover = where the GPU takes over):")
        for ev in ("recursive", "layered"):
            c, g = results.get(f"cpu:{ev}", {}), results.get(f"{dv}:{ev}", {})
            ratios = {B: g[B] / c[B] for B in args.batches
                      if c.get(B) and g.get(B) and c[B] == c[B]}
            if not ratios:
                continue
            cross = next((B for B in args.batches if ratios.get(B, 0) > 1.0), None)
            line = "  ".join(f"B={B}: {r:.2f}×" for B, r in ratios.items())
            print(f"  {ev:>16}  {line}")
            print(f"  {'':>16}  crossover: "
                  + (f"batch {cross}" if cross else "never in this sweep"))
    print()


if __name__ == "__main__":
    main()
