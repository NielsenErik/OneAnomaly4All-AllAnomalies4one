"""
Run-level logging for the time-series experiment pipeline.

Every run — one (config variant, seed) pair — owns a directory and writes the
same seven artifacts, so a batch of two hundred runs can be aggregated,
resumed, or audited months later without remembering how it was launched:

    <run_dir>/
      config.json        fully resolved config (post grid-expansion, post --set)
      env.json           git commit + dirty flag, host, python/torch/CUDA, GPU,
                         command line, thread counts — everything needed to
                         explain a number that does not reproduce
      run.log            complete console output (stdout AND stderr, tee'd)
      history_<tag>.csv  per-epoch training curves
      metrics.json       the run's final metrics, one nested dict
      results.jsonl      one line per (stage, method) row — the aggregation input
      status.json        running | ok | failed  (+ traceback, wall time, peak RAM)
      artifacts/         scores.npz, attributions.npz, figures, ...

`status.json` is what makes a batch resumable: a run whose status is "ok" AND
whose config hash matches is skipped on the next launch.  A run that died
mid-way is "running" (stale) or "failed", and gets redone.  This is deliberate
— an interrupted overnight batch on a workstation is the normal case, not the
exception.

Design note: the previous PoC drivers printed tables to stdout and dumped one
JSON per script.  That is fine for three runs and useless for two hundred, so
this module keeps the console output (nothing is lost) while making the machine
-readable half the primary artifact.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:                                              # optional, only for RSS
    import psutil
except Exception:                                 # pragma: no cover
    psutil = None


# ═══════════════════════════════════════════════════════════════════════════
# Environment capture
# ═══════════════════════════════════════════════════════════════════════════

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return ""


def environment_report() -> Dict[str, Any]:
    """Everything that could make a number differ between two machines."""
    import torch

    gpus: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name,
                         "total_memory_gb": round(p.total_memory / 2 ** 30, 2),
                         "capability": f"{p.major}.{p.minor}"})
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpus": gpus,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": (round(psutil.virtual_memory().total / 2 ** 30, 1)
                         if psutil else None),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }


def config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of a resolved config — the resume key."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# stdout/stderr tee
# ═══════════════════════════════════════════════════════════════════════════

class _Tee(io.TextIOBase):
    """Write to the real stream and to a file at once, line-buffered."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s: str) -> int:                       # type: ignore[override]
        self.stream.write(s)
        self.stream.flush()
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()


# ═══════════════════════════════════════════════════════════════════════════
# RunLogger
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RunLogger:
    """
    One run's artifact directory.  Use as a context manager:

        with RunLogger(run_dir, config, seed) as log:
            log.info("training")
            log.history("train_nll", losses)
            log.result({"stage": "ad", "method": "PC", "auroc": 0.93})
            log.metrics({"ad": {...}})

    On exit it writes status.json with wall time, peak RSS, peak GPU memory,
    and — if the body raised — the traceback.  The exception is re-raised
    unless `swallow=True`, which is what the batch runner uses so one broken
    variant cannot take down a twelve-hour launch.
    """

    run_dir: str
    config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    echo: bool = True
    swallow: bool = False

    def __post_init__(self) -> None:
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._t0 = time.time()
        self._fh = None
        self._saved = None
        self._results: List[dict] = []
        self._metrics: Dict[str, Any] = {}
        self.failed = False

    # ── paths ────────────────────────────────────────────────────────────

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.run_dir, "artifacts")

    def path(self, *parts: str) -> str:
        return os.path.join(self.run_dir, *parts)

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> "RunLogger":
        # A run directory holds ONE run.  results.jsonl used to be opened in
        # append mode and never truncated, so every re-run (--force, a fixed
        # bug, a changed evaluator) stacked another copy of every row on top of
        # the old ones.  The aggregator then averaged them: seven attempts of a
        # three-seed config reported "21 seeds", mixing results from before and
        # after the fix being tested.  Truncating here is what makes a re-run
        # mean "replace", which is what every caller already assumes.
        open(self.path("results.jsonl"), "w").close()
        self._fh = open(self.path("run.log"), "a", buffering=1)
        self._saved = (sys.stdout, sys.stderr)
        if self.echo:
            sys.stdout = _Tee(self._saved[0], self._fh)
            sys.stderr = _Tee(self._saved[1], self._fh)
        else:                                              # file only
            sys.stdout = sys.stderr = self._fh             # type: ignore[assignment]

        self._write("config.json", {"seed": self.seed, "config_hash": self.hash,
                                    **self.config})
        self._write("env.json", environment_report())
        self._status("running")
        self.info(f"=== run start · {os.path.basename(self.run_dir)} · "
                  f"seed {self.seed} · hash {self.hash} ===")
        self._reset_peak_gpu()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall = time.time() - self._t0
        info: Dict[str, Any] = {
            "wall_s": round(wall, 2),
            "peak_rss_gb": self._peak_rss_gb(),
            "peak_gpu_gb": self._peak_gpu_gb(),
        }
        if exc is not None:
            self.failed = True
            info["error"] = f"{exc_type.__name__}: {exc}"
            info["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))
            self.info(f"!!! FAILED after {wall:.1f}s: {info['error']}")
            self._status("failed", **info)
        else:
            if self._metrics:
                self._write("metrics.json", self._metrics)
            self.info(f"=== run ok · {wall:.1f}s · "
                      f"peak RSS {info['peak_rss_gb']} GB ===")
            self._status("ok", **info)

        sys.stdout, sys.stderr = self._saved                # type: ignore[assignment]
        if self._fh:
            self._fh.close()
        return bool(exc is not None and self.swallow)

    # ── writers ──────────────────────────────────────────────────────────

    @property
    def hash(self) -> str:
        return config_hash(self.config)

    def info(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def history(self, tag: str, values: Sequence[float]) -> None:
        """Per-epoch curve → CSV (one file per tag)."""
        with open(self.path(f"history_{tag}.csv"), "w") as f:
            f.write("epoch,value\n")
            for i, v in enumerate(values):
                f.write(f"{i},{float(v)}\n")
        if len(values):
            self.info(f"  {tag}: first={float(values[0]):.4f} "
                      f"last={float(values[-1]):.4f} ({len(values)} epochs)")

    def result(self, row: Dict[str, Any]) -> None:
        """One comparable row (stage, method, metric...) → results.jsonl."""
        row = {"seed": self.seed, **row}
        self._results.append(row)
        with open(self.path("results.jsonl"), "a") as f:
            f.write(json.dumps(row, default=_jsonable) + "\n")

    def metrics(self, block: Dict[str, Any]) -> None:
        """Merge into the run's metrics.json (written at exit)."""
        self._metrics.update(block)

    def artifact_npz(self, name: str, **arrays) -> str:
        """Raw arrays (scores, attributions, curves) for later re-analysis."""
        path = os.path.join(self.artifacts_dir, f"{name}.npz")
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
        return path

    def artifact_json(self, name: str, obj: Any) -> str:
        path = os.path.join(self.artifacts_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_jsonable)
        return path

    # ── internals ────────────────────────────────────────────────────────

    def _write(self, name: str, obj: Any) -> None:
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=2, default=_jsonable)

    def _status(self, status: str, **extra) -> None:
        self._write("status.json", {"status": status, "seed": self.seed,
                                    "config_hash": self.hash,
                                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    **extra})

    @staticmethod
    def _reset_peak_gpu() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    @staticmethod
    def _peak_gpu_gb() -> Optional[float]:
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        except Exception:
            pass
        return None

    @staticmethod
    def _peak_rss_gb() -> Optional[float]:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is BYTES on macOS and KiB on Linux; both -> GiB
            scale = 2 ** 30 if sys.platform == "darwin" else 2 ** 20
            return round(peak / scale, 3)
        except Exception:
            return None


def _jsonable(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
    except Exception:
        pass
    return str(o)


# ═══════════════════════════════════════════════════════════════════════════
# Resume support
# ═══════════════════════════════════════════════════════════════════════════

def run_status(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, "status.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def is_complete(run_dir: str, config: Dict[str, Any]) -> bool:
    """True when this exact config already finished successfully here."""
    st = run_status(run_dir)
    return bool(st and st.get("status") == "ok"
                and st.get("config_hash") == config_hash(config))


def read_results(root: str) -> List[dict]:
    """Every results.jsonl row under `root`, annotated with its run directory."""
    rows: List[dict] = []
    for dirpath, _, files in os.walk(root):
        if "results.jsonl" not in files:
            continue
        rel = os.path.relpath(dirpath, root)
        cfg: Dict[str, Any] = {}
        cpath = os.path.join(dirpath, "config.json")
        if os.path.exists(cpath):
            try:
                with open(cpath) as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        with open(os.path.join(dirpath, "results.jsonl")) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("run_dir", rel)
                row.setdefault("variant", cfg.get("variant", rel.split(os.sep)[0]))
                row.setdefault("experiment", cfg.get("name"))
                rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation helpers (mean ± std over seeds)
# ═══════════════════════════════════════════════════════════════════════════

def group_stats(rows: Iterable[dict], group_keys: Sequence[str],
                value_keys: Sequence[str]) -> List[dict]:
    """Group rows and report mean/std/n for every value key present."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in group_keys), []).append(r)
    out = []
    for key, rs in groups.items():
        entry: Dict[str, Any] = dict(zip(group_keys, key))
        for vk in value_keys:
            vals = [float(r[vk]) for r in rs
                    if r.get(vk) is not None and _is_number(r[vk])]
            if not vals:
                continue
            a = np.asarray(vals, dtype=float)
            a = a[~np.isnan(a)]
            if not len(a):
                continue
            entry[f"{vk}_mean"] = float(a.mean())
            entry[f"{vk}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        entry["n_seeds"] = len({r.get("seed") for r in rs})
        entry["seeds"] = sorted({r.get("seed") for r in rs})
        out.append(entry)
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
