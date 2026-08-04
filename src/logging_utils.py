"""
Seed-aware experiment logging and reproducibility utilities.

Every run writes its artifacts under logs/<seed>/ :

    logs/
      0/
        run.log              # full text log (appended across runs)
        config.json          # configuration of the latest run
        results.jsonl        # one JSON line per (dataset, adaptation) result
        history_<tag>.csv    # per-epoch training curves
      1/
        ...
      summary.json           # cross-seed aggregate (written by the CLI)

set_global_seed pins torch / numpy / random so a run is reproducible
end-to-end (data splits, featurizer projections, leaf jitter, batch
shuffling, weight init).  aggregate_results turns per-seed result lists into
mean ± std tables for robustness reporting.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)


def set_global_seed(seed: int) -> None:
    """Pin every RNG the pipeline touches (python, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentLogger:
    """
    Logger bound to one seed: writes to logs/<seed>/ and mirrors to stdout.

        elog = ExperimentLogger(seed=0)
        elog.log_config({"latent_dim": 64})
        elog.info("training started")
        elog.log_history("train_nll", history)
        elog.log_result({"dataset": "adbench:thyroid", "auroc": 0.86, ...})
    """

    def __init__(self, seed: int, root: Optional[str] = None, echo: bool = True):
        self.seed = seed
        self.dir = os.path.join(root or DEFAULT_LOG_ROOT, str(seed))
        os.makedirs(self.dir, exist_ok=True)

        self.logger = logging.getLogger(f"oneanomaly.seed{seed}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s [seed %(seed)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(self.dir, "run.log"))
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        if echo:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)
        self._extra = {"seed": seed}

    # ── plain text ───────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self.logger.info(msg, extra=self._extra)

    # ── structured artifacts ─────────────────────────────────────────────

    def log_config(self, config: dict) -> None:
        config = {"seed": self.seed, "timestamp": _now(), **config}
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        self.info(f"config: {json.dumps(config, default=str)}")

    def log_history(self, tag: str, history: Sequence[float]) -> None:
        path = os.path.join(self.dir, f"history_{tag}.csv")
        with open(path, "w") as f:
            f.write("epoch,value\n")
            for i, v in enumerate(history):
                f.write(f"{i},{v}\n")
        if history:
            self.info(f"{tag}: first={history[0]:.4f} last={history[-1]:.4f} "
                      f"({len(history)} epochs)")

    def log_result(self, result: dict) -> None:
        result = {"seed": self.seed, "timestamp": _now(), **result}
        with open(os.path.join(self.dir, "results.jsonl"), "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
        self.info(f"result: {json.dumps(result, default=str)}")

    def read_results(self) -> List[dict]:
        path = os.path.join(self.dir, "results.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def close(self) -> None:
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ─── Cross-seed aggregation ─────────────────────────────────────────────────

def aggregate_results(
    results: Iterable[dict],
    group_keys: Tuple[str, ...] = ("dataset", "adaptation", "role", "vtree"),
    value_key: str = "auroc",
    extra_keys: Tuple[str, ...] = ("nll",),
) -> List[dict]:
    """
    Group per-seed results and report mean ± std (the robustness table).
    "vtree" is part of the default grouping so vtree-ablation runs aggregate
    per structure-learning method; rows without the tag (e.g. baselines)
    group exactly as before.  extra_keys (held-out NLL) are aggregated for
    any group where every row carries them.

    Returns one dict per group:
        {group keys..., "<value>_mean", "<value>_std", "n_seeds", "seeds"}
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in results:
        groups[tuple(r.get(k) for k in group_keys)].append(r)

    out = []
    for key, rs in groups.items():
        entry = dict(zip(group_keys, key))
        for vk in (value_key,) + tuple(extra_keys):
            if any(vk not in r for r in rs):
                continue
            vals = np.array([float(r[vk]) for r in rs])
            entry[f"{vk}_mean"] = float(vals.mean())
            entry[f"{vk}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        entry["n_seeds"] = len(rs)
        entry["seeds"] = sorted({r.get("seed") for r in rs})
        out.append(entry)
    return out


def save_summary(
    aggregated: List[dict], root: Optional[str] = None, filename: str = "summary.json"
) -> str:
    """Write the cross-seed aggregate next to the per-seed folders."""
    root = root or DEFAULT_LOG_ROOT
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    with open(path, "w") as f:
        json.dump({"timestamp": _now(), "results": aggregated}, f, indent=2)
    return path


def format_summary_table(aggregated: List[dict], value_key: str = "auroc") -> str:
    """mean ± std table for the console.  The vtree and held-out-NLL columns
    appear automatically when any row carries them (vtree ablations)."""
    show_vtree = any(r.get("vtree") for r in aggregated)
    show_nll = any("nll_mean" in r for r in aggregated)
    head = f"{'dataset':<24} {'role':<9} {'adaptation':<10} "
    if show_vtree:
        head += f"{'vtree':<9} "
    head += f"{value_key + ' (mean±std)':>20}"
    if show_nll:
        head += f" {'nll (mean±std)':>18}"
    head += f" {'seeds':>6}"
    lines = [head]
    for r in sorted(aggregated, key=lambda r: (str(r.get('role')),
                                               str(r.get('dataset')),
                                               str(r.get('adaptation')),
                                               str(r.get('vtree')))):
        row = (f"{str(r.get('dataset')):<24} {str(r.get('role', '-')):<9} "
               f"{str(r.get('adaptation')):<10} ")
        if show_vtree:
            row += f"{str(r.get('vtree') or '-'):<9} "
        row += f"{r[f'{value_key}_mean']:>13.3f} ±{r[f'{value_key}_std']:.3f}"
        if show_nll:
            row += (f" {r['nll_mean']:>11.3f} ±{r['nll_std']:.3f}"
                    if "nll_mean" in r else f" {'-':>18}")
        row += f" {r['n_seeds']:>6}"
        lines.append(row)
    return "\n".join(lines)
