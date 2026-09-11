"""
Trivial detection comparators under the SAME missing-sensor workload.

The masked-diagnosis comparators in `diagnosis_baselines.py` answer the
circuit's own question — conditionals and complements under a mask.  These
answer the blunter one, and it is the question that decides whether any of this
is worth doing:

    a practitioner who loses three sensors does not marginalise them out
    exactly.  They drop the columns and refit, or they impute and carry on.
    Is exact marginalisation better than that?

So each detector here is fitted on the training windows RESTRICTED TO THE
OBSERVED CHANNELS of the mask in question and scores the evaluation windows
restricted the same way.  That is a real method, not a straw one: it sees the
same data, pays a refit per sensor configuration (which is recorded), and on
the full mask it reduces to the ordinary detector the AD study already
reports.  `impute` is the cheaper alternative — fit once on every channel and
mean-fill the missing ones at query time — and is offered beside it because it
is what most deployments actually do.

Two things this deliberately does not do.  It does not give the detectors a
localisation score: they produce one number per window, so a per-channel
attribution would have to be invented, and an invented attribution compared
against an exact one is not a comparison.  And it does not reuse a fit across
masks: a detector fitted on fifteen channels and asked about eight has not
been fitted on the question, which is exactly the shortcut being tested.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .baselines import detection_baselines

# The line-up, by the key a config asks for.  Restricted to the detectors that
# fit in seconds: this runs once per mask per run, and a detector that costs
# more than the circuit's whole query budget cannot be swept over a mask
# workload.  `include_slow` in `baselines.detection_baselines` also brings the
# autoencoder and Deep SVDD, which are minutes each and are left out here.
_WANTED = {
    "zscore": "z-score (per channel)",
    "moving_average": "moving-average residual",
    "diag_gaussian": "diagonal Gaussian",
    "pca": "PCA reconstruction",
    "knn": "1-NN distance",
    "mahalanobis": "Mahalanobis",
    "iforest": "Isolation Forest",
    "gmm": "GMM (8 comp.)",
}

_BY_NAME = {
    "z-score (per channel)": "zscore",
    "moving-average residual": "moving_average",
    "diagonal Gaussian": "diag_gaussian",
    "PCA reconstruction": "pca",
    "1-NN distance": "knn",
    "Mahalanobis": "mahalanobis",
    "Isolation Forest": "iforest",
    "GMM (8 comp.)": "gmm",
}


def available() -> List[str]:
    return sorted(_WANTED)


def _columns(mask: torch.Tensor, window: int, channels: int) -> np.ndarray:
    """Flattened `t * C + c` columns of the observed channels."""
    m = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if m.shape != (channels,):
        raise ValueError("detector comparators take one shared channel pattern")
    keep = [int(c) for c in range(channels) if bool(m[c])]
    if not keep:
        raise ValueError("a detector needs at least one observed channel")
    return np.asarray([t * channels + c for t in range(window) for c in keep], dtype=int)


class MaskedDetector:
    """One `fit`/`score` detector, refitted per observation pattern.

    `policy`:
      `refit`   fit on the observed columns only.  The honest strong version:
                the detector is given the problem it is actually asked about.
      `impute`  fit once on all columns, then mean-fill the missing ones from
                the TRAINING mean at query time.  Cheaper, and what a
                deployment that cannot refit per sensor outage would do.
    """

    def __init__(self, key: str, window: int, channels: int, seed: int = 0,
                 policy: str = "refit"):
        if key not in _WANTED:
            raise KeyError(f"unknown detector {key!r}; have {available()}")
        if policy not in ("refit", "impute"):
            raise ValueError("policy must be refit or impute")
        self.key, self.window, self.channels = key, int(window), int(channels)
        self.seed, self.policy = int(seed), policy
        self.name = f"{key}_{policy}"
        self.display = _WANTED[key]
        self._cache: Dict[Tuple[int, ...], object] = {}
        self._train: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self.fit_seconds: float = 0.0
        self.n_fits: int = 0

    # ── fitting ──────────────────────────────────────────────────────────
    def _make(self):
        for det in detection_baselines(self.window, self.channels, seed=self.seed,
                                       include_slow=True, device="cpu"):
            if _BY_NAME.get(getattr(det, "name", None)) == self.key:
                return det
        raise KeyError(f"{self.key} is not in the baseline line-up")

    def _make_for(self, n_kept_channels: int):
        """A detector sized for a restricted channel count.

        The two window-shaped detectors (`zscore`, `moving_average`) need the
        geometry of the data they are given, so they are constructed for the
        RESTRICTED shape rather than the original one.  Handing them the full
        channel count and a narrower matrix silently reshapes the window.
        """
        for det in detection_baselines(self.window, n_kept_channels, seed=self.seed,
                                       include_slow=True, device="cpu"):
            if _BY_NAME.get(getattr(det, "name", None)) == self.key:
                return det
        raise KeyError(f"{self.key} is not in the baseline line-up")

    def fit(self, X_train: torch.Tensor) -> "MaskedDetector":
        self._train = np.asarray(X_train.detach().cpu().numpy(), dtype=np.float64)
        if self._train.ndim != 2 or self._train.shape[1] != self.window * self.channels:
            raise ValueError("training windows must be (N, window * channels)")
        self._mean = self._train.mean(axis=0)
        self._cache.clear()
        if self.policy == "impute":         # one fit, reused under every mask
            self._fitted_full = self._fit_on(np.arange(self._train.shape[1]),
                                             self.channels)
        return self

    def _fit_on(self, cols: np.ndarray, n_kept: int):
        t0 = time.perf_counter()
        det = self._make_for(n_kept)
        det.fit(torch.from_numpy(self._train[:, cols]).float())
        self.fit_seconds += time.perf_counter() - t0
        self.n_fits += 1
        return det

    # ── scoring ──────────────────────────────────────────────────────────
    def score(self, X: torch.Tensor, mask) -> np.ndarray:
        if self._train is None:
            raise RuntimeError(f"{self.name}: fit before scoring")
        cols = _columns(mask, self.window, self.channels)
        n_kept = len(cols) // self.window
        x = np.asarray(X.detach().cpu().numpy(), dtype=np.float64)
        if self.policy == "impute":
            filled = x.copy()
            missing = np.setdiff1d(np.arange(x.shape[1]), cols)
            filled[:, missing] = self._mean[missing]
            return np.asarray(self._fitted_full.score(
                torch.from_numpy(filled).float()), dtype=float)
        key = tuple(cols.tolist())
        det = self._cache.get(key)
        if det is None:
            det = self._cache[key] = self._fit_on(cols, n_kept)
        return np.asarray(det.score(torch.from_numpy(x[:, cols]).float()), dtype=float)

    def report(self) -> Dict[str, object]:
        return {"detector": self.key, "policy": self.policy,
                "display": self.display, "n_fits": self.n_fits,
                "fit_seconds": self.fit_seconds,
                "interpretation": (
                    "refit = the detector is fitted on the observed channels of this "
                    "mask; impute = one full-channel fit with the training mean filled "
                    "in for the missing ones. Neither produces a per-channel "
                    "attribution, so neither appears in a localisation table.")}


def build_detectors(names: Sequence[str], window: int, n_channels: int,
                    seed: int = 0, policies: Sequence[str] = ("refit",)
                    ) -> List[MaskedDetector]:
    out: List[MaskedDetector] = []
    for name in names:
        for policy in policies:
            out.append(MaskedDetector(name, window, n_channels, seed=seed,
                                      policy=policy))
    return out
