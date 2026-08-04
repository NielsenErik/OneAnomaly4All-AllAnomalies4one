"""
Baselines for the time-series PoC.

Two tiers, because the recurring finding in this project — and the headline of
TSB-AD (NeurIPS 2024) — is that exotic detectors routinely lose to trivial
ones, so a result that only beats deep baselines proves nothing:

  SIMPLE   per-channel z-score, moving-average residual, PCA reconstruction,
           1-NN distance, Mahalanobis, diagonal Gaussian.  These are the
           adversaries that decide whether the circuit's joint modelling is
           doing any work at all.
  ADVANCED windowed Isolation Forest, GMM, conv autoencoder, Deep SVDD.

For RUL, the adversary that matters is CONFORMALISED QUANTILE REGRESSION: it
is the field's current answer to calibrated prediction intervals, and if it
matches the circuit on calibration then exact inference bought nothing.  It is
implemented here honestly, with a proper held-out calibration split.

Where `src/baselines.py` already has a correct implementation it is reused
rather than duplicated.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.baselines import (
    KNNDistance,
    MahalanobisBaseline,
    PCAReconstruction,
    make_baseline,
)

from .circuits import resolve_device


def _bdev(device) -> torch.device:
    """Device for a torch BASELINE.

    `None` means CPU here, NOT "auto".  A baseline that quietly follows
    `resolve_device(None)` onto the GPU while the run is configured for CPU is
    both a crash waiting to happen (numpy consumers downstream) and a silent
    change to numbers that were recorded on CPU.  The caller has to ask.
    """
    return torch.device("cpu") if device is None else resolve_device(device)


def _np(X) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return np.asarray(X, dtype=np.float64)


# The torch baselines honour the same `--device` as the circuit, for one
# reason: a timing comparison between the circuit and a conv autoencoder is
# meaningless if one of them silently ran on the CPU.  The sklearn baselines
# accept the argument and ignore it — they have no device.


# ═══════════════════════════════════════════════════════════════════════════
# Detection — simple tier
# ═══════════════════════════════════════════════════════════════════════════

class ChannelZScore:
    """
    max_t max_c |x_tc − μ_c| / σ_c over the window.

    The honest adversary for any multivariate claim: it cannot see a single
    cross-channel relation, so whatever margin the circuit holds over it is
    exactly the value of joint modelling on this data.
    """

    name = "z-score (per channel)"

    def __init__(self, window: int, n_channels: int, **kw):
        self.w, self.C = window, n_channels

    def fit(self, X) -> "ChannelZScore":
        A = _np(X).reshape(-1, self.w, self.C)
        flat = A.reshape(-1, self.C)
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        return self

    def score(self, X) -> np.ndarray:
        A = _np(X).reshape(-1, self.w, self.C)
        return np.abs((A - self.mu) / self.sd).max(axis=(1, 2))


class MovingAverageResidual:
    """
    Residual of each timestep against the window's own moving average, scaled
    by the training residual spread.  The classic "prediction error" detector,
    with no learned model at all.
    """

    name = "moving-average residual"

    def __init__(self, window: int, n_channels: int, k: int = 3, **kw):
        self.w, self.C, self.k = window, n_channels, k

    def _resid(self, X) -> np.ndarray:
        A = _np(X).reshape(-1, self.w, self.C)
        ker = np.ones(self.k) / self.k
        sm = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, A)
        return np.abs(A - sm)

    def fit(self, X) -> "MovingAverageResidual":
        r = self._resid(X)
        self.sd = r.reshape(-1, self.C).std(0) + 1e-8
        return self

    def score(self, X) -> np.ndarray:
        return (self._resid(X) / self.sd).max(axis=(1, 2))


class DiagGaussian:
    """
    Full-window diagonal Gaussian: an exact density that assumes every feature
    independent.  Its gap to the circuit isolates the value of DEPENDENCE
    modelling while holding the "density estimation" paradigm fixed — the
    cleanest possible ablation of what the vtree buys.
    """

    name = "diagonal Gaussian"

    def __init__(self, **kw):
        pass

    def fit(self, X) -> "DiagGaussian":
        A = _np(X)
        self.mu, self.var = A.mean(0), A.var(0) + 1e-6
        return self

    def score(self, X) -> np.ndarray:
        A = _np(X)
        ll = -0.5 * (((A - self.mu) ** 2) / self.var + np.log(2 * np.pi * self.var))
        return -ll.sum(1)


# ═══════════════════════════════════════════════════════════════════════════
# Detection — advanced tier
# ═══════════════════════════════════════════════════════════════════════════

class ConvAutoencoder:
    """1-D conv autoencoder over (C, window); score = reconstruction error."""

    name = "conv autoencoder"

    def __init__(self, window: int, n_channels: int, hidden: int = 32,
                 latent: int = 16, epochs: int = 60, lr: float = 1e-3,
                 seed: int = 0, device=None, **kw):
        self.w, self.C = window, n_channels
        self.hidden, self.latent, self.epochs, self.lr, self.seed = (
            hidden, latent, epochs, lr, seed)
        self.device = _bdev(device)

    def _reshape(self, X) -> torch.Tensor:
        t = X if isinstance(X, torch.Tensor) else torch.from_numpy(_np(X)).float()
        return t.reshape(-1, self.w, self.C).permute(0, 2, 1).float().to(self.device)

    def fit(self, X) -> "ConvAutoencoder":
        torch.manual_seed(self.seed)
        A = self._reshape(X)
        self.net = nn.Sequential(
            nn.Conv1d(self.C, self.hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(self.hidden, self.latent, 3, padding=1), nn.ReLU(),
            nn.Conv1d(self.latent, self.hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(self.hidden, self.C, 3, padding=1),
        ).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            perm = torch.randperm(len(A), device=self.device)
            for s in range(0, len(A), 256):
                b = A[perm[s:s + 256]]
                loss = ((self.net(b) - b) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def score(self, X) -> np.ndarray:
        A = self._reshape(X)
        return ((self.net(A) - A) ** 2).mean(dim=(1, 2)).cpu().numpy()


class SklearnWrapper:
    """Adapter for the detectors already implemented in src/baselines.py."""

    # torch-backed keys in src/baselines.py; everything else is sklearn/numpy
    _TORCH_KEYS = ("ae", "deep_svdd")

    def __init__(self, key: str, name: str, seed: int = 0, device=None, **kw):
        if key in self._TORCH_KEYS:
            # RESOLVE here: src/baselines.py must never receive "auto".  It has
            # no business knowing what this project's device policy is, and
            # torch.device("auto") is a hard RuntimeError deep inside a fit.
            kw["device"] = _bdev(device)
        self.inner = make_baseline(key, seed=seed, **kw)
        self.name = name

    def fit(self, X) -> "SklearnWrapper":
        self.inner.fit(X if isinstance(X, torch.Tensor) else torch.from_numpy(_np(X)).float())
        return self

    def score(self, X) -> np.ndarray:
        t = X if isinstance(X, torch.Tensor) else torch.from_numpy(_np(X)).float()
        return _np(self.inner.score(t))


def detection_baselines(window: int, n_channels: int, seed: int = 0,
                        include_slow: bool = True, device=None) -> List:
    """The full detector line-up, simple tier first.

    `device` reaches the two torch detectors only; the rest are sklearn/numpy
    and run on CPU whatever is asked for.
    """
    out = [
        ChannelZScore(window, n_channels),
        MovingAverageResidual(window, n_channels),
        DiagGaussian(),
        SklearnWrapper("pca", "PCA reconstruction", seed=seed),
        SklearnWrapper("knn", "1-NN distance", seed=seed, k=1),
        SklearnWrapper("mahalanobis", "Mahalanobis", seed=seed),
    ]
    if include_slow:
        out += [
            SklearnWrapper("iforest", "Isolation Forest", seed=seed),
            SklearnWrapper("gmm", "GMM (8 comp.)", seed=seed),
            ConvAutoencoder(window, n_channels, seed=seed, device=device),
            SklearnWrapper("deep_svdd", "Deep SVDD", seed=seed, device=device),
        ]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RUL baselines
# ═══════════════════════════════════════════════════════════════════════════

class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class RidgeRUL:
    """Least-squares regression on the raw window. The trivial RUL adversary."""

    name = "ridge regression"

    def __init__(self, alpha: float = 1.0, **kw):
        self.alpha = alpha

    def fit(self, X, y, delta=None) -> "RidgeRUL":
        A, b = _np(X), _np(y)
        A = np.hstack([A, np.ones((len(A), 1))])
        G = A.T @ A + self.alpha * np.eye(A.shape[1])
        self.w = np.linalg.solve(G, A.T @ b)
        return self

    def predict(self, X) -> Dict[str, np.ndarray]:
        A = np.hstack([_np(X), np.ones((len(_np(X)), 1))])
        return {"mean": A @ self.w}


class MLPRegressorRUL:
    """Plain MSE regressor — the standard deep RUL baseline, no uncertainty."""

    name = "MLP regressor"

    def __init__(self, epochs: int = 80, lr: float = 1e-3, seed: int = 0,
                 device=None, **kw):
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.device = _bdev(device)

    def fit(self, X, y, delta=None) -> "MLPRegressorRUL":
        torch.manual_seed(self.seed)
        A = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        b = torch.as_tensor(_np(y), dtype=torch.float32).to(self.device)
        self.net = _MLP(A.shape[1], 1).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            perm = torch.randperm(len(A), device=self.device)
            for s in range(0, len(A), 256):
                i = perm[s:s + 256]
                loss = ((self.net(A[i]).squeeze(-1) - b[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X) -> Dict[str, np.ndarray]:
        A = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        return {"mean": self.net(A).squeeze(-1).cpu().numpy()}


class ConformalQuantileRUL:
    """
    Quantile-regression MLP + split conformal calibration (CQR, Romano et al.).

    THE adversary for the survival claim: the prognostics field's current
    answer to calibrated RUL intervals (arXiv:2212.14612; CQR-LSTM, Sensors
    2026).  It gives finite-sample marginal coverage without any distributional
    assumption — so if it matches the circuit's calibration, "exact" bought
    nothing on this axis and the claim has to move to queries CQR cannot
    express at all (survival under partial evidence, joint multi-horizon).

    Note what it cannot do by construction: it needs a POINT LABEL per training
    sample, so right-censored units must be dropped or mislabelled.
    """

    name = "quantile reg. + conformal (CQR)"

    def __init__(self, alpha: float = 0.10, epochs: int = 80, lr: float = 1e-3,
                 cal_frac: float = 0.3, seed: int = 0, device=None, **kw):
        self.alpha, self.epochs, self.lr = alpha, epochs, lr
        self.cal_frac, self.seed = cal_frac, seed
        self.device = _bdev(device)

    @staticmethod
    def _pinball(pred, target, q):
        e = target - pred
        return torch.maximum(q * e, (q - 1) * e).mean()

    def fit(self, X, y, delta=None) -> "ConformalQuantileRUL":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        A = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        b = torch.as_tensor(_np(y), dtype=torch.float32).to(self.device)
        idx = rng.permutation(len(A))
        n_cal = max(int(len(A) * self.cal_frac), 1)
        cal, tr = idx[:n_cal], idx[n_cal:]

        lo_q, hi_q = self.alpha / 2, 1 - self.alpha / 2
        self.net = _MLP(A.shape[1], 3).to(self.device)     # [lo, median, hi]
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        At, bt = A[tr], b[tr]
        for _ in range(self.epochs):
            perm = torch.randperm(len(At), device=self.device)
            for s in range(0, len(At), 256):
                i = perm[s:s + 256]
                out = self.net(At[i])
                loss = (self._pinball(out[:, 0], bt[i], lo_q)
                        + self._pinball(out[:, 1], bt[i], 0.5)
                        + self._pinball(out[:, 2], bt[i], hi_q))
                opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():                       # split-conformal calibration
            oc = self.net(A[cal])
            scores = torch.maximum(oc[:, 0] - b[cal], b[cal] - oc[:, 2])
        k = int(np.ceil((n_cal + 1) * (1 - self.alpha))) - 1
        k = min(max(k, 0), n_cal - 1)
        self.q_hat = float(torch.sort(scores).values[k])
        return self

    @torch.no_grad()
    def predict(self, X) -> Dict[str, np.ndarray]:
        A = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        out = self.net(A).cpu()
        return {"mean": out[:, 1].numpy(),
                "lo": (out[:, 0] - self.q_hat).numpy(),
                "hi": (out[:, 2] + self.q_hat).numpy()}


def rul_baselines(seed: int = 0, alpha: float = 0.10, device=None) -> List:
    """Ridge is numpy; the other two are torch and honour `device`."""
    return [RidgeRUL(), MLPRegressorRUL(seed=seed, device=device),
            ConformalQuantileRUL(seed=seed, alpha=alpha, device=device)]
