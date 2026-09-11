"""
Fitted masked-diagnosis baselines — step 4 of the 10 September roadmap.

The circuit's claim is a COST curve and a missing-sensor operating regime, not
a new statistic.  A comparison against a known-law Gaussian oracle cannot test
that claim: the oracle knows the generating parameters, so it is a correctness
reference, not a competitor.  What a competitor has to be is a model of the
same data, FITTED on the same training split, answering the same masked
queries, and emitting the same score/artifact schema.  That is what lives here:

    GaussianDiagnoser         full covariance with shrinkage toward a scaled
                              identity; the regularisation strength is chosen
                              on CHECKPOINT windows, never on evaluation ones.
    LowRankGaussianDiagnoser  factor analysis (heteroscedastic diagonal plus
                              rank-r loadings), fitted by EM on the training
                              covariance, queried through Woodbury so the
                              per-mask cost model is genuinely the low-rank
                              one and not a dense solve wearing its name.
    GMMDiagnoser              full-covariance Gaussian mixture fitted by EM.

Three things are easy to get wrong here and each is a silent, plausible-looking
bug, so each is a test:

  * a mixture MARGINAL is  Σ_k π_k N(x_A; μ_kA, Σ_kA)  — the sum of component
    marginal densities under the FITTED weights.  Averaging component log
    densities is a different (and smaller) quantity, and reusing responsibilities
    computed under the FULL evidence after changing the evidence is a different
    model altogether.  `_MixtureFactorization` sums densities, and nothing here
    ever caches a posterior.
  * the conditional term is  log p(x_obs\c) − log p(x_obs) , so both terms must
    be taken INSIDE the same observation pattern.  A conditional computed
    against the full sensor set is not the quantity the masked study reports.
  * cached factorisations are keyed by the exact feature tuple and by a fit
    token, so refitting or moving device/dtype cannot serve a stale Cholesky.

Feature indexing is the flattened `t * C + c` layout the rest of the pipeline
uses, so `channel(i) = i % C`, and every method here returns exactly the keys
`WindowPC.diagnosis_map` returns (`log_px`, `marginal`, `conditional`,
`structural`, `R`, `mask`, `cost`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.probabilistic_circuits import QueryCost

Features = Tuple[int, ...]

_LOG2PI = math.log(2.0 * math.pi)


class SingularModelError(RuntimeError):
    """Raised when no admissible regularisation makes the fit factorisable.

    Deliberately loud.  A baseline that silently degrades to a diagonal model
    when the training covariance is near singular is no longer the comparator
    the report says it is.
    """


# ═══════════════════════════════════════════════════════════════════════════
# Factorisations — one cached object per (model, feature subset)
# ═══════════════════════════════════════════════════════════════════════════

def _cholesky(cov: torch.Tensor, jitter: float,
              max_tries: int = 6) -> Tuple[torch.Tensor, float, int]:
    """Cholesky with escalating jitter; reports how much was actually needed."""
    scale = float(torch.diagonal(cov).mean().clamp_min(1e-12))
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    for step in range(max_tries):
        add = 0.0 if (step == 0 and jitter == 0) else jitter * scale * (10.0 ** step)
        try:
            return torch.linalg.cholesky(cov + add * eye), add, step
        except Exception:
            continue
    raise SingularModelError(
        f"covariance of dimension {cov.shape[0]} is not positive definite after "
        f"{max_tries} jitter escalations from {jitter}")


@dataclass
class _DenseFactorization:
    """μ, chol(Σ) and log|Σ| for one feature subset of one Gaussian."""
    mean: torch.Tensor
    chol: torch.Tensor
    half_logdet: float
    jitter_used: float = 0.0

    @property
    def dim(self) -> int:
        return int(self.mean.numel())

    @property
    def n_bytes(self) -> int:
        return (self.mean.numel() + self.chol.numel()) * self.mean.element_size()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        r = (x - self.mean).unsqueeze(-1)
        z = torch.linalg.solve_triangular(self.chol, r, upper=False).squeeze(-1)
        return -0.5 * (z.pow(2).sum(-1) + self.dim * _LOG2PI) - self.half_logdet


@dataclass
class _WoodburyFactorization:
    """Ψ_A + W_A W_Aᵀ, stored as the pieces the Woodbury identity needs.

    logdet  = Σ log ψ_i + 2 Σ log diag(L),  L = chol(I_r + W_Aᵀ Ψ_A⁻¹ W_A)
    x'Σ⁻¹x  = r'Ψ⁻¹r − ‖L⁻¹ W_Aᵀ Ψ_A⁻¹ r‖²

    Never materialises the k×k covariance, which is the entire point of having
    a low-rank comparator in a cost table.
    """
    mean: torch.Tensor
    psi_inv: torch.Tensor            # (k,)
    wt_psi_inv: torch.Tensor         # (r, k) = W_Aᵀ Ψ_A⁻¹
    chol: torch.Tensor               # (r, r)
    half_logdet: float
    jitter_used: float = 0.0

    @property
    def dim(self) -> int:
        return int(self.mean.numel())

    @property
    def n_bytes(self) -> int:
        n = (self.mean.numel() + self.psi_inv.numel()
             + self.wt_psi_inv.numel() + self.chol.numel())
        return n * self.mean.element_size()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        r = x - self.mean
        quad = (r.pow(2) * self.psi_inv).sum(-1)
        b = r @ self.wt_psi_inv.transpose(0, 1)          # (n, r)
        z = torch.linalg.solve_triangular(self.chol, b.unsqueeze(-1),
                                          upper=False).squeeze(-1)
        quad = quad - z.pow(2).sum(-1)
        return -0.5 * (quad + self.dim * _LOG2PI) - self.half_logdet


@dataclass
class _MixtureFactorization:
    """Σ_k π_k N(x_A; μ_kA, Σ_kA) — component MARGINALS, fitted weights.

    The weights are the ones EM ended with.  They are not responsibilities and
    they do not change when the evidence changes; that is what makes this the
    correct marginal of the fitted mixture rather than a plausible-looking
    quantity that happens to be close to it under full observation.
    """
    components: List[_DenseFactorization]
    log_weights: torch.Tensor

    @property
    def dim(self) -> int:
        return self.components[0].dim if self.components else 0

    @property
    def n_bytes(self) -> int:
        return sum(c.n_bytes for c in self.components) + self.log_weights.numel() * 8

    @property
    def jitter_used(self) -> float:
        return max((c.jitter_used for c in self.components), default=0.0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([c.log_prob(x) for c in self.components], dim=-1)
        return torch.logsumexp(stacked + self.log_weights, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Shared contract
# ═══════════════════════════════════════════════════════════════════════════

class MaskedDiagnoser:
    """`fit(X_train)` / `diagnosis_map(X, observed)` with the circuit's schema.

    Subclasses supply `_fit_parameters` (training only), `_select` (checkpoint
    only) and `_factorize(features)`.  Everything that must be identical across
    methods — the query set, the singleton identity, the NaN convention for
    unobserved channels, mask grouping and row restoration, the cost ledger —
    lives here exactly once so the comparison cannot drift between methods.
    """

    name = "masked_diagnoser"

    def __init__(self, window: int, n_channels: int, dtype=torch.float64,
                 device: str | torch.device = "cpu"):
        if window < 1 or n_channels < 1:
            raise ValueError("positive window and channel count required")
        self.window, self.n_channels = int(window), int(n_channels)
        self.d = self.window * self.n_channels
        self.dtype, self.device = dtype, torch.device(device)
        self.fitted = False
        self.selection_: Dict[str, object] = {}
        self.stabilization_: List[Dict[str, float]] = []
        self.fit_seconds = float("nan")
        self._cache: Dict[Features, object] = {}
        self._token: Tuple = ()

    # ── indexing ─────────────────────────────────────────────────────────

    def channel_features(self, c: int) -> Features:
        return tuple(t * self.n_channels + c for t in range(self.window))

    def observed_features(self, pattern: Sequence[bool]) -> Features:
        return tuple(f for f in range(self.d) if bool(pattern[f % self.n_channels]))

    # ── fitting ──────────────────────────────────────────────────────────

    def fit(self, X_train: torch.Tensor,
            X_checkpoint: Optional[torch.Tensor] = None) -> "MaskedDiagnoser":
        """Parameters from TRAINING data; model selection on CHECKPOINT data.

        `X_checkpoint` is optional only so the class is usable in a unit test.
        When it is absent the selection split is recorded as `train`, and any
        report that prints that value is telling the reader the selection was
        in-sample.
        """
        import time
        t0 = time.perf_counter()
        X = self._prepare(X_train, "training")
        if len(X) < 2:
            raise ValueError(f"{self.name}: at least two training windows required")
        checkpoint = None if X_checkpoint is None or not len(X_checkpoint) else \
            self._prepare(X_checkpoint, "checkpoint")
        self.selection_ = {"split": "checkpoint" if checkpoint is not None else "train",
                           "n_train": int(len(X)),
                           "n_checkpoint": 0 if checkpoint is None else int(len(checkpoint))}
        self.stabilization_ = []
        self._fit_and_select(X, checkpoint if checkpoint is not None else X)
        self.fitted = True
        self._cache = {}
        self._token = self._fit_token(self._fit_counter())
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _fit_token(self, counter: int) -> Tuple:
        """What a cached factorisation is only valid for.

        Deliberately NOT a function of `stabilization_`: jitter events are
        appended while factorising, so folding them in here would invalidate
        the cache on the very call that filled it.
        """
        return (id(self), self.dtype, str(self.device), counter)

    def _fit_counter(self) -> int:
        counter = getattr(self, "_fits", 0) + 1
        self._fits = counter
        return counter

    def _prepare(self, X: torch.Tensor, what: str) -> torch.Tensor:
        X = torch.as_tensor(X)
        if X.ndim != 2 or X.shape[1] != self.d or not len(X):
            raise ValueError(f"{self.name}: {what} data must be (N, {self.d})")
        if not bool(torch.isfinite(X).all()):
            raise FloatingPointError(f"{self.name}: non-finite {what} value")
        return X.to(device=self.device, dtype=self.dtype)

    def _fit_and_select(self, X: torch.Tensor, checkpoint: torch.Tensor) -> None:
        raise NotImplementedError

    def _factorize(self, features: Features):
        raise NotImplementedError

    def size(self) -> Dict[str, int]:
        raise NotImplementedError

    # ── cached density on an arbitrary feature subset ─────────────────────

    def factorization(self, features: Features):
        if not self.fitted:
            raise RuntimeError(f"{self.name}: fit before querying")
        token = self._fit_token(getattr(self, "_fits", 0))
        if token != self._token:            # refit, or a device/dtype change
            self._cache, self._token = {}, token
        hit = self._cache.get(features)
        if hit is None:
            hit = self._factorize(features)
            self._cache[features] = hit
        return hit

    def log_density(self, x: torch.Tensor, features: Features,
                    cost: Optional[QueryCost] = None) -> torch.Tensor:
        """log p(x_features).  The empty subset has log-density exactly zero."""
        if not features:
            return torch.zeros(len(x), dtype=self.dtype, device=x.device)
        before = len(self._cache)
        fac = self.factorization(features)
        value = fac.log_prob(x[:, list(features)])
        if cost is not None:
            cost.passes += 1
            cost.node_visits += len(features)
            cost.node_evaluations += len(features) * len(x)
            if len(self._cache) > before:
                cost.boundary_visits += 1
        return value

    @property
    def cache_bytes(self) -> int:
        return int(sum(getattr(f, "n_bytes", 0) for f in self._cache.values()))

    # ── the shared masked query ──────────────────────────────────────────

    @torch.no_grad()
    def diagnosis_map(self, X: torch.Tensor, mask=None,
                      batch_size: int = 512) -> Dict[str, torch.Tensor]:
        """Exactly the keys and conventions of `WindowPC.diagnosis_map`.

        Masks are grouped so a shared pattern costs one set of factorisations,
        and row order is restored by index assignment, never by concatenation
        of the groups (which would silently permute a paired comparison).
        """
        if not self.fitted:
            raise RuntimeError(f"{self.name}: fit before querying")
        X = torch.as_tensor(X)
        if X.ndim != 2 or X.shape[1] != self.d or not len(X) or batch_size < 1:
            raise ValueError(f"expected nonempty (N, {self.d}) input")
        Xd = X.to(device=self.device, dtype=self.dtype)
        C = self.n_channels
        m = (torch.ones(len(X), C, dtype=torch.bool) if mask is None
             else torch.as_tensor(mask, dtype=torch.bool).cpu())
        if m.ndim == 1:
            m = m.unsqueeze(0).expand(len(X), -1)
        if m.shape != (len(X), C):
            raise ValueError("mask must have shape (C,) or (N, C)")
        m = m.contiguous()

        marginal = torch.full((len(X), C), float("nan"), dtype=self.dtype)
        conditional = torch.full_like(marginal, float("nan"))
        joint = torch.zeros(len(X), dtype=self.dtype)
        patterns = torch.unique(m, dim=0)
        cost = QueryCost(n_masks=len(patterns))
        for pattern in patterns:
            rows = torch.nonzero((m == pattern).all(1)).flatten()
            obs_feats = self.observed_features(pattern)
            observed = [c for c in range(C) if bool(pattern[c])]
            for start in range(0, len(rows), batch_size):
                idx = rows[start:start + batch_size]
                xb = Xd[idx.to(Xd.device)]
                lp = self.log_density(xb, obs_feats, cost).cpu()
                joint[idx] = lp
                for c in observed:
                    chan = self.channel_features(c)
                    rest = tuple(f for f in obs_feats if f % C != c)
                    marginal[idx, c] = -self.log_density(xb, chan, cost).cpu()
                    conditional[idx, c] = self.log_density(xb, rest, cost).cpu() - lp
        cost.output_elements = len(X) * C
        cost.peak_boundary_bytes = self.cache_bytes
        singleton = (m.sum(1) == 1).unsqueeze(1) & m
        conditional = torch.where(singleton, marginal, conditional)
        return dict(marginal=marginal, conditional=conditional,
                    structural=conditional - marginal, R=conditional - marginal,
                    log_px=joint, mask=m, cost=cost)

    # ── reporting ────────────────────────────────────────────────────────

    def report(self) -> Dict[str, object]:
        return {"method": self.name, "fitted": self.fitted,
                "fit_seconds": round(self.fit_seconds, 4),
                "selection": dict(self.selection_),
                "stabilization_events": list(self.stabilization_),
                **{f"size_{k}": v for k, v in self.size().items()}}

    def _note_stabilization(self, where: str, jitter: float, step: int) -> None:
        if jitter > 0:
            self.stabilization_.append({"where": where, "jitter": float(jitter),
                                        "escalations": int(step)})


# ═══════════════════════════════════════════════════════════════════════════
# Full-covariance Gaussian with shrinkage
# ═══════════════════════════════════════════════════════════════════════════

def _covariance(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = X.mean(0)
    centred = X - mean
    return mean, centred.transpose(0, 1) @ centred / max(len(X) - 1, 1)


def _shrink(cov: torch.Tensor, lam: float) -> torch.Tensor:
    """(1−λ)·S + λ·(tr S / d)·I — the standard target, so λ is comparable."""
    scale = float(torch.diagonal(cov).mean())
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return (1.0 - lam) * cov + lam * scale * eye


class GaussianDiagnoser(MaskedDiagnoser):
    """One Gaussian over the flattened window; shrinkage chosen on checkpoint."""

    name = "gaussian_fitted"

    def __init__(self, window: int, n_channels: int,
                 shrinkage_grid: Sequence[float] = (0.0, 1e-3, 1e-2, 5e-2, 0.2, 0.5),
                 jitter: float = 1e-8, **kwargs):
        super().__init__(window, n_channels, **kwargs)
        if not len(shrinkage_grid) or any(not 0 <= float(l) < 1 for l in shrinkage_grid):
            raise ValueError("shrinkage grid must be nonempty with entries in [0, 1)")
        self.shrinkage_grid = tuple(float(l) for l in shrinkage_grid)
        self.jitter = float(jitter)
        self.mean_: Optional[torch.Tensor] = None
        self.cov_: Optional[torch.Tensor] = None
        self.lam_ = float("nan")

    def _fit_and_select(self, X: torch.Tensor, checkpoint: torch.Tensor) -> None:
        mean, cov = _covariance(X)
        scores, best, best_score = {}, None, -float("inf")
        for lam in self.shrinkage_grid:
            candidate = _shrink(cov, lam)
            try:
                chol, jit, step = _cholesky(candidate, self.jitter)
            except SingularModelError:
                scores[str(lam)] = None
                continue
            fac = _DenseFactorization(mean, chol, float(torch.log(torch.diagonal(chol)).sum()), jit)
            value = float(fac.log_prob(checkpoint).mean())
            scores[str(lam)] = value
            if value > best_score:
                best_score, best = value, (lam, candidate, jit, step)
        if best is None:
            raise SingularModelError(
                f"{self.name}: no shrinkage level in {self.shrinkage_grid} gave a "
                "positive-definite covariance on this training split")
        self.lam_, self.cov_, jit, step = best
        self.mean_ = mean
        self._note_stabilization("full_covariance", jit, step)
        self.selection_.update({"criterion": "mean checkpoint log-density",
                                "grid": {"shrinkage": list(self.shrinkage_grid)},
                                "scores": scores, "chosen": {"shrinkage": self.lam_},
                                "chosen_score": best_score})

    def _factorize(self, features: Features) -> _DenseFactorization:
        idx = list(features)
        cov = self.cov_[idx][:, idx]
        chol, jit, step = _cholesky(cov, self.jitter)
        self._note_stabilization(f"subset[{len(idx)}]", jit, step)
        return _DenseFactorization(self.mean_[idx], chol,
                                   float(torch.log(torch.diagonal(chol)).sum()), jit)

    def size(self) -> Dict[str, int]:
        d = self.d
        return {"parameters": d + d * (d + 1) // 2, "components": 1, "rank": d}


# ═══════════════════════════════════════════════════════════════════════════
# Diagonal-plus-low-rank Gaussian (factor analysis)
# ═══════════════════════════════════════════════════════════════════════════

def _factor_analysis(cov: torch.Tensor, rank: int, iterations: int = 200,
                     tol: float = 1e-7, floor: float = 1e-6
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """EM for Ψ (diagonal) and W (d×r) from the sample covariance.

    The Ghahramani–Hinton updates written on S rather than on samples, which is
    the same fixed point and does not re-touch the data each sweep.  `floor`
    keeps Ψ strictly positive: a feature whose residual variance reaches zero
    makes every later Woodbury solve undefined, and that is the failure mode
    that produced a silently degenerate model earlier in this project.
    """
    d = cov.shape[0]
    rank = max(1, min(int(rank), d - 1))
    diag = torch.diagonal(cov).clamp_min(floor)
    psi = diag.clone()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top = eigenvalues[-rank:].clamp_min(floor).sqrt()
    W = eigenvectors[:, -rank:] * top
    eye = torch.eye(rank, dtype=cov.dtype, device=cov.device)
    previous = -float("inf")
    for _ in range(iterations):
        psi_inv = 1.0 / psi
        wt_psi_inv = W.transpose(0, 1) * psi_inv                     # (r, d)
        middle = eye + wt_psi_inv @ W                                # (r, r)
        beta = torch.linalg.solve(middle, wt_psi_inv)                # (r, d)
        beta_s = beta @ cov                                          # (r, d)
        expected = eye - beta @ W + beta_s @ beta.transpose(0, 1)    # (r, r)
        W = torch.linalg.solve(expected.transpose(0, 1),
                               (cov @ beta.transpose(0, 1)).transpose(0, 1)).transpose(0, 1)
        psi = (torch.diagonal(cov) - torch.diagonal(W @ beta_s)).clamp_min(floor)
        # convergence on the model's own log-likelihood of S
        sign_logdet = torch.log(psi).sum() + torch.logdet(eye + (W.transpose(0, 1)
                                                                 * (1.0 / psi)) @ W)
        ll = -0.5 * float(sign_logdet)
        if abs(ll - previous) < tol * max(1.0, abs(previous)):
            break
        previous = ll
    return psi, W


class LowRankGaussianDiagnoser(MaskedDiagnoser):
    """Ψ + W Wᵀ with rank chosen on checkpoint; every query uses Woodbury."""

    name = "lowrank_fitted"

    def __init__(self, window: int, n_channels: int,
                 rank_grid: Sequence[int] = (1, 2, 4, 8), floor: float = 1e-6,
                 iterations: int = 200, **kwargs):
        super().__init__(window, n_channels, **kwargs)
        if not len(rank_grid) or any(int(r) < 1 for r in rank_grid):
            raise ValueError("rank grid must be nonempty with positive ranks")
        self.rank_grid = tuple(sorted({max(1, min(int(r), self.d - 1)) for r in rank_grid}))
        self.floor, self.iterations = float(floor), int(iterations)
        self.psi_: Optional[torch.Tensor] = None
        self.W_: Optional[torch.Tensor] = None
        self.mean_: Optional[torch.Tensor] = None
        self.rank_ = -1

    def _fit_and_select(self, X: torch.Tensor, checkpoint: torch.Tensor) -> None:
        mean, cov = _covariance(X)
        self.mean_ = mean
        scores, best, best_score = {}, None, -float("inf")
        for rank in self.rank_grid:
            psi, W = _factor_analysis(cov, rank, self.iterations, floor=self.floor)
            fac = self._woodbury(mean, psi, W, tuple(range(self.d)))
            value = float(fac.log_prob(checkpoint).mean())
            scores[str(rank)] = value
            if value > best_score:
                best_score, best = value, (rank, psi, W)
        self.rank_, self.psi_, self.W_ = best
        self.selection_.update({"criterion": "mean checkpoint log-density",
                                "grid": {"rank": list(self.rank_grid)},
                                "scores": scores, "chosen": {"rank": self.rank_},
                                "chosen_score": best_score})

    def _woodbury(self, mean, psi, W, features: Features) -> _WoodburyFactorization:
        idx = list(features)
        psi_a = psi[idx]
        w_a = W[idx]
        psi_inv = 1.0 / psi_a
        wt_psi_inv = w_a.transpose(0, 1) * psi_inv
        middle = torch.eye(w_a.shape[1], dtype=psi.dtype, device=psi.device) + wt_psi_inv @ w_a
        chol, jit, step = _cholesky(middle, 1e-10)
        self._note_stabilization(f"woodbury[{len(idx)}]", jit, step)
        half_logdet = 0.5 * float(torch.log(psi_a).sum()) + float(
            torch.log(torch.diagonal(chol)).sum())
        return _WoodburyFactorization(mean[idx], psi_inv, wt_psi_inv, chol, half_logdet, jit)

    def _factorize(self, features: Features) -> _WoodburyFactorization:
        return self._woodbury(self.mean_, self.psi_, self.W_, features)

    def size(self) -> Dict[str, int]:
        return {"parameters": self.d * (2 + max(self.rank_, 0)), "components": 1,
                "rank": int(self.rank_)}


# ═══════════════════════════════════════════════════════════════════════════
# Full-covariance Gaussian mixture
# ═══════════════════════════════════════════════════════════════════════════

def _kmeanspp(X: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    centres = [X[torch.randint(len(X), (1,), generator=generator).item()]]
    for _ in range(1, k):
        distance = torch.stack([(X - c).pow(2).sum(1) for c in centres]).min(0).values
        total = float(distance.sum())
        if not math.isfinite(total) or total <= 0:
            centres.append(X[torch.randint(len(X), (1,), generator=generator).item()])
            continue
        probability = (distance / distance.sum()).cpu()
        centres.append(X[int(torch.multinomial(probability, 1, generator=generator))])
    return torch.stack(centres)


@dataclass
class _MixtureParameters:
    log_weights: torch.Tensor
    means: torch.Tensor              # (K, d)
    covariances: torch.Tensor        # (K, d, d)


def _fit_gmm(X: torch.Tensor, k: int, reg: float, iterations: int, seed: int,
             tol: float = 1e-6) -> Tuple[_MixtureParameters, float]:
    """Plain EM.  `reg` is added to every covariance diagonal each M-step, so a
    component that collapses onto a handful of windows stays factorisable."""
    n, d = X.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means = _kmeanspp(X.cpu(), k, generator).to(X.device, X.dtype)
    scale = float(X.var(0).mean().clamp_min(1e-6))
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    covariances = (scale * eye).unsqueeze(0).repeat(k, 1, 1)
    log_weights = torch.full((k,), -math.log(k), dtype=X.dtype, device=X.device)
    previous = -float("inf")
    ll = -float("inf")
    for _ in range(iterations):
        components = []
        for j in range(k):
            chol, _, _ = _cholesky(covariances[j], max(reg, 1e-10))
            fac = _DenseFactorization(means[j], chol,
                                      float(torch.log(torch.diagonal(chol)).sum()))
            components.append(fac.log_prob(X))
        joint = torch.stack(components, dim=1) + log_weights
        total = torch.logsumexp(joint, dim=1)
        ll = float(total.mean())
        responsibility = torch.exp(joint - total.unsqueeze(1))       # (n, k)
        counts = responsibility.sum(0).clamp_min(1e-10)
        log_weights = torch.log(counts / n)
        means = (responsibility.transpose(0, 1) @ X) / counts.unsqueeze(1)
        for j in range(k):
            centred = X - means[j]
            weighted = centred * responsibility[:, j].unsqueeze(1)
            covariances[j] = weighted.transpose(0, 1) @ centred / counts[j] + reg * eye
        if abs(ll - previous) < tol * max(1.0, abs(previous)):
            break
        previous = ll
    return _MixtureParameters(log_weights, means, covariances), ll


class GMMDiagnoser(MaskedDiagnoser):
    """K full-covariance components; K and the covariance floor from checkpoint."""

    name = "gmm_fitted"

    def __init__(self, window: int, n_channels: int,
                 component_grid: Sequence[int] = (1, 2, 4),
                 reg_grid: Sequence[float] = (1e-4, 1e-2),
                 iterations: int = 100, seed: int = 0, **kwargs):
        super().__init__(window, n_channels, **kwargs)
        if not len(component_grid) or any(int(k) < 1 for k in component_grid):
            raise ValueError("component grid must be nonempty with positive K")
        if not len(reg_grid) or any(float(r) <= 0 for r in reg_grid):
            raise ValueError("covariance regularisation must be strictly positive")
        self.component_grid = tuple(int(k) for k in component_grid)
        self.reg_grid = tuple(float(r) for r in reg_grid)
        self.iterations, self.seed = int(iterations), int(seed)
        self.parameters_: Optional[_MixtureParameters] = None
        self.k_, self.reg_ = -1, float("nan")

    def _fit_and_select(self, X: torch.Tensor, checkpoint: torch.Tensor) -> None:
        scores, best, best_score = {}, None, -float("inf")
        for k in self.component_grid:
            if k > len(X):
                continue
            for reg in self.reg_grid:
                params, _ = _fit_gmm(X, k, reg, self.iterations, self.seed)
                value = float(self._mixture(params, tuple(range(self.d))
                                            ).log_prob(checkpoint).mean())
                scores[f"K={k},reg={reg}"] = value
                if value > best_score:
                    best_score, best = value, (k, reg, params)
        if best is None:
            raise SingularModelError(f"{self.name}: no admissible mixture was fitted")
        self.k_, self.reg_, self.parameters_ = best
        self.selection_.update({"criterion": "mean checkpoint log-density",
                                "grid": {"components": list(self.component_grid),
                                         "reg": list(self.reg_grid)},
                                "scores": scores,
                                "chosen": {"components": self.k_, "reg": self.reg_},
                                "chosen_score": best_score})

    def _mixture(self, params: _MixtureParameters,
                 features: Features) -> _MixtureFactorization:
        idx = list(features)
        components = []
        for j in range(len(params.log_weights)):
            cov = params.covariances[j][idx][:, idx]
            chol, jit, step = _cholesky(cov, 1e-10)
            self._note_stabilization(f"component{j}[{len(idx)}]", jit, step)
            components.append(_DenseFactorization(
                params.means[j][idx], chol,
                float(torch.log(torch.diagonal(chol)).sum()), jit))
        return _MixtureFactorization(components, params.log_weights)

    def _factorize(self, features: Features) -> _MixtureFactorization:
        return self._mixture(self.parameters_, features)

    def size(self) -> Dict[str, int]:
        d, k = self.d, max(self.k_, 0)
        return {"parameters": k * (d + d * (d + 1) // 2) + max(k - 1, 0),
                "components": k, "rank": d}


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

BASELINES = {
    "gaussian": GaussianDiagnoser,
    "lowrank": LowRankGaussianDiagnoser,
    "gmm": GMMDiagnoser,
}


def build_baselines(names: Sequence[str], window: int, n_channels: int,
                    seed: int = 0, device="cpu",
                    settings: Optional[Dict[str, Dict]] = None
                    ) -> List[MaskedDiagnoser]:
    """Instantiate the requested comparators; an unknown name is an error."""
    settings = settings or {}
    out: List[MaskedDiagnoser] = []
    for name in names:
        if name not in BASELINES:
            raise KeyError(f"unknown diagnosis baseline {name!r}; "
                           f"have {sorted(BASELINES)}")
        kwargs = dict(settings.get(name, {}))
        if name == "gmm":
            kwargs.setdefault("seed", seed)
        out.append(BASELINES[name](window, n_channels, device=device, **kwargs))
    return out


def dense_reference_map(mean: torch.Tensor, cov: torch.Tensor, window: int,
                        n_channels: int, X: torch.Tensor, mask=None) -> Dict:
    """A deliberately naive dense Gaussian map, for tests to compare against.

    Builds every covariance submatrix from scratch with no caching, no
    Woodbury and no mask grouping, so agreement with a `MaskedDiagnoser` is
    evidence about the fast paths rather than a restatement of them.
    """
    x = X.double()
    d = window * n_channels
    mask = (torch.ones(n_channels, dtype=torch.bool) if mask is None
            else torch.as_tensor(mask, dtype=torch.bool))
    if mask.ndim != 1 or mask.shape[0] != n_channels:
        raise ValueError("reference map takes one shared (C,) mask")

    def lp(features: Sequence[int]) -> torch.Tensor:
        if not len(features):
            return torch.zeros(len(x), dtype=torch.double)
        f = list(features)
        law = torch.distributions.MultivariateNormal(
            mean.double()[f], covariance_matrix=cov.double()[f][:, f])
        return law.log_prob(x[:, f])

    observed = [i for i in range(d) if mask[i % n_channels]]
    joint = lp(observed)
    marginal = torch.full((len(x), n_channels), float("nan"), dtype=torch.double)
    conditional = marginal.clone()
    for c in range(n_channels):
        if mask[c]:
            marginal[:, c] = -lp([t * n_channels + c for t in range(window)])
            conditional[:, c] = lp([i for i in observed if i % n_channels != c]) - joint
    return dict(log_px=joint, marginal=marginal, conditional=conditional,
                R=conditional - marginal, structural=conditional - marginal,
                mask=mask.expand(len(x), -1))
