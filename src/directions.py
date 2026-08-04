"""
The two research directions, built on src/probabilistic_circuits.py, behind a
shared anomaly-detection interface so they can be compared directly.

Direction 1 — LatentPCDetector (encoder + latent PC, lower risk):
    Per-modality encoders map raw inputs into a shared latent space; a single
    exact PC (DensityPC or SquaredPC) does density estimation over the latent
    variables.  Anomaly score = −log p(latent).  The encoder is the only
    source of approximation; the PC part stays exact.

Direction 2 — RoutedRawPC (encoder-free PC over raw input, the research bet):
    One exact PC per modality directly over the raw features, routed by
    modality.  When modalities share dimensionality, all sub-circuits can be
    built over a single shared vtree (structure-level transfer), including a
    consensus vtree distilled from per-modality single-task structures
    (build_consensus_routed_pc).  With an unknown modality, scoring falls
    back to an exact mixture over the sub-circuits of matching dimension.

Shared interface: AnomalyDetector — fit / log_prob / score (higher = more
anomalous) — plus rank-based AUROC and evaluate_detector for side-by-side
benchmarking.

Both directions support two training regimes (arXiv:2605.05953):
  - fit / fit_unsupervised: exact-NLL maximum likelihood on normal data only
    (α = 1; no labels — the most efficient regime and the primary target);
  - fit_contrastive: the supervised composite objective (Eq. 4) — generative
    NLL on positives plus a margin penalty displacing labeled anomalies into
    low-density regions.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .probabilistic_circuits import (
    DensityPC,
    LeafFactory,
    SquaredPC,
    VtreeNode,
    cardinality_log_posterior,
    cardinality_moments,
    learned_vtree,
    consensus_vtree,
    random_balanced_vtree,
    selection_marginals,
    single_task_vtree,
)

TensorDict = Dict[str, torch.Tensor]


# ─── Shared interface ───────────────────────────────────────────────────────

class AnomalyDetector(nn.Module):
    """
    Common interface for both directions.  Scores are negative log-densities:
    higher = more anomalous.
    """

    def fit(self, data, **kwargs) -> "AnomalyDetector":
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        raise NotImplementedError

    def score(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        return -self.log_prob(x, modality=modality)

    def validate(self) -> None:
        raise NotImplementedError


# ─── Training loops ─────────────────────────────────────────────────────────
#
# Two regimes, per arXiv:2605.05953 ("Hallucination as an Anomaly", Eq. 4):
#
#   L(θ, φ) = α · E_{h⁺}[ −log p(z⁺) ]                            (generative NLL)
#           + (1−α) · E_{h⁺,h⁻}[ max(0, γ + log p(z⁻) − log p(z⁺)) ]  (contrastive margin)
#
# 1. UNSUPERVISED (α = 1): pure exact-NLL maximum likelihood on normal data
#    only.  No labels, no anomaly examples, no pairing — the most efficient
#    regime and the project's primary target (fit_pc_unsupervised / .fit).
# 2. SUPERVISED CONTRASTIVE (α < 1): when labeled anomalies (h⁻) exist, the
#    margin term actively displaces them into low-density regions while the
#    NLL term builds the high-density manifold of normal states
#    (fit_pc_contrastive / .fit_contrastive).
#
# Both optimize the exact log-density of the circuit, so tractability and
# the circuit properties are untouched by the choice of objective.


def nll_contrastive_loss(
    log_p_pos: torch.Tensor,
    log_p_neg: torch.Tensor,
    alpha: float = 0.5,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Composite objective of arXiv:2605.05953 Eq. 4 on a batch of exact
    log-densities.  The expectation over (h⁺, h⁻) pairs is estimated with all
    B⁺ × B⁻ pairs of the batch.
    """
    gen = -log_p_pos.mean()
    contrast = torch.relu(
        margin + log_p_neg.unsqueeze(0) - log_p_pos.unsqueeze(1)
    ).mean()
    return alpha * gen + (1.0 - alpha) * contrast


def fit_pc_nll(
    model: nn.Module,
    X: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
    batch_size: Optional[int] = None,
    parameters: Optional[Iterable[nn.Parameter]] = None,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = False,
) -> list:
    """
    Unsupervised maximum-likelihood training: minimise mean exact NLL with
    Adam (the α = 1 case of Eq. 4).  `model` must expose log_prob(X) -> (B,).
    Returns the per-epoch loss history.
    """
    params = list(parameters) if parameters is not None else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    n = X.shape[0]
    bs = batch_size or n
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, bs):
            batch = X[perm[start:start + bs]]
            loss = -model.log_prob(batch).mean()
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            epoch_loss += float(loss.detach()) * batch.shape[0]
        history.append(epoch_loss / n)
        if verbose and (epoch % max(epochs // 10, 1) == 0):
            print(f"epoch {epoch:4d}  nll {history[-1]:.4f}")
    return history


# The unsupervised loop IS fit_pc_nll; the alias makes the regime explicit.
fit_pc_unsupervised = fit_pc_nll


def fit_pc_contrastive(
    model: nn.Module,
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
    alpha: float = 0.5,
    margin: float = 1.0,
    batch_size: Optional[int] = None,
    parameters: Optional[Iterable[nn.Parameter]] = None,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = False,
    encode_pos=None,
    encode_neg=None,
) -> list:
    """
    Supervised contrastive training (Eq. 4 of arXiv:2605.05953): generative
    NLL on positives (normal / factual states) plus a margin penalty pushing
    negatives (anomalous / hallucinated states) at least `margin` nats below
    matched positives in log-density.  Gradients are clipped at ‖∇‖₂ ≤ 1 as
    in the paper.

    encode_pos / encode_neg: optional callables applied to each raw batch
    inside the loop (used by LatentPCDetector to train θ and fϕ end-to-end).
    Returns the per-epoch loss history.
    """
    params = list(parameters) if parameters is not None else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    n_pos, n_neg = X_pos.shape[0], X_neg.shape[0]
    bs = batch_size or n_pos
    for epoch in range(epochs):
        perm_pos = torch.randperm(n_pos)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n_pos, bs):
            pos = X_pos[perm_pos[start:start + bs]]
            neg = X_neg[torch.randint(0, n_neg, (min(bs, n_neg),))]
            if encode_pos is not None:
                pos = encode_pos(pos)
            if encode_neg is not None:
                neg = encode_neg(neg)
            loss = nll_contrastive_loss(
                model.log_prob(pos), model.log_prob(neg), alpha=alpha, margin=margin
            )
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        history.append(epoch_loss / n_batches)
        if verbose and (epoch % max(epochs // 10, 1) == 0):
            print(f"epoch {epoch:4d}  loss {history[-1]:.4f}")
    return history


# ─── Evaluation ─────────────────────────────────────────────────────────────

def auroc(scores_normal: torch.Tensor, scores_anomalous: torch.Tensor) -> float:
    """
    Rank-based AUROC (Mann–Whitney U) for anomaly scores where higher =
    more anomalous.  1.0 = perfect separation, 0.5 = chance.
    """
    s_n = np.asarray(scores_normal.detach().cpu(), dtype=float).ravel()
    s_a = np.asarray(scores_anomalous.detach().cpu(), dtype=float).ravel()
    combined = np.concatenate([s_n, s_a])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ranks for ties
    for v in np.unique(combined):
        mask = combined == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum_anom = ranks[len(s_n):].sum()
    n_a, n_n = len(s_a), len(s_n)
    return float((rank_sum_anom - n_a * (n_a + 1) / 2) / (n_a * n_n))


def evaluate_detector(
    detector: AnomalyDetector,
    X_normal: torch.Tensor,
    X_anomalous: torch.Tensor,
    modality: Optional[str] = None,
) -> float:
    """AUROC of a fitted detector on held-out normal vs. anomalous data."""
    with torch.no_grad():
        return auroc(
            detector.score(X_normal, modality=modality),
            detector.score(X_anomalous, modality=modality),
        )


# ─── Direction 1: encoder + latent PC ───────────────────────────────────────

def default_mlp_encoder(in_dim: int, latent_dim: int, hidden: int = 64, seed: int = 0) -> nn.Module:
    """Small MLP encoder (random init = random-projection baseline)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, latent_dim),
    )


class LatentPCDetector(AnomalyDetector):
    """
    Direction 1.  Per-modality encoders into a shared latent space; one exact
    PC over the latents.  All modalities share the same latent PC, which is
    what makes the framework unified: one density model, many input types.

    By default encoders are FROZEN during fit: training an encoder to
    maximise latent likelihood collapses the latent distribution (the encoder
    can map everything to the densest point).  Set train_encoder=True only
    with an external safeguard (e.g. a pretrained encoder being fine-tuned
    with small lr, or an added reconstruction loss).

    Args:
        encoders:   modality name -> nn.Module mapping (B, d_in) -> (B, latent_dim),
                    or a single nn.Module (registered as modality "default").
        latent_dim: dimension of the shared latent space.
        vtree:      vtree over latent dims; random balanced if None.
        use_sos:    use the SquaredPC (SOS) density instead of the monotone one.
    """

    def __init__(
        self,
        encoders: Union[nn.Module, Dict[str, nn.Module]],
        latent_dim: int,
        vtree: Optional[VtreeNode] = None,
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
        use_sos: bool = False,
        train_encoder: bool = False,
        vtree_method: str = "random",
        seed: int = 0,
    ):
        super().__init__()
        if isinstance(encoders, nn.Module) and not isinstance(encoders, nn.ModuleDict):
            encoders = {"default": encoders}
        self.encoders = nn.ModuleDict(encoders)
        self.latent_dim = latent_dim
        self.train_encoder = train_encoder
        self.vtree_method = vtree_method

        def make_pc(vt: VtreeNode):
            if use_sos:
                return SquaredPC(vt, n_sum_components, leaf_factory, seed=seed)
            return DensityPC(vt, n_sum_components, leaf_factory)

        self._make_pc = make_pc
        self._seed = seed
        if vtree is None and vtree_method != "random":
            self.pc = None  # learned from the training latents at fit time
        else:
            vtree = vtree or random_balanced_vtree(list(range(latent_dim)), seed=seed)
            self.pc = make_pc(vtree)

    def _ensure_pc(self, latents: torch.Tensor) -> None:
        """Build the PC from a vtree learned on the latents (lazy;
        chow_liu / spectral / orc / forman)."""
        if self.pc is None:
            self.pc = self._make_pc(
                learned_vtree(latents, method=self.vtree_method, seed=self._seed))

    def encode(self, x: torch.Tensor, modality: str = "default") -> torch.Tensor:
        enc = self.encoders[modality]
        if self.train_encoder:
            return enc(x)
        with torch.no_grad():
            return enc(x)

    def fit(
        self,
        data: Union[torch.Tensor, TensorDict],
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> "LatentPCDetector":
        """
        Fit the shared latent PC on the pooled latents of all modalities.
        `data` is a tensor (single modality "default") or a dict
        modality -> tensor of raw inputs.
        """
        if isinstance(data, torch.Tensor):
            data = {"default": data}
        latents = torch.cat(
            [self.encode(x, m).detach() for m, x in data.items()], dim=0
        )
        self._ensure_pc(latents)
        self.pc.fit_leaves(latents)
        params = list(self.pc.parameters())
        if self.train_encoder:
            params += [p for enc in self.encoders.values() for p in enc.parameters()]
        fit_pc_nll(self.pc, latents, epochs=epochs, lr=lr,
                   batch_size=batch_size, parameters=params, verbose=verbose)
        return self

    # `fit` IS the unsupervised regime (α = 1, NLL only) — the primary target.
    fit_unsupervised = fit

    def fit_contrastive(
        self,
        data_pos: Union[torch.Tensor, TensorDict],
        data_neg: Union[torch.Tensor, TensorDict],
        epochs: int = 100,
        lr: float = 0.05,
        alpha: float = 0.5,
        margin: float = 1.0,
        batch_size: Optional[int] = None,
        train_encoder: bool = True,
        verbose: bool = False,
    ) -> "LatentPCDetector":
        """
        Supervised regime (Eq. 4 of arXiv:2605.05953): jointly optimise the
        PC parameters θ and — by default, as in the paper — the projection
        fϕ, on paired normal (pos) and anomalous (neg) raw inputs.  The
        margin term makes joint encoder training safe against latent
        collapse, so train_encoder defaults to True here (unlike the
        unsupervised `fit`).  Multimodal dicts are trained modality by
        modality against the shared latent PC.
        """
        if isinstance(data_pos, torch.Tensor):
            data_pos = {"default": data_pos}
        if isinstance(data_neg, torch.Tensor):
            data_neg = {"default": data_neg}
        missing = set(data_pos) - set(data_neg)
        if missing:
            raise KeyError(f"Negatives missing for modalities: {sorted(missing)}")

        with torch.no_grad():
            init_latents = torch.cat(
                [self.encoders[m](x) for m, x in data_pos.items()], dim=0
            )
        self._ensure_pc(init_latents)
        self.pc.fit_leaves(init_latents)

        params = list(self.pc.parameters())
        if train_encoder:
            params += [p for enc in self.encoders.values() for p in enc.parameters()]

        for m, X_pos in data_pos.items():
            enc = self.encoders[m]
            encode = enc if train_encoder else (lambda x, e=enc: e(x).detach())
            fit_pc_contrastive(
                self.pc, X_pos, data_neg[m],
                epochs=epochs, lr=lr, alpha=alpha, margin=margin,
                batch_size=batch_size, parameters=params, verbose=verbose,
                encode_pos=encode, encode_neg=encode,
            )
        return self

    def log_prob(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        if self.pc is None:
            raise RuntimeError(f"vtree_method={self.vtree_method!r} learns the "
                               "structure from data — call fit/fit_contrastive first")
        z = self.encode(x, modality or "default")
        return self.pc.log_prob(z)

    def validate(self) -> None:
        self.pc.validate()


# ─── Direction 2: encoder-free routed / vtree PC ────────────────────────────

class RoutedRawPC(AnomalyDetector):
    """
    Direction 2.  Exact density estimation directly over raw inputs: one
    sub-circuit per modality, routed by modality name.  No encoder anywhere —
    every score is an exact log-density of the raw data.

    Structure sharing (the transfer-learning bet): when all modalities have
    the same feature dimension and shared_structure=True, every sub-circuit
    is built from the SAME vtree object, so the scope-partition hierarchy —
    the structural knowledge — is shared across modalities while leaf/weight
    parameters stay modality-specific.

    Unknown modality: log_prob(x, modality=None) is the exact mixture
    log Σ_m π_m p_m(x) over the sub-circuits whose dimension matches x.

    Args:
        feature_dims: modality name -> raw feature dimension.
        vtrees:       modality -> vtree, or one vtree for all (must cover the
                      right features).  Defaults to a shared random balanced
                      vtree when dims agree, else per-modality random vtrees.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        vtrees: Optional[Union[VtreeNode, Dict[str, VtreeNode]]] = None,
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
        use_sos: bool = False,
        shared_structure: bool = True,
        vtree_method: str = "random",
        seed: int = 0,
    ):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.shared_structure = shared_structure
        self.vtree_method = vtree_method
        self.n_sum_components = n_sum_components
        self._seed = seed

        def make_pc(vt):
            if use_sos:
                return SquaredPC(vt, n_sum_components, leaf_factory, seed=seed)
            return DensityPC(vt, n_sum_components, leaf_factory)

        self._make_pc = make_pc

        if vtrees is None and vtree_method != "random":
            # structures are learned from the training data at fit time
            self.pcs = nn.ModuleDict()
        else:
            if isinstance(vtrees, dict):
                vtree_map = vtrees
            elif vtrees is not None:
                vtree_map = {m: vtrees for m in feature_dims}
            elif shared_structure and len(set(feature_dims.values())) == 1:
                d = next(iter(feature_dims.values()))
                shared = random_balanced_vtree(list(range(d)), seed=seed)
                vtree_map = {m: shared for m in feature_dims}
            else:
                vtree_map = {
                    m: random_balanced_vtree(list(range(d)), seed=seed + i)
                    for i, (m, d) in enumerate(feature_dims.items())
                }
            self.pcs = nn.ModuleDict({m: make_pc(vt) for m, vt in vtree_map.items()})

        # mixture weights over modalities, used when the modality is unknown
        self.register_buffer("log_priors", torch.full((len(feature_dims),),
                                                      -float(np.log(len(feature_dims)))))
        self._modalities = list(feature_dims)

    def _ensure_pcs(self, data: TensorDict) -> None:
        """Lazy structure learning (chow_liu / spectral / orc / forman): one
        vtree from the pooled data when modalities share their dimension
        (shared structure / transfer), otherwise one per modality."""
        if len(self.pcs) > 0:
            return
        if self.shared_structure and len(set(self.feature_dims.values())) == 1:
            pooled = torch.cat([data[m] for m in self._modalities], dim=0)
            shared = learned_vtree(pooled, method=self.vtree_method,
                                   seed=self._seed)
            vtree_map = {m: shared for m in self._modalities}
        else:
            vtree_map = {m: learned_vtree(data[m], method=self.vtree_method,
                                          seed=self._seed)
                         for m in self._modalities}
        for m, vt in vtree_map.items():
            self.pcs[m] = self._make_pc(vt)

    def fit(
        self,
        data: TensorDict,
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> "RoutedRawPC":
        """Fit each modality's sub-circuit by exact maximum likelihood."""
        self._ensure_pcs(data)
        counts = []
        for m in self._modalities:
            X = data[m]
            pc = self.pcs[m]
            pc.fit_leaves(X)
            fit_pc_nll(pc, X, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose)
            counts.append(X.shape[0])
        priors = torch.tensor(counts, dtype=torch.float32)
        self.log_priors = torch.log(priors / priors.sum())
        return self

    # `fit` IS the unsupervised regime (α = 1, NLL only) — the primary target.
    fit_unsupervised = fit

    def fit_contrastive(
        self,
        data_pos: TensorDict,
        data_neg: TensorDict,
        epochs: int = 100,
        lr: float = 0.05,
        alpha: float = 0.5,
        margin: float = 1.0,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> "RoutedRawPC":
        """
        Supervised regime (Eq. 4 of arXiv:2605.05953) per modality: each
        sub-circuit gets generative NLL on its normal data plus the margin
        penalty on that modality's labeled anomalies.  Modalities absent from
        data_neg fall back to plain unsupervised NLL.
        """
        self._ensure_pcs(data_pos)
        counts = []
        for m in self._modalities:
            X_pos = data_pos[m]
            pc = self.pcs[m]
            pc.fit_leaves(X_pos)
            if m in data_neg:
                fit_pc_contrastive(pc, X_pos, data_neg[m], epochs=epochs, lr=lr,
                                   alpha=alpha, margin=margin,
                                   batch_size=batch_size, verbose=verbose)
            else:
                fit_pc_nll(pc, X_pos, epochs=epochs, lr=lr,
                           batch_size=batch_size, verbose=verbose)
            counts.append(X_pos.shape[0])
        priors = torch.tensor(counts, dtype=torch.float32)
        self.log_priors = torch.log(priors / priors.sum())
        return self

    def add_modality(
        self,
        name: str,
        X: torch.Tensor,
        vtree: Optional[VtreeNode] = None,
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: Optional[int] = None,
        use_sos: bool = False,
        n_sum_components: Optional[int] = None,
        leaf_factory: Optional[LeafFactory] = None,
        verbose: bool = False,
    ) -> "RoutedRawPC":
        """
        Integrate a NEW dataset/modality without touching the existing
        sub-circuits: build one new PC — reusing the already-shared vtree
        when the dimension matches (structure transfer, zero structure
        search) — and train ONLY its parameters.  Every previously fitted
        modality keeps its weights bit-for-bit.
        """
        if name in self.pcs:
            raise KeyError(f"Modality {name!r} already exists")
        d = X.shape[1]
        if vtree is None:
            shared = [pc.vtree for m, pc in self.pcs.items()
                      if self.feature_dims[m] == d]
            if shared:
                vtree = shared[0]          # structure transfer, zero search
            elif self.vtree_method != "random":
                vtree = learned_vtree(X, method=self.vtree_method,
                                      seed=self._seed)  # from the new data
            else:
                vtree = random_balanced_vtree(list(range(d)), seed=len(self.pcs))
        k = n_sum_components or self.n_sum_components
        pc = (SquaredPC(vtree, k, leaf_factory) if use_sos
              else DensityPC(vtree, k, leaf_factory))
        pc.fit_leaves(X)
        fit_pc_nll(pc, X, epochs=epochs, lr=lr, batch_size=batch_size,
                   parameters=list(pc.parameters()), verbose=verbose)
        self.pcs[name] = pc
        self.feature_dims[name] = d
        self._modalities.append(name)
        # extend mixture priors: new modality gets a 1/n share, others rescale
        n = len(self._modalities)
        priors = torch.cat([torch.exp(self.log_priors) * (n - 1) / n,
                            torch.tensor([1.0 / n])])
        self.log_priors = torch.log(priors / priors.sum())
        return self

    def log_prob(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        if modality is not None:
            return self.pcs[modality].log_prob(x)
        # unknown modality: exact mixture over dimension-compatible circuits
        terms = []
        for i, m in enumerate(self._modalities):
            if self.feature_dims[m] == x.shape[1]:
                terms.append(self.log_priors[i] + self.pcs[m].log_prob(x))
        if not terms:
            raise ValueError(
                f"No sub-circuit matches input dimension {x.shape[1]}; "
                f"known dims: {self.feature_dims}"
            )
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)

    def validate(self) -> None:
        for pc in self.pcs.values():
            pc.validate()


def build_consensus_routed_pc(
    data: TensorDict,
    n_sum_components: int = 3,
    leaf_factory: Optional[LeafFactory] = None,
    pretrain_epochs: int = 30,
    epochs: int = 100,
    lr: float = 0.05,
    seed: int = 0,
) -> RoutedRawPC:
    """
    Direction 2 with a learned shared structure: fit a quick single-task PC
    per modality, extract each task's vtree, distil them into one consensus
    vtree (co-grouping matrix), then build and fit a RoutedRawPC where every
    modality shares the consensus structure.

    Requires all modalities to have the same (aligned) feature dimension —
    the critical alignment dependency from dev.md §3.2.
    """
    dims = {m: X.shape[1] for m, X in data.items()}
    if len(set(dims.values())) != 1:
        raise ValueError(f"Consensus structure needs aligned dimensions, got {dims}")
    d = next(iter(dims.values()))

    source_vtrees = []
    for i, (m, X) in enumerate(data.items()):
        vt = random_balanced_vtree(list(range(d)), seed=seed + i)
        pc = DensityPC(vt, n_sum_components, leaf_factory)
        pc.fit_leaves(X)
        fit_pc_nll(pc, X, epochs=pretrain_epochs, lr=lr)
        source_vtrees.append(single_task_vtree(pc))

    shared = consensus_vtree(source_vtrees, n_features=d, fallback_seed=seed)
    model = RoutedRawPC(dims, vtrees=shared, n_sum_components=n_sum_components,
                        leaf_factory=leaf_factory, seed=seed)
    return model.fit(data, epochs=epochs, lr=lr)


# ─── Direction 2 + ProbMoE: differentiable probabilistic routing ────────────

class ProbRoutedRawPC(RoutedRawPC):
    """
    Direction 2 with a ProbMoE-style cardinality-constrained router over the
    per-modality sub-circuits (Zhao, Shao, Van den Broeck, Zeng, ICML 2026).

    The encoder-free sub-circuits, their exact-NLL training, and the exact
    mixture density log p(x) = log Σ_m π_m p_m(x) are inherited UNCHANGED from
    RoutedRawPC, so the project's exactness invariant is untouched: `log_prob`
    /`score` remain the exact normalised density (the data-independent SumNode
    over experts).

    On top of that this class adds an INPUT-CONDITIONAL router whose logits are
    built only from the experts' own exact log-densities — no encoder, no extra
    network:
        r_i(x) = β · (log p_i(x) − mean_j log p_j(x)) + b_i,
    with a shared temperature β ≥ 0 and per-modality bias b_i.  Feeding these
    logits to the tractable cardinality machinery in probabilistic_circuits
    yields a family of EXACT routing queries that are anomaly signals distinct
    from (and meant to be benchmarked against) the scalar density — the F1
    "cardinality / routing posterior as a tractable query" idea:

      • cardinality_log_posterior / cardinality_moments → P(k|x), E[k|x],
        H[P(k|x)], MAP-k:  "how much of the model does x need?"
      • selection_marginals → m_j(x) = P(expert j fires | x): WHICH modality /
        sub-structure explains x (localisation), exact.
      • routing_shift → distance of m_j(x) from the training-time mean routing
        m̄_j: the recommended *detection* signal (its sign is well defined,
        unlike raw k*).

    The router is deliberately NOT part of the density (an input-dependent
    mixing weight would break normalisation); it is a separate exact query.

    Restriction: the router evaluates every expert on the same input, so it
    needs the routed sub-circuits to share their feature dimension (the aligned
    encoder-free / shared-vtree regime, as in build_consensus_routed_pc).  The
    density `log_prob` keeps RoutedRawPC's mixed-dimension fallback.

    Extra args:
        k_min, k_max: feasible cardinality range for the router (Dynamic-k).
                      Defaults to [1, N].  Set k_min=k_max=k for Exact-k.  FREEZE
                      these on a held-out modality before scoring — sweeping them
                      to maximise AUROC is p-hacking (see the F1 evaluation).
        beta_init:    initial router temperature β.
    """

    def __init__(self, *args, k_min: int = 1, k_max: Optional[int] = None,
                 beta_init: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_min = k_min
        self.k_max = k_max
        inv = float(np.log(np.expm1(max(beta_init, 1e-3))))  # softplus(raw)=β
        self.router_log_temp = nn.Parameter(torch.tensor(inv))
        self.router_bias = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(())) for m in self._modalities}
        )
        self._m_bar: Dict[str, float] = {}

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.router_log_temp)

    # ── router internals ──────────────────────────────────────────────────

    def _compatible(self, d: int) -> list:
        """Experts whose feature dimension matches d, in registration order."""
        return [m for m in self._modalities if self.feature_dims[m] == d]

    def _krange(self, n: int) -> tuple:
        kmax = self.k_max if self.k_max is not None else n
        return self.k_min, min(kmax, n)

    def expert_log_probs(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Stacked exact log-densities log p_m(x) of every dimension-compatible
        expert.  Returns ((B, N) tensor, list of N modality names).  This is
        the raw material for the cheap baselines (per-expert max log-lik,
        mixture-responsibility entropy) the F1 kill-shot must beat.
        """
        names = self._compatible(x.shape[1])
        if not names:
            raise ValueError(
                f"No sub-circuit matches input dimension {x.shape[1]}; "
                f"known dims: {self.feature_dims}")
        cols = [self.pcs[m].log_prob(x) for m in names]
        return torch.stack(cols, dim=1), names

    def router_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Encoder-free router logits r_i(x) = β·(log p_i(x) − mean_j log p_j(x))
        + b_i, shape (B, N).  Centring makes the logits scale-free in the
        absolute density level (ProbMoE-style competition among experts).
        """
        logp, names = self.expert_log_probs(x)
        centered = logp - logp.mean(dim=1, keepdim=True)
        bias = torch.stack([self.router_bias[m] for m in names])
        return self.beta * centered + bias, names

    # ── exact routing queries (anomaly signals) ─────────────────────────────

    def cardinality_log_posterior(self, x: torch.Tensor) -> torch.Tensor:
        """log P(|S| = k | x) over the feasible range, shape (B, k_max−k_min+1)."""
        logits, names = self.router_logits(x)
        kmin, kmax = self._krange(len(names))
        return cardinality_log_posterior(logits, kmin, kmax)

    def cardinality_moments(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """{'map','expected','entropy'} of P(k|x); each (B,).  All exact."""
        logits, names = self.router_logits(x)
        kmin, kmax = self._krange(len(names))
        return cardinality_moments(logits, kmin, kmax)

    def selection_marginals(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Exact m_j(x) = P(expert j fires | x), ((B, N), names).  Row sums = E[k|x].
        The localisation query: argmax_j m_j(x) is the modality the router holds
        most responsible for x.
        """
        logits, names = self.router_logits(x)
        kmin, kmax = self._krange(len(names))
        return selection_marginals(logits, kmin, kmax), names

    def mixture_responsibilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Plain mixture posterior γ_j(x) = π_j p_j(x) / Σ_l π_l p_l(x), ((B,N),names)
        — the GMM-responsibility baseline the cardinality marginals must beat to
        justify the routing machinery (see the F1 evaluation's redundancy risk).
        """
        logp, names = self.expert_log_probs(x)
        idx = [self._modalities.index(m) for m in names]
        log_pri = self.log_priors[idx].to(logp.dtype)
        log_joint = logp + log_pri.unsqueeze(0)
        return torch.softmax(log_joint, dim=1), names

    def expert_argmax(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """argmax_j log p_j(x): the no-router localisation baseline. ((B,),names)."""
        logp, names = self.expert_log_probs(x)
        return torch.argmax(logp, dim=1), names

    def localize(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Most-responsible expert per input under the exact marginals. ((B,),names)."""
        m, names = self.selection_marginals(x)
        return torch.argmax(m, dim=1), names

    # ── detection scores (higher = more anomalous) ──────────────────────────

    def routing_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """H[P(k|x)] (nats): flat cardinality posterior ⇒ more anomalous."""
        return self.cardinality_moments(x)["entropy"]

    def routing_shift(self, x: torch.Tensor, kind: str = "l1") -> torch.Tensor:
        """
        Distance of the per-input routing distribution m_j(x)/E[k|x] from the
        training-time mean routing m̄ (computed at fit).  Larger ⇒ x routes
        unlike normal data ⇒ more anomalous.  kind ∈ {'l1','kl'}.  Falls back to
        a uniform reference if fit has not populated m̄ for these experts.
        """
        m, names = self.selection_marginals(x)
        p = m / m.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if all(n in self._m_bar for n in names):
            ref = torch.tensor([self._m_bar[n] for n in names], dtype=p.dtype)
        else:
            ref = torch.full((len(names),), 1.0 / len(names), dtype=p.dtype)
        ref = (ref / ref.sum()).unsqueeze(0)
        if kind == "kl":
            return (p * (p.clamp_min(1e-12).log() - ref.log())).sum(dim=1)
        return (p - ref).abs().sum(dim=1)

    def routing_score(self, x: torch.Tensor, signal: str = "shift") -> torch.Tensor:
        """
        A single routing-based anomaly score, higher = more anomalous.
        signal ∈ {'shift','entropy','expected_k','neg_max_marginal'}.  These are
        meant to be combined with / tested against `score` (= −log p(x)) in the
        F1 detection kill-shot, NOT to replace it.
        """
        if signal == "shift":
            return self.routing_shift(x)
        if signal == "entropy":
            return self.routing_entropy(x)
        if signal == "expected_k":
            return self.cardinality_moments(x)["expected"]
        if signal == "neg_max_marginal":
            m, _ = self.selection_marginals(x)
            return -m.max(dim=1).values
        raise ValueError(f"unknown routing signal {signal!r}")

    # ── fitting: experts trained exactly as before, then cache m̄ ───────────

    def _cache_mean_routing(self, data: TensorDict) -> None:
        """Store m̄_j = mean_x m_j(x)/E[k|x] over training data (the routing_shift
        reference).  Pools modalities sharing the scored dimension."""
        with torch.no_grad():
            by_dim: Dict[int, list] = {}
            for m in self._modalities:
                by_dim.setdefault(self.feature_dims[m], []).append(data[m])
            for d, batches in by_dim.items():
                X = torch.cat(batches, dim=0)
                mj, names = self.selection_marginals(X)
                p = mj / mj.sum(dim=1, keepdim=True).clamp_min(1e-12)
                mean = p.mean(dim=0)
                for n, v in zip(names, mean):
                    self._m_bar[n] = float(v)

    def fit(self, data: TensorDict, *args, **kwargs) -> "ProbRoutedRawPC":
        super().fit(data, *args, **kwargs)
        self._cache_mean_routing(data)
        return self

    fit_unsupervised = fit

    def fit_contrastive(self, data_pos: TensorDict, *args, **kwargs) -> "ProbRoutedRawPC":
        super().fit_contrastive(data_pos, *args, **kwargs)
        self._cache_mean_routing(data_pos)
        return self

    def add_modality(self, name: str, X: torch.Tensor, *args, **kwargs) -> "ProbRoutedRawPC":
        super().add_modality(name, X, *args, **kwargs)
        self.router_bias[name] = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            mj, names = self.selection_marginals(X)
            p = (mj / mj.sum(dim=1, keepdim=True).clamp_min(1e-12)).mean(dim=0)
            for n, v in zip(names, p):
                self._m_bar[n] = float(v)
        return self
