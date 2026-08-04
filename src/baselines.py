"""
Literature baselines for anomaly detection, behind the same scoring interface
as the PC detectors (fit on normal data → score, higher = more anomalous), so
every baseline drops into the same evaluation/aggregation machinery.

Registry (BASELINE_REGISTRY) — references:

  iforest      Isolation Forest.
               Liu, Ting, Zhou. "Isolation Forest." ICDM 2008.
               https://doi.org/10.1109/ICDM.2008.17
  lof          Local Outlier Factor.
               Breunig, Kriegel, Ng, Sander. "LOF: Identifying Density-Based
               Local Outliers." SIGMOD 2000.
               https://doi.org/10.1145/342009.335388
  ocsvm        One-Class SVM.
               Schölkopf, Platt, Shawe-Taylor, Smola, Williamson. "Estimating
               the Support of a High-Dimensional Distribution." Neural
               Computation 13(7), 2001 (also NeurIPS 1999).
               https://doi.org/10.1162/089976601750264965
  knn          k-NN distance outlier score.
               Ramaswamy, Rastogi, Shim. "Efficient Algorithms for Mining
               Outliers from Large Data Sets." SIGMOD 2000.
               https://doi.org/10.1145/342009.335437
  gmm          Gaussian Mixture Model NLL (parametric density baseline; EM).
               Dempster, Laird, Rubin. "Maximum Likelihood from Incomplete
               Data via the EM Algorithm." JRSS-B 39(1), 1977.  Used as an AD
               baseline in Han et al., "ADBench: Anomaly Detection
               Benchmark." NeurIPS 2022 D&B. https://arxiv.org/abs/2206.09426
  mahalanobis  Mahalanobis distance to the normal-data Gaussian.
               Mahalanobis. "On the Generalised Distance in Statistics."
               Proc. Nat. Inst. Sci. India 2(1), 1936.  Revived for deep OOD
               by Lee, Lee, Lee, Shin. NeurIPS 2018.
               https://arxiv.org/abs/1807.03888
  pca          PCA reconstruction error.
               Shyu, Chen, Sarinnapakorn, Chang. "A Novel Anomaly Detection
               Scheme Based on Principal Component Classifier." ICDM
               Workshop on Foundations and New Directions of Data Mining, 2003.
  ecod         Empirical-CDF tail probabilities (parameter-free).
               Li, Zhao, Hu, Botta, Ionescu, Chen. "ECOD: Unsupervised
               Outlier Detection Using Empirical Cumulative Distribution
               Functions." IEEE TKDE 35(12), 2022.
               https://arxiv.org/abs/2201.00382
  ae           Autoencoder reconstruction error.
               Hawkins, He, Williams, Baxter. "Outlier Detection Using
               Replicator Neural Networks." DaWaK 2002; Sakurada, Yairi.
               "Anomaly Detection Using Autoencoders with Nonlinear
               Dimensionality Reduction." MLSDA @ ACM PRICAI 2014.
               https://doi.org/10.1145/2689746.2689747
  deep_svdd    Deep Support Vector Data Description.
               Ruff, Vandermeulen, Görnitz, Deecke, Siddiqui, Binder,
               Müller, Kloft. "Deep One-Class Classification." ICML 2018.
               https://proceedings.mlr.press/v80/ruff18a.html

Image-only zero-/few-shot baselines (input_type="image": they consume RAW
images, not featurized latents — the evaluation glue routes accordingly):

  winclip      Zero-/few-shot CLIP prompt-ensemble anomaly scoring.
               Jeong, Zou, Kim, Zhang, Ravichandran, Dabeer. "WinCLIP:
               Zero-/Few-Shot Anomaly Classification and Segmentation."
               CVPR 2023. https://arxiv.org/abs/2303.14814
               Faithful at image level: two-class (normal/anomalous) state +
               template prompt ensemble, multi-window crops, optional
               few-shot normal memory bank (WinCLIP+).
  anomalyclip  Object-agnostic CLIP prompt scoring — "lite" variant.
               Zhou, Pang, Tian, He, Chen. "AnomalyCLIP: Object-agnostic
               Prompt Learning for Zero-shot Anomaly Detection." ICLR 2024.
               https://arxiv.org/abs/2310.18961
               NOTE: full AnomalyCLIP LEARNS its prompt vectors on auxiliary
               AD data; this implementation reproduces the object-agnostic
               inference scheme with fixed generic prompts (no training), so
               treat reported numbers as a lower bound of the full method.
  anomalygpt   LVLM-based industrial AD — integration adapter only.
               Gu, Zhu, Zhu, Chen, Tang, Wang. "AnomalyGPT: Detecting
               Industrial Anomalies Using Large Vision-Language Models."
               AAAI 2024. https://arxiv.org/abs/2308.15366
               Requires the official released checkpoint + repo (multi-GB
               LVLM); this class wires it in when available and raises with
               setup instructions otherwise — it is NOT re-implemented here.

Survey context for baseline choice: Han et al., NeurIPS 2022 (ADBench) and
Ruff et al., "A Unifying Review of Deep and Shallow Anomaly Detection,"
Proceedings of the IEEE 109(5), 2021. https://arxiv.org/abs/2009.11732

Usage:
    from src.baselines import make_baseline, evaluate_baselines
    bl = make_baseline("iforest", seed=0).fit(X_train)
    scores = bl.score(X_test)
    rows = evaluate_baselines(ds, names=["iforest", "lof", "gmm"], seed=0)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .datasets import AnomalyDataset, Featurizer, featurizer_for
from .directions import auroc


def _np(X) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


class Baseline:
    """fit(X_train_normal) → score(X), higher = more anomalous.

    input_type: "features" (works on tabular data / featurized latents) or
    "image" (consumes raw image tensors (N, 3, H, W) in [0, 1])."""

    name: str = "abstract"
    input_type: str = "features"

    def fit(self, X) -> "Baseline":
        raise NotImplementedError

    def score(self, X) -> torch.Tensor:
        raise NotImplementedError


# ─── sklearn-backed classical baselines ─────────────────────────────────────

class _SklearnBaseline(Baseline):
    """Wraps estimators exposing score_samples (higher = more normal)."""

    def __init__(self, estimator, name: str):
        self.est = estimator
        self.name = name

    def fit(self, X) -> "Baseline":
        self.est.fit(_np(X))
        return self

    def score(self, X) -> torch.Tensor:
        return torch.tensor(-self.est.score_samples(_np(X)), dtype=torch.float32)


def _iforest(seed: int = 0, n_estimators: int = 200, **kw) -> Baseline:
    from sklearn.ensemble import IsolationForest

    return _SklearnBaseline(
        IsolationForest(n_estimators=n_estimators, random_state=seed, **kw), "iforest")


def _lof(seed: int = 0, n_neighbors: int = 20, **kw) -> Baseline:
    from sklearn.neighbors import LocalOutlierFactor

    # deterministic given the data; `seed` accepted for interface uniformity
    return _SklearnBaseline(
        LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, **kw), "lof")


def _ocsvm(seed: int = 0, nu: float = 0.1, gamma: str = "scale", **kw) -> Baseline:
    from sklearn.svm import OneClassSVM

    return _SklearnBaseline(OneClassSVM(nu=nu, gamma=gamma, **kw), "ocsvm")


def _gmm(seed: int = 0, n_components: int = 8, **kw) -> Baseline:
    from sklearn.mixture import GaussianMixture

    return _SklearnBaseline(
        GaussianMixture(n_components=n_components, covariance_type="full",
                        random_state=seed, reg_covar=1e-4, **kw), "gmm")


class KNNDistance(Baseline):
    """Ramaswamy et al. 2000: distance to the k-th (here: mean of k) nearest
    training neighbors."""

    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X) -> "Baseline":
        from sklearn.neighbors import NearestNeighbors

        self.nn = NearestNeighbors(n_neighbors=self.k).fit(_np(X))
        return self

    def score(self, X) -> torch.Tensor:
        dist, _ = self.nn.kneighbors(_np(X))
        return torch.tensor(dist.mean(axis=1), dtype=torch.float32)


class MahalanobisBaseline(Baseline):
    """Mahalanobis 1936 / Lee et al. 2018: squared distance to the training
    Gaussian with shrinkage-regularized covariance."""

    name = "mahalanobis"

    def fit(self, X) -> "Baseline":
        X = _np(X)
        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, X) -> torch.Tensor:
        D = _np(X) - self.mu
        return torch.tensor(np.einsum("bi,ij,bj->b", D, self.prec, D),
                            dtype=torch.float32)


class PCAReconstruction(Baseline):
    """Shyu et al. 2003 (principal component classifier): eigenvalue-
    normalized score over the major components (Mahalanobis in the retained
    subspace) plus the variance-normalized residual of the minor subspace.
    Plain reconstruction error alone would be blind to anomalies that shift
    the mean inside the retained subspace."""

    name = "pca"

    def __init__(self, n_components: float = 0.9):
        self.n_components = n_components

    def fit(self, X) -> "Baseline":
        from sklearn.decomposition import PCA

        X = _np(X)
        n_comp = self.n_components
        if isinstance(n_comp, float):
            n_comp = min(n_comp, 0.999)
        self.pca = PCA(n_components=n_comp).fit(X)
        rec = self.pca.inverse_transform(self.pca.transform(X))
        self.resid_var = ((X - rec) ** 2).sum(axis=1).mean() + 1e-12
        return self

    def score(self, X) -> torch.Tensor:
        X = _np(X)
        Y = self.pca.transform(X)
        major = (Y ** 2 / (self.pca.explained_variance_ + 1e-12)).sum(axis=1)
        rec = self.pca.inverse_transform(Y)
        minor = ((X - rec) ** 2).sum(axis=1) / self.resid_var
        return torch.tensor(major + minor, dtype=torch.float32)


class ECOD(Baseline):
    """Li et al., TKDE 2022: aggregate −log empirical tail probabilities per
    dimension; parameter-free."""

    name = "ecod"

    def fit(self, X) -> "Baseline":
        self.train = _np(X)
        self.skew = _skewness(self.train)
        return self

    def score(self, X) -> torch.Tensor:
        X = _np(X)
        n = self.train.shape[0]
        # empirical left/right tail probabilities against the training ECDF
        left = (self.train[None, :, :] <= X[:, None, :]).sum(axis=1) / n
        right = (self.train[None, :, :] >= X[:, None, :]).sum(axis=1) / n
        eps = 1.0 / n
        o_l = -np.log(np.clip(left, eps, 1)).sum(axis=1)
        o_r = -np.log(np.clip(right, eps, 1)).sum(axis=1)
        auto = np.where(self.skew < 0,
                        -np.log(np.clip(left, eps, 1)),
                        -np.log(np.clip(right, eps, 1))).sum(axis=1)
        return torch.tensor(np.maximum.reduce([o_l, o_r, auto]), dtype=torch.float32)


def _skewness(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    return (((X - mu) / sd) ** 3).mean(axis=0)


# ─── Deep baselines (torch) ─────────────────────────────────────────────────

class AutoencoderBaseline(Baseline):
    """Hawkins et al. 2002 / Sakurada & Yairi 2014: MLP autoencoder trained
    on normals; anomaly score = reconstruction MSE."""

    name = "ae"

    def __init__(self, hidden: int = 64, bottleneck: int = 8,
                 epochs: int = 100, lr: float = 1e-3, seed: int = 0,
                 device=None):
        self.hidden, self.bottleneck = hidden, bottleneck
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.device = torch.device(device) if device else torch.device("cpu")

    def fit(self, X) -> "Baseline":
        torch.manual_seed(self.seed)
        X = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        d = X.shape[1]
        self.net = nn.Sequential(
            nn.Linear(d, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.bottleneck), nn.ReLU(),
            nn.Linear(self.bottleneck, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, d),
        ).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            loss = ((self.net(X) - X) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def score(self, X) -> torch.Tensor:
        X = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return ((self.net(X) - X) ** 2).sum(dim=1).cpu()


class DeepSVDD(Baseline):
    """Ruff et al., ICML 2018: map normals close to a fixed center c in
    feature space; score = squared distance to c.  Bias-free layers, as in
    the paper, to avoid the trivial collapsed solution."""

    name = "deep_svdd"

    def __init__(self, hidden: int = 64, rep_dim: int = 16,
                 epochs: int = 100, lr: float = 1e-3, seed: int = 0,
                 device=None):
        self.hidden, self.rep_dim = hidden, rep_dim
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.device = torch.device(device) if device else torch.device("cpu")

    def fit(self, X) -> "Baseline":
        torch.manual_seed(self.seed)
        X = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        d = X.shape[1]
        self.net = nn.Sequential(
            nn.Linear(d, self.hidden, bias=False), nn.ReLU(),
            nn.Linear(self.hidden, self.rep_dim, bias=False),
        ).to(self.device)
        with torch.no_grad():
            self.c = self.net(X).mean(dim=0)
            # keep the center away from 0 (paper's collapse safeguard)
            self.c[self.c.abs() < 0.1] = 0.1
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            loss = ((self.net(X) - self.c) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def score(self, X) -> torch.Tensor:
        X = torch.as_tensor(_np(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return ((self.net(X) - self.c) ** 2).sum(dim=1).cpu()


# ─── CLIP-based image baselines (WinCLIP / AnomalyCLIP-lite / AnomalyGPT) ───

class ClipBackend:
    """
    Thin wrapper around a HF CLIP model (lazy download via transformers).
    Tests inject a fake object exposing the same two methods.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16",
                 batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPTokenizer

            self._model = CLIPModel.from_pretrained(self.model_name).eval()
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        return self._model, self._tokenizer

    def embed_images(self, X: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [0, 1] → L2-normalized (N, D)."""
        model, _ = self._load()
        size = model.config.vision_config.image_size
        mean = torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.CLIP_STD).view(1, 3, 1, 1)
        outs = []
        with torch.no_grad():
            for s in range(0, X.shape[0], self.batch_size):
                batch = X[s:s + self.batch_size].float()
                batch = torch.nn.functional.interpolate(
                    batch, size=(size, size), mode="bilinear", align_corners=False)
                emb = model.get_image_features(pixel_values=(batch - mean) / std)
                outs.append(emb)
        emb = torch.cat(outs, dim=0)
        return emb / emb.norm(dim=-1, keepdim=True)

    def embed_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        """list of prompts → L2-normalized (P, D)."""
        model, tok = self._load()
        with torch.no_grad():
            inputs = tok(list(prompts), padding=True, return_tensors="pt")
            emb = model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)


# Prompt ensemble in the spirit of WinCLIP's compositional prompt ensemble
_PROMPT_TEMPLATES = [
    "a photo of a {state} {cls}.",
    "a cropped photo of a {state} {cls}.",
    "a close-up photo of a {state} {cls}.",
    "a bright photo of a {state} {cls}.",
    "a photo of a small {state} {cls}.",
    "a photo of a large {state} {cls}.",
]
_NORMAL_STATES = ["flawless", "perfect", "normal", "unblemished"]
_ANOMALOUS_STATES = ["damaged", "broken", "defective", "anomalous", "flawed"]


class WinCLIPBaseline(Baseline):
    """
    WinCLIP (Jeong et al., CVPR 2023), image-level scoring: two-class
    state+template prompt ensemble vs CLIP image embeddings, multi-window
    crops aggregated by max, and (k_shot > 0, "WinCLIP+") a few-shot memory
    bank of normal-image embeddings averaged into the score.
    """

    name = "winclip"
    input_type = "image"

    def __init__(self, class_name: str = "object", k_shot: int = 0,
                 grid: int = 2, logit_scale: float = 100.0,
                 backend: Optional[ClipBackend] = None, seed: int = 0,
                 model_name: str = "openai/clip-vit-base-patch16"):
        self.class_name = class_name
        self.k_shot = k_shot
        self.grid = grid
        self.logit_scale = logit_scale
        self.backend = backend or ClipBackend(model_name)
        self.seed = seed
        self._memory: Optional[torch.Tensor] = None
        self._text: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _prompts(self, states) -> List[str]:
        return [t.format(state=s, cls=self.class_name)
                for t in _PROMPT_TEMPLATES for s in states]

    def _text_anchors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._text is None:
            normal = self.backend.embed_texts(self._prompts(_NORMAL_STATES)).mean(0)
            anom = self.backend.embed_texts(self._prompts(_ANOMALOUS_STATES)).mean(0)
            self._text = (normal / normal.norm(), anom / anom.norm())
        return self._text

    def _windows(self, X: torch.Tensor) -> List[torch.Tensor]:
        """Full image plus a grid×grid crop pyramid."""
        views = [X]
        if self.grid > 1:
            H, W = X.shape[-2:]
            hs, ws = H // self.grid, W // self.grid
            for i in range(self.grid):
                for j in range(self.grid):
                    views.append(X[..., i * hs:(i + 1) * hs, j * ws:(j + 1) * ws])
        return views

    def _text_score(self, emb: torch.Tensor) -> torch.Tensor:
        """p(anomalous) from the two-class prompt softmax (per WinCLIP Eq. 1)."""
        normal, anom = self._text_anchors()
        logits = self.logit_scale * torch.stack(
            [emb @ normal, emb @ anom], dim=-1)
        return torch.softmax(logits, dim=-1)[..., 1]

    def fit(self, X) -> "Baseline":
        """Zero-shot: no-op. k_shot > 0: build the normal memory bank."""
        if self.k_shot > 0:
            X = torch.as_tensor(X)[: self.k_shot]
            self._memory = self.backend.embed_images(X)
        return self

    def score(self, X) -> torch.Tensor:
        X = torch.as_tensor(X)
        window_scores = []
        for view in self._windows(X):
            emb = self.backend.embed_images(view)
            s = self._text_score(emb)
            if self._memory is not None:
                # WinCLIP+: combine with distance to the normal memory bank
                mem = (1.0 - (emb @ self._memory.T).max(dim=-1).values) / 2.0
                s = (s + mem) / 2.0
            window_scores.append(s)
        return torch.stack(window_scores, dim=0).max(dim=0).values


class AnomalyCLIPLiteBaseline(WinCLIPBaseline):
    """
    AnomalyCLIP (Zhou et al., ICLR 2024) — object-AGNOSTIC prompt scoring.
    LITE variant: reproduces the object-agnostic inference scheme with fixed
    generic prompts ("... normal object" / "... damaged object"); the full
    method learns the prompt vectors on auxiliary AD data, so this is a
    lower bound of published AnomalyCLIP numbers.
    """

    name = "anomalyclip"

    def __init__(self, k_shot: int = 0, grid: int = 2,
                 logit_scale: float = 100.0,
                 backend: Optional[ClipBackend] = None, seed: int = 0,
                 model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__(class_name="object", k_shot=k_shot, grid=grid,
                         logit_scale=logit_scale, backend=backend, seed=seed,
                         model_name=model_name)


class AnomalyGPTBaseline(Baseline):
    """
    AnomalyGPT (Gu et al., AAAI 2024) — integration adapter, NOT a
    re-implementation: the method is a fine-tuned multi-GB LVLM
    (ImageBind encoder + Vicuna + prompt learner).  Point `checkpoint_path`
    at the official release (https://github.com/CASIA-IVA-Lab/AnomalyGPT)
    with the repo importable; otherwise fit/score raise with instructions.
    """

    name = "anomalygpt"
    input_type = "image"

    def __init__(self, checkpoint_path: Optional[str] = None, seed: int = 0):
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _require(self):
        if self.checkpoint_path is None:
            raise RuntimeError(
                "AnomalyGPT is an external LVLM and is not re-implemented "
                "here. Download the official checkpoint and repo from "
                "https://github.com/CASIA-IVA-Lab/AnomalyGPT, then pass "
                "checkpoint_path= (and make the repo importable) — e.g. "
                "baseline_kwargs={'anomalygpt': {'checkpoint_path': '...'}}.")
        try:
            from anomalygpt.model import AnomalyGPTDetector  # official repo
        except ImportError as e:
            raise RuntimeError(
                "AnomalyGPT checkpoint given but the official repo is not "
                "importable — clone https://github.com/CASIA-IVA-Lab/AnomalyGPT "
                "and add it to PYTHONPATH.") from e
        if self._model is None:
            self._model = AnomalyGPTDetector(self.checkpoint_path)
        return self._model

    def fit(self, X) -> "Baseline":
        self._require()
        return self

    def score(self, X) -> torch.Tensor:
        model = self._require()
        return torch.tensor(model.score_images(_np(X)), dtype=torch.float32)


# ─── Registry + evaluation glue ─────────────────────────────────────────────

BASELINE_REGISTRY: Dict[str, Callable[..., Baseline]] = {
    "iforest": _iforest,
    "lof": _lof,
    "ocsvm": _ocsvm,
    "knn": lambda seed=0, **kw: KNNDistance(**kw),
    "gmm": _gmm,
    "mahalanobis": lambda seed=0, **kw: MahalanobisBaseline(**kw),
    "pca": lambda seed=0, **kw: PCAReconstruction(**kw),
    "ecod": lambda seed=0, **kw: ECOD(**kw),
    "ae": lambda seed=0, **kw: AutoencoderBaseline(seed=seed, **kw),
    "deep_svdd": lambda seed=0, **kw: DeepSVDD(seed=seed, **kw),
    # image-only zero-/few-shot baselines (consume raw images)
    "winclip": lambda seed=0, **kw: WinCLIPBaseline(seed=seed, **kw),
    "anomalyclip": lambda seed=0, **kw: AnomalyCLIPLiteBaseline(seed=seed, **kw),
    "anomalygpt": lambda seed=0, **kw: AnomalyGPTBaseline(seed=seed, **kw),
}


def make_baseline(name: str, seed: int = 0, **kwargs) -> Baseline:
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](seed=seed, **kwargs)


def baseline_input_type(name: str) -> str:
    """'features' or 'image' — what raw input the named baseline consumes."""
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name]().input_type


def evaluate_baselines(
    ds: AnomalyDataset,
    names: Optional[Sequence[str]] = None,
    featurizer: Optional[Featurizer] = None,
    latent_dim: int = 64,
    seed: int = 0,
    baseline_kwargs: Optional[dict] = None,
) -> List[dict]:
    """
    Fit each baseline on the dataset's normal split and report AUROC on its
    test split — the same protocol as the PC detectors, so rows are directly
    comparable (and aggregate with the same logging machinery: the baseline
    name goes into the 'adaptation' column, role='baseline').

    Non-tabular datasets are featurized first (same featurizers as the
    pipeline; pass the pipeline's own featurizer for an exact apples-to-
    apples comparison in latent space).  Image-input baselines (WinCLIP,
    AnomalyCLIP, AnomalyGPT) bypass the featurizer and receive raw images;
    WinCLIP additionally receives the dataset's class name for its prompts
    (from ds.name, e.g. "mvtec:metal_nut" → "metal nut") unless overridden
    via baseline_kwargs.
    """
    names = list(names) if names is not None else sorted(BASELINE_REGISTRY)
    kw = baseline_kwargs or {}

    feat_train = feat_test = None

    def featurized():
        nonlocal featurizer, feat_train, feat_test
        if feat_train is None:
            if featurizer is None and ds.modality != "tabular":
                featurizer = featurizer_for(ds.modality, latent_dim, seed=seed)
                featurizer.fit(ds.X_train)
            if featurizer is not None:
                feat_train = featurizer.transform(ds.X_train)
                feat_test = featurizer.transform(ds.X_test)
            else:
                feat_train, feat_test = ds.X_train, ds.X_test
        return feat_train, feat_test

    rows = []
    for name in names:
        bl_kw = dict(kw.get(name, {}))
        proto = BASELINE_REGISTRY.get(name)
        if proto is not None and name in ("winclip",) and "class_name" not in bl_kw:
            bl_kw["class_name"] = ds.name.split(":")[-1].replace("_", " ")
        bl = make_baseline(name, seed=seed, **bl_kw)
        if bl.input_type == "image":
            if ds.modality != "image":
                raise ValueError(
                    f"Baseline {name!r} consumes raw images but dataset "
                    f"{ds.name!r} has modality {ds.modality!r}")
            X_train, X_test = ds.X_train, ds.X_test
        else:
            X_train, X_test = featurized()
        bl.fit(X_train)
        scores = bl.score(X_test)
        rows.append({
            "dataset": ds.name,
            "modality": ds.modality,
            "adaptation": name,
            "role": "baseline",
            "auroc": auroc(scores[ds.y_test == 0], scores[ds.y_test == 1]),
            "n_test": int(ds.y_test.numel()),
            "seed": seed,
        })
    return rows
