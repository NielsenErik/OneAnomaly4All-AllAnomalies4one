"""
Dataset integration for the multimodal anomaly-detection pipeline.

Three sources, one common format (AnomalyDataset):

  - ADBench tabular datasets (github.com/Minqi824/ADBench): each dataset is a
    single small .npz fetched lazily and cached under data/adbench/ — the
    benchmark is never downloaded wholesale.
  - MVTec-AD images (huggingface.co/datasets/Voxel51/mvtec-ad): the tiny
    samples.json manifest is fetched first, then ONLY the images of the
    requested category are downloaded (HF hub cache), not the full ~5 GB.
  - Text anomaly detection: one-class 20 Newsgroups via sklearn's built-in
    fetcher (small, package-managed download).

Featurizers map each raw modality into the SHARED latent space (fixed
dimension D) the multimodal PC operates on:

  tabular  → standardize + frozen seeded random projection
  image    → frozen CNN backbone (torchvision resnet18) → standardize + projection
  text     → TF-IDF + truncated SVD → standardize

Featurizer fitting is closed-form preprocessing on the new dataset's normal
split (statistics, SVD); it never touches the PC's parameters.  This is the
alignment step (dev.md §3.2) that makes a single shared-density model usable
across datasets without retraining.

Heavy/optional imports (sklearn, torchvision, PIL, huggingface_hub, requests)
are lazy, so the core package works without them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

RawData = Union[torch.Tensor, List[str]]

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ═══════════════════════════════════════════════════════════════════════════
# Common format
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyDataset:
    """
    One anomaly-detection task in the standard protocol: train on normal data
    only, evaluate on a contaminated test set.

      X_train: normal-only raw data (tensor, or list of strings for text)
      X_test:  mixed raw data
      y_test:  1 = anomaly, 0 = normal
    """
    name: str
    modality: str  # "tabular" | "image" | "text"
    X_train: RawData
    X_test: RawData
    y_test: torch.Tensor
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        n_tr = len(self.X_train)
        n_te = len(self.X_test)
        return (f"AnomalyDataset({self.name!r}, {self.modality}, "
                f"train={n_tr}, test={n_te}, "
                f"anomaly_rate={float(self.y_test.float().mean()):.2%})")


def _split_normal_train(
    X: np.ndarray, y: np.ndarray, train_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard semi-supervised AD split: train on a fraction of the normals,
    test on the remaining normals plus all anomalies."""
    rng = np.random.default_rng(seed)
    normal_idx = np.flatnonzero(y == 0)
    anom_idx = np.flatnonzero(y == 1)
    rng.shuffle(normal_idx)
    n_train = int(len(normal_idx) * train_ratio)
    train_idx = normal_idx[:n_train]
    test_idx = np.concatenate([normal_idx[n_train:], anom_idx])
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[test_idx]


# ═══════════════════════════════════════════════════════════════════════════
# 1. ADBench (tabular)
# ═══════════════════════════════════════════════════════════════════════════

_ADBENCH_API = ("https://api.github.com/repos/Minqi824/ADBench/contents/"
                "adbench/datasets/Classical")
_ADBENCH_RAW = ("https://github.com/Minqi824/ADBench/raw/main/"
                "adbench/datasets/Classical/{filename}")


def list_adbench(data_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Map friendly dataset name (lowercase, e.g. 'thyroid') -> npz filename
    (e.g. '38_thyroid.npz').  The index is fetched once from the GitHub API
    and cached on disk, so repeated calls are offline.
    """
    data_dir = data_dir or os.path.join(DEFAULT_DATA_DIR, "adbench")
    os.makedirs(data_dir, exist_ok=True)
    index_path = os.path.join(data_dir, "_index.json")
    if not os.path.exists(index_path):
        import requests

        resp = requests.get(_ADBENCH_API, timeout=30)
        resp.raise_for_status()
        files = [e["name"] for e in resp.json() if e["name"].endswith(".npz")]
        with open(index_path, "w") as f:
            json.dump(files, f)
    with open(index_path) as f:
        files = json.load(f)
    # "38_thyroid.npz" -> "thyroid"
    return {fn.split("_", 1)[1].rsplit(".", 1)[0].lower(): fn for fn in files}


def load_adbench(
    name: str,
    data_dir: Optional[str] = None,
    train_ratio: float = 0.5,
    seed: int = 0,
) -> AnomalyDataset:
    """
    Load a single ADBench tabular dataset by friendly name ('thyroid',
    'cardio', …).  Only that dataset's .npz (typically a few MB) is
    downloaded, then cached under data/adbench/.
    """
    data_dir = data_dir or os.path.join(DEFAULT_DATA_DIR, "adbench")
    os.makedirs(data_dir, exist_ok=True)
    index = list_adbench(data_dir)
    key = name.lower()
    if key not in index:
        raise KeyError(f"Unknown ADBench dataset {name!r}. Available: {sorted(index)}")
    filename = index[key]
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        import requests

        url = _ADBENCH_RAW.format(filename=filename)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)

    with np.load(path, allow_pickle=True) as npz:
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64).reshape(-1)

    X_tr, X_te, y_te = _split_normal_train(X, y, train_ratio, seed)
    return AnomalyDataset(
        name=f"adbench:{key}",
        modality="tabular",
        X_train=torch.from_numpy(X_tr),
        X_test=torch.from_numpy(X_te),
        y_test=torch.from_numpy(y_te),
        meta={"n_features": X.shape[1], "file": filename},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. MVTec-AD (images, HF Voxel51/mvtec-ad)
# ═══════════════════════════════════════════════════════════════════════════

_MVTEC_REPO = "Voxel51/mvtec-ad"

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def _select_mvtec_samples(samples: List[dict], category: str) -> Tuple[List[dict], List[dict]]:
    """
    Pure selection logic over the samples.json manifest: returns
    (train_good, test_all) entries for one category.  Each entry keeps
    filepath, split and defect label.
    """
    picked_train, picked_test = [], []
    for s in samples:
        if s.get("category", {}).get("label") != category:
            continue
        entry = {
            "filepath": s["filepath"],
            "defect": s.get("defect", {}).get("label", "unknown"),
            "split": s.get("split"),
        }
        if entry["split"] == "train" and entry["defect"] == "good":
            picked_train.append(entry)
        elif entry["split"] == "test":
            picked_test.append(entry)
    return picked_train, picked_test


def _load_image_tensor(paths: Sequence[str], image_size: int) -> torch.Tensor:
    from PIL import Image

    out = torch.empty(len(paths), 3, image_size, image_size)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((image_size, image_size))
        out[i] = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return out


def load_mvtec(
    category: str,
    image_size: int = 224,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 0,
) -> AnomalyDataset:
    """
    Load ONE MVTec-AD category from huggingface.co/datasets/Voxel51/mvtec-ad.

    Downloads the small samples.json manifest, then fetches only that
    category's images through the HF hub cache (so nothing is re-downloaded
    across runs, and the other 14 categories are never touched).

    Protocol: train = 'good' images of the train split; test = the full test
    split, y = 1 for any defect.
    """
    if category not in MVTEC_CATEGORIES:
        raise KeyError(f"Unknown MVTec category {category!r}. Available: {MVTEC_CATEGORIES}")
    from huggingface_hub import hf_hub_download

    manifest = hf_hub_download(_MVTEC_REPO, "samples.json", repo_type="dataset")
    with open(manifest) as f:
        samples = json.load(f)["samples"]

    train_entries, test_entries = _select_mvtec_samples(samples, category)
    rng = np.random.default_rng(seed)
    if max_train is not None and len(train_entries) > max_train:
        train_entries = [train_entries[i] for i in rng.permutation(len(train_entries))[:max_train]]
    if max_test is not None and len(test_entries) > max_test:
        test_entries = [test_entries[i] for i in rng.permutation(len(test_entries))[:max_test]]

    def fetch(entries):
        return [hf_hub_download(_MVTEC_REPO, e["filepath"], repo_type="dataset")
                for e in entries]

    X_train = _load_image_tensor(fetch(train_entries), image_size)
    X_test = _load_image_tensor(fetch(test_entries), image_size)
    y_test = torch.tensor([0 if e["defect"] == "good" else 1 for e in test_entries])
    return AnomalyDataset(
        name=f"mvtec:{category}",
        modality="image",
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        meta={"image_size": image_size,
              "defects": sorted({e["defect"] for e in test_entries})},
    )


def load_mvtec_local(
    root: str, category: str, image_size: int = 224
) -> AnomalyDataset:
    """
    Same protocol from a standard local MVTec-AD directory layout
    (root/<category>/train/good/*.png, root/<category>/test/<defect>/*.png) —
    for machines where the data is already on disk.
    """
    cat_dir = os.path.join(root, category)
    train_paths = sorted(
        os.path.join(cat_dir, "train", "good", f)
        for f in os.listdir(os.path.join(cat_dir, "train", "good"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    test_paths, y = [], []
    test_dir = os.path.join(cat_dir, "test")
    for defect in sorted(os.listdir(test_dir)):
        ddir = os.path.join(test_dir, defect)
        if not os.path.isdir(ddir):
            continue
        for f in sorted(os.listdir(ddir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                test_paths.append(os.path.join(ddir, f))
                y.append(0 if defect == "good" else 1)
    return AnomalyDataset(
        name=f"mvtec:{category}",
        modality="image",
        X_train=_load_image_tensor(train_paths, image_size),
        X_test=_load_image_tensor(test_paths, image_size),
        y_test=torch.tensor(y),
        meta={"image_size": image_size, "root": root},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Text anomaly detection (one-class 20 Newsgroups)
# ═══════════════════════════════════════════════════════════════════════════

TEXT_TOPIC_GROUPS = {
    "comp": ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
             "comp.sys.mac.hardware", "comp.windows.x"],
    "rec": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "sci": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "talk": ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
             "talk.religion.misc"],
}


def load_text_ad(
    inlier_group: str = "sci",
    max_train: int = 1500,
    max_test: int = 1000,
    anomaly_ratio: float = 0.1,
    seed: int = 0,
) -> AnomalyDataset:
    """
    One-class text anomaly detection on 20 Newsgroups (the standard CVDD /
    text-AD protocol): documents of one topic group are normal; documents
    from all other groups are anomalies.  Fetched through sklearn's built-in
    package fetcher (small, cached by sklearn).
    """
    if inlier_group not in TEXT_TOPIC_GROUPS:
        raise KeyError(f"Unknown topic group {inlier_group!r}. Available: {sorted(TEXT_TOPIC_GROUPS)}")
    from sklearn.datasets import fetch_20newsgroups

    strip = ("headers", "footers", "quotes")
    inlier_cats = TEXT_TOPIC_GROUPS[inlier_group]
    train = fetch_20newsgroups(subset="train", categories=inlier_cats, remove=strip)
    test_in = fetch_20newsgroups(subset="test", categories=inlier_cats, remove=strip)
    outlier_cats = [c for g, cats in TEXT_TOPIC_GROUPS.items() if g != inlier_group for c in cats]
    test_out = fetch_20newsgroups(subset="test", categories=outlier_cats, remove=strip)

    rng = np.random.default_rng(seed)

    def sample(texts, k):
        texts = [t for t in texts if t.strip()]
        idx = rng.permutation(len(texts))[:k]
        return [texts[i] for i in idx]

    X_train = sample(list(train.data), max_train)
    n_anom = int(max_test * anomaly_ratio)
    test_normal = sample(list(test_in.data), max_test - n_anom)
    test_anom = sample(list(test_out.data), n_anom)
    X_test = test_normal + test_anom
    y_test = torch.tensor([0] * len(test_normal) + [1] * len(test_anom))
    return AnomalyDataset(
        name=f"text:{inlier_group}",
        modality="text",
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        meta={"inlier_categories": inlier_cats},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Featurizers: raw modality → shared D-dim latent space
# ═══════════════════════════════════════════════════════════════════════════

class Featurizer:
    """
    Fit-once, frozen mapping raw data → (N, out_dim) latents.  Fitting is
    closed-form preprocessing (statistics / SVD) on a dataset's normal split;
    it never involves the PC.  All featurizers end with standardization, so
    every dataset lands in the shared latent space on a comparable scale —
    the alignment step that lets one density model serve many datasets.
    """

    modality: str = "abstract"

    def __init__(self, out_dim: int, seed: int = 0):
        self.out_dim = out_dim
        self.seed = seed
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    # subclasses implement: _embed(raw) -> (N, d_embed) tensor, and fit()
    def _embed(self, raw: RawData) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, raw_train: RawData) -> "Featurizer":
        emb = self._embed(raw_train)
        self._fit_projection(emb.shape[1])
        proj = self._project(emb)
        self._mean = proj.mean(dim=0)
        self._std = proj.std(dim=0) + 1e-6
        return self

    def transform(self, raw: RawData) -> torch.Tensor:
        if self._mean is None:
            raise RuntimeError("Featurizer.fit must be called first")
        proj = self._project(self._embed(raw))
        return (proj - self._mean) / self._std

    # frozen seeded projection d_embed -> out_dim (identity when equal)
    def _fit_projection(self, d_embed: int) -> None:
        if d_embed == self.out_dim:
            self._proj = None
            return
        gen = torch.Generator().manual_seed(self.seed)
        W = torch.randn(d_embed, self.out_dim, generator=gen) / np.sqrt(d_embed)
        self._proj = W

    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        if self._proj is None:
            return emb
        return emb @ self._proj


class TabularFeaturizer(Featurizer):
    """Standardize + frozen random projection to the shared dimension."""

    modality = "tabular"

    def _embed(self, raw: RawData) -> torch.Tensor:
        return torch.as_tensor(raw, dtype=torch.float32)


class ImageFeaturizer(Featurizer):
    """
    Frozen torchvision resnet18 (final-pool features, 512-d) followed by the
    shared projection + standardization.  pretrained=True downloads the
    backbone weights once through torchvision's cache; pretrained=False uses
    a random (still frozen, still seeded) backbone — a weaker but fully
    offline baseline.
    """

    modality = "image"

    def __init__(self, out_dim: int, seed: int = 0, pretrained: bool = True,
                 batch_size: int = 64):
        super().__init__(out_dim, seed)
        self.pretrained = pretrained
        self.batch_size = batch_size
        self._backbone = None

    def _get_backbone(self):
        if self._backbone is None:
            import torchvision

            torch.manual_seed(self.seed)
            weights = torchvision.models.ResNet18_Weights.DEFAULT if self.pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            net.fc = torch.nn.Identity()
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._backbone = net
        return self._backbone

    def _embed(self, raw: RawData) -> torch.Tensor:
        net = self._get_backbone()
        X = torch.as_tensor(raw, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        outs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                batch = (X[start:start + self.batch_size] - mean) / std
                outs.append(net(batch))
        return torch.cat(outs, dim=0)


class TextFeaturizer(Featurizer):
    """TF-IDF + truncated SVD (LSA) to the shared dimension, standardized."""

    modality = "text"

    def __init__(self, out_dim: int, seed: int = 0, max_features: int = 5000):
        super().__init__(out_dim, seed)
        self.max_features = max_features
        self._vectorizer = None
        self._svd = None

    def fit(self, raw_train: RawData) -> "TextFeaturizer":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features, sublinear_tf=True, stop_words="english"
        )
        tfidf = self._vectorizer.fit_transform(list(raw_train))
        n_comp = min(self.out_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        emb = torch.tensor(self._svd.fit_transform(tfidf), dtype=torch.float32)
        # pad if the corpus is too small for out_dim components
        self._pad = self.out_dim - emb.shape[1]
        self._proj = None
        padded = self._pad_emb(emb)
        self._mean = padded.mean(dim=0)
        self._std = padded.std(dim=0) + 1e-6
        return self

    def _pad_emb(self, emb: torch.Tensor) -> torch.Tensor:
        if self._pad > 0:
            return torch.cat([emb, torch.zeros(emb.shape[0], self._pad)], dim=1)
        return emb

    def transform(self, raw: RawData) -> torch.Tensor:
        if self._vectorizer is None:
            raise RuntimeError("Featurizer.fit must be called first")
        tfidf = self._vectorizer.transform(list(raw))
        emb = torch.tensor(self._svd.transform(tfidf), dtype=torch.float32)
        return (self._pad_emb(emb) - self._mean) / self._std


def featurizer_for(modality: str, out_dim: int, seed: int = 0, **kwargs) -> Featurizer:
    """Factory: the right featurizer for a dataset's modality."""
    if modality == "tabular":
        return TabularFeaturizer(out_dim, seed=seed, **kwargs)
    if modality == "image":
        return ImageFeaturizer(out_dim, seed=seed, **kwargs)
    if modality == "text":
        return TextFeaturizer(out_dim, seed=seed, **kwargs)
    raise KeyError(f"Unknown modality {modality!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Spec parsing ("adbench:thyroid", "mvtec:bottle", "text:sci")
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset_spec(spec: str, **kwargs) -> AnomalyDataset:
    """Load a dataset from a 'source:name' spec string."""
    source, _, name = spec.partition(":")
    if source == "adbench":
        return load_adbench(name, **kwargs)
    if source == "mvtec":
        return load_mvtec(name, **kwargs)
    if source == "text":
        return load_text_ad(name or "sci", **kwargs)
    raise KeyError(f"Unknown dataset source {source!r} (use adbench:/mvtec:/text:)")
