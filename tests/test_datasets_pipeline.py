"""
Offline tests for dataset integration (src/datasets.py) and the multimodal
pipeline (src/experiment.py).

No network access: ADBench parsing runs on a locally written fake .npz, the
MVTec manifest selection on a fake samples.json structure, the local MVTec
loader on synthetic PNGs, and the image featurizer with a random
(non-pretrained) backbone.  The key invariant under test: integrating a new
dataset must NOT mutate the shared PC.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets import (
    AnomalyDataset,
    TabularFeaturizer,
    TextFeaturizer,
    _select_mvtec_samples,
    _split_normal_train,
    featurizer_for,
    load_adbench,
    load_mvtec_local,
)
from src.directions import RoutedRawPC
from src.experiment import MultimodalPipeline
from src.probabilistic_circuits import GaussianLeaf

torch.manual_seed(0)


def synthetic_tabular(d=10, n=300, shift=0.0, seed=0, name="synth", anomaly_shift=6.0):
    g = torch.Generator().manual_seed(seed)
    X_train = torch.randn(n, d, generator=g) + shift
    X_test_n = torch.randn(n // 2, d, generator=g) + shift
    X_test_a = torch.randn(n // 4, d, generator=g) * 0.5 + shift + anomaly_shift
    X_test = torch.cat([X_test_n, X_test_a])
    y = torch.cat([torch.zeros(len(X_test_n)), torch.ones(len(X_test_a))]).long()
    return AnomalyDataset(name=name, modality="tabular",
                          X_train=X_train, X_test=X_test, y_test=y)


# ─── Split protocol ───────────────────────────────────────────────────────────

def test_split_normal_train_protocol():
    X = np.arange(100, dtype=np.float32).reshape(-1, 1)
    y = np.array([0] * 80 + [1] * 20)
    X_tr, X_te, y_te = _split_normal_train(X, y, train_ratio=0.5, seed=0)
    assert len(X_tr) == 40                      # half of the normals
    assert len(X_te) == 60                      # rest of normals + all anomalies
    assert y_te.sum() == 20                     # all anomalies are in test
    # train contains only normals (normal values are 0..79)
    assert X_tr.max() < 80


# ─── ADBench loader (offline: pre-seeded cache, no download) ─────────────────

def test_load_adbench_from_cache(tmp_path):
    data_dir = str(tmp_path)
    # fake index + fake npz so no network is touched
    with open(os.path.join(data_dir, "_index.json"), "w") as f:
        json.dump(["99_faketask.npz"], f)
    X = np.random.randn(200, 7).astype(np.float32)
    y = np.array([0] * 180 + [1] * 20)
    np.savez(os.path.join(data_dir, "99_faketask.npz"), X=X, y=y)

    ds = load_adbench("faketask", data_dir=data_dir, train_ratio=0.5, seed=0)
    assert ds.modality == "tabular"
    assert ds.name == "adbench:faketask"
    assert ds.X_train.shape == (90, 7)          # 50% of 180 normals
    assert int(ds.y_test.sum()) == 20
    assert ds.meta["n_features"] == 7


def test_load_adbench_unknown_name(tmp_path):
    data_dir = str(tmp_path)
    with open(os.path.join(data_dir, "_index.json"), "w") as f:
        json.dump(["99_faketask.npz"], f)
    with pytest.raises(KeyError, match="Unknown ADBench dataset"):
        load_adbench("nope", data_dir=data_dir)


# ─── MVTec manifest selection + local-folder loader ──────────────────────────

def _fake_sample(category, split, defect, path):
    return {"filepath": path, "split": split,
            "category": {"label": category}, "defect": {"label": defect}}


def test_select_mvtec_samples():
    samples = [
        _fake_sample("bottle", "train", "good", "data/a/0.png"),
        _fake_sample("bottle", "test", "good", "data/a/1.png"),
        _fake_sample("bottle", "test", "broken_large", "data/a/2.png"),
        _fake_sample("cable", "train", "good", "data/a/3.png"),
        _fake_sample("bottle", "train", "scratch", "data/a/4.png"),  # never happens, but: excluded
    ]
    train, test = _select_mvtec_samples(samples, "bottle")
    assert [e["filepath"] for e in train] == ["data/a/0.png"]
    assert {e["defect"] for e in test} == {"good", "broken_large"}


def test_load_mvtec_local(tmp_path):
    from PIL import Image

    root = tmp_path / "mvtec"
    for sub, n in [("bottle/train/good", 6), ("bottle/test/good", 3),
                   ("bottle/test/crack", 2)]:
        d = root / sub
        d.mkdir(parents=True)
        for i in range(n):
            arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")

    ds = load_mvtec_local(str(root), "bottle", image_size=16)
    assert ds.modality == "image"
    assert ds.X_train.shape == (6, 3, 16, 16)
    assert ds.X_test.shape == (5, 3, 16, 16)
    assert int(ds.y_test.sum()) == 2
    assert float(ds.X_train.max()) <= 1.0


# ─── Featurizers ─────────────────────────────────────────────────────────────

def test_tabular_featurizer_shape_and_standardization():
    f = TabularFeaturizer(out_dim=8, seed=0)
    X = torch.randn(300, 20) * 5 + 3
    Z = f.fit(X).transform(X)
    assert Z.shape == (300, 8)
    assert torch.allclose(Z.mean(0), torch.zeros(8), atol=0.1)
    assert torch.allclose(Z.std(0), torch.ones(8), atol=0.1)


def test_tabular_featurizer_identity_when_dims_match():
    f = TabularFeaturizer(out_dim=5, seed=0)
    X = torch.randn(100, 5)
    Z = f.fit(X).transform(X)
    assert Z.shape == (100, 5)


def test_tabular_featurizer_deterministic_per_seed():
    X = torch.randn(50, 12)
    Z1 = TabularFeaturizer(8, seed=3).fit(X).transform(X)
    Z2 = TabularFeaturizer(8, seed=3).fit(X).transform(X)
    assert torch.allclose(Z1, Z2)


def test_text_featurizer():
    docs = [f"the quick brown fox jumps over dog number {i}" for i in range(30)] + \
           [f"probabilistic circuits compute exact marginals case {i}" for i in range(30)]
    f = TextFeaturizer(out_dim=16, seed=0, max_features=100)
    Z = f.fit(docs).transform(docs)
    assert Z.shape == (60, 16)
    assert torch.isfinite(Z).all()
    # unseen documents also transform
    Z2 = f.transform(["exact marginals of the brown fox"])
    assert Z2.shape == (1, 16)


def test_image_featurizer_random_backbone():
    f = featurizer_for("image", 8, seed=0, pretrained=False)
    X = torch.rand(6, 3, 32, 32)
    Z = f.fit(X).transform(X)
    assert Z.shape == (6, 8)
    assert torch.isfinite(Z).all()


def test_featurizer_unknown_modality():
    with pytest.raises(KeyError, match="Unknown modality"):
        featurizer_for("audio", 8)


# ─── MultimodalPipeline: train once, integrate new datasets cheaply ──────────

def test_pipeline_fit_and_zero_shot_new_dataset():
    src_a = synthetic_tabular(d=10, seed=1, name="src_a")
    src_b = synthetic_tabular(d=15, seed=2, name="src_b")
    new = synthetic_tabular(d=12, seed=3, name="new")

    pipe = MultimodalPipeline(latent_dim=8, n_sum_components=2,
                              leaf_factory=GaussianLeaf, seed=0)
    pipe.fit([src_a, src_b], epochs=40)
    pipe.validate()

    params_before = [p.clone() for p in pipe.pc.parameters()]
    res = pipe.evaluate(new, "zero_shot")
    assert res["auroc"] > 0.9                   # anomalies are far; latents shift
    # zero-shot must not touch the shared PC
    for a, b in zip(params_before, pipe.pc.parameters()):
        assert torch.equal(a, b)


def test_pipeline_leaves_adaptation_does_not_mutate_shared_pc():
    src = synthetic_tabular(d=10, seed=4, name="src")
    new = synthetic_tabular(d=20, shift=2.0, seed=5, name="new")
    pipe = MultimodalPipeline(latent_dim=8, n_sum_components=2,
                              leaf_factory=GaussianLeaf, seed=0)
    pipe.fit([src], epochs=30)
    params_before = [p.clone() for p in pipe.pc.parameters()]
    res = pipe.evaluate(new, "leaves")
    assert res["auroc"] > 0.9
    for a, b in zip(params_before, pipe.pc.parameters()):
        assert torch.equal(a, b)                # adaptation happened on a copy


def test_pipeline_finetune_adaptation():
    src = synthetic_tabular(d=10, seed=6, name="src")
    new = synthetic_tabular(d=10, seed=7, name="new")
    pipe = MultimodalPipeline(latent_dim=8, n_sum_components=2,
                              leaf_factory=GaussianLeaf, seed=0)
    pipe.fit([src], epochs=30)
    res = pipe.evaluate(new, "finetune", finetune_epochs=5)
    assert res["auroc"] > 0.9


def test_pipeline_unknown_mode():
    src = synthetic_tabular(d=6, seed=8)
    pipe = MultimodalPipeline(latent_dim=4, n_sum_components=2,
                              leaf_factory=GaussianLeaf)
    pipe.fit([src], epochs=5)
    with pytest.raises(KeyError, match="Unknown adaptation mode"):
        pipe.evaluate(src, "full_retrain")


def test_pipeline_featurizers_are_cached_per_dataset():
    src = synthetic_tabular(d=6, seed=9, name="cached")
    pipe = MultimodalPipeline(latent_dim=4, n_sum_components=2,
                              leaf_factory=GaussianLeaf)
    pipe.fit([src], epochs=5)
    f1 = pipe.featurizers["cached"]
    pipe.evaluate(src, "zero_shot")
    assert pipe.featurizers["cached"] is f1


def test_pipeline_multimodal_tabular_plus_text():
    tab = synthetic_tabular(d=10, seed=10, name="tab")
    docs_n = [f"normal operating report number {i} all sensors fine" for i in range(80)]
    docs_a = [f"catastrophic meltdown alert {i} reactor breach" for i in range(20)]
    txt = AnomalyDataset(
        name="txt", modality="text",
        X_train=docs_n,
        X_test=docs_n[:40] + docs_a,
        y_test=torch.tensor([0] * 40 + [1] * 20),
    )
    pipe = MultimodalPipeline(latent_dim=6, n_sum_components=2,
                              leaf_factory=GaussianLeaf)
    pipe.fit([tab, txt], epochs=30)
    r_tab = pipe.evaluate(tab, "zero_shot")
    r_txt = pipe.evaluate(txt, "leaves")
    assert r_tab["auroc"] > 0.8
    # toy 60-doc corpus: out-of-vocabulary anomalies collapse toward the SVD
    # origin, so only partial separation is achievable — this is a plumbing
    # smoke test, not a benchmark claim
    assert r_txt["auroc"] > 0.65


# ─── Direction 2: add_modality without retraining existing circuits ──────────

def test_add_modality_preserves_existing_circuits():
    d = 8
    Xa = torch.randn(200, d)
    det = RoutedRawPC({"a": d}, n_sum_components=2, leaf_factory=GaussianLeaf)
    det.fit({"a": Xa}, epochs=30)
    params_a_before = [p.clone() for p in det.pcs["a"].parameters()]

    Xb = torch.randn(200, d) + 3.0
    det.add_modality("b", Xb, epochs=30)

    # existing sub-circuit untouched, bit-for-bit
    for a, b in zip(params_a_before, det.pcs["a"].parameters()):
        assert torch.equal(a, b)
    # new sub-circuit reuses the shared structure object (structure transfer)
    assert det.pcs["b"].vtree is det.pcs["a"].vtree
    # priors extended and normalized
    assert abs(float(torch.exp(det.log_priors).sum()) - 1.0) < 1e-5
    # and the new modality actually works
    with torch.no_grad():
        s_in = det.score(Xb, modality="b")
        s_out = det.score(torch.randn(50, d) - 5.0, modality="b")
    assert float(s_out.mean()) > float(s_in.mean())


def test_add_modality_duplicate_name_raises():
    det = RoutedRawPC({"a": 4}, n_sum_components=2, leaf_factory=GaussianLeaf)
    with pytest.raises(KeyError, match="already exists"):
        det.add_modality("a", torch.randn(10, 4), epochs=1)
