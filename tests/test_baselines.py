"""
Tests for the literature baselines (src/baselines.py): every registered
baseline must train on normals only, score with higher-=-more-anomalous
orientation, and separate an easy synthetic anomaly cluster.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines import BASELINE_REGISTRY, evaluate_baselines, make_baseline
from src.directions import auroc
from tests.test_datasets_pipeline import synthetic_tabular

torch.manual_seed(0)


def easy_data(d=8, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    X_train = torch.randn(n, d, generator=g)
    X_in = torch.randn(150, d, generator=g)
    X_out = torch.randn(60, d, generator=g) * 0.5 + 5.0
    return X_train, X_in, X_out


def test_registry_has_at_least_five():
    assert len(BASELINE_REGISTRY) >= 5
    # the five canonical classical methods are all present
    assert {"iforest", "lof", "ocsvm", "knn", "gmm"} <= set(BASELINE_REGISTRY)


def test_unknown_baseline_raises():
    with pytest.raises(KeyError, match="Unknown baseline"):
        make_baseline("supermodel9000")


FEATURE_BASELINES = sorted(
    n for n, f in BASELINE_REGISTRY.items() if f().input_type == "features"
)


@pytest.mark.parametrize("name", FEATURE_BASELINES)
def test_baseline_separates_easy_anomalies(name):
    X_train, X_in, X_out = easy_data(seed=hash(name) % 1000)
    kw = {"epochs": 50} if name in ("ae", "deep_svdd") else {}
    bl = make_baseline(name, seed=0, **kw)
    bl.fit(X_train)
    s_in, s_out = bl.score(X_in), bl.score(X_out)
    assert s_in.shape == (150,) and torch.isfinite(s_in).all()
    a = auroc(s_in, s_out)
    assert a > 0.9, f"{name}: AUROC {a:.3f} on trivially separable data"


def test_baseline_seed_reproducible():
    X_train, X_in, _ = easy_data()
    s1 = make_baseline("ae", seed=3, epochs=20).fit(X_train).score(X_in)
    s2 = make_baseline("ae", seed=3, epochs=20).fit(X_train).score(X_in)
    assert torch.allclose(s1, s2)


def test_evaluate_baselines_rows():
    ds = synthetic_tabular(d=8, seed=30, name="bl_ds")
    rows = evaluate_baselines(ds, names=["iforest", "knn", "mahalanobis"],
                              seed=0)
    assert len(rows) == 3
    for r in rows:
        assert r["role"] == "baseline"
        assert r["dataset"] == "bl_ds"
        assert r["adaptation"] in {"iforest", "knn", "mahalanobis"}
        assert r["auroc"] > 0.9      # synthetic anomalies are far
        assert r["seed"] == 0


def test_evaluate_baselines_kwargs_passthrough():
    ds = synthetic_tabular(d=6, seed=31, name="bl_kw")
    rows = evaluate_baselines(ds, names=["knn"],
                              baseline_kwargs={"knn": {"k": 3}})
    assert rows[0]["auroc"] > 0.9


# ─── CLIP-based image baselines (offline: fake backend) ──────────────────────

class FakeClipBackend:
    """Deterministic stand-in for ClipBackend: image embedding = direction
    set by mean pixel intensity; 'damaged/...' prompts anchor at +e1,
    normal-state prompts at −e1.  Bright images → anomalous."""

    dim = 8

    def embed_images(self, X):
        m = torch.as_tensor(X).float().mean(dim=(1, 2, 3))
        emb = torch.zeros(len(m), self.dim)
        emb[:, 0] = 2.0 * m - 1.0          # in [−1, 1] as mean goes 0→1
        emb[:, 1] = 1.0
        return emb / emb.norm(dim=-1, keepdim=True)

    def embed_texts(self, prompts):
        from src.baselines import _ANOMALOUS_STATES

        emb = torch.zeros(len(prompts), self.dim)
        for i, p in enumerate(prompts):
            anom = any(s in p for s in _ANOMALOUS_STATES)
            emb[i, 0] = 1.0 if anom else -1.0
        return emb / emb.norm(dim=-1, keepdim=True)


def fake_image_data(n=24, hw=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    normal = (torch.rand(n, 3, hw, hw, generator=g) * 0.2)          # dark
    anom = 0.8 + torch.rand(n // 2, 3, hw, hw, generator=g) * 0.2   # bright
    return normal, anom


def test_registry_includes_clip_baselines():
    assert {"winclip", "anomalyclip", "anomalygpt"} <= set(BASELINE_REGISTRY)
    assert make_baseline("winclip", backend=FakeClipBackend()).input_type == "image"


def test_winclip_zero_shot_scores_and_separates():
    normal, anom = fake_image_data()
    bl = make_baseline("winclip", backend=FakeClipBackend(), class_name="widget")
    bl.fit(normal)                          # zero-shot: no-op
    s_in, s_out = bl.score(normal), bl.score(anom)
    assert s_in.shape == (24,) and torch.isfinite(s_in).all()
    assert (s_in >= 0).all() and (s_in <= 1).all()
    assert auroc(s_in, s_out) > 0.95


def test_winclip_few_shot_memory_bank():
    normal, anom = fake_image_data(seed=1)
    bl = make_baseline("winclip", backend=FakeClipBackend(), k_shot=8)
    bl.fit(normal)
    assert bl._memory is not None and bl._memory.shape[0] == 8
    assert auroc(bl.score(normal), bl.score(anom)) > 0.95


def test_anomalyclip_lite_is_object_agnostic():
    normal, anom = fake_image_data(seed=2)
    bl = make_baseline("anomalyclip", backend=FakeClipBackend())
    assert bl.class_name == "object"        # no class name needed
    bl.fit(normal)
    assert auroc(bl.score(normal), bl.score(anom)) > 0.95


def test_anomalygpt_requires_checkpoint():
    bl = make_baseline("anomalygpt")
    with pytest.raises(RuntimeError, match="checkpoint"):
        bl.fit(torch.rand(2, 3, 16, 16))


def test_evaluate_baselines_routes_raw_images():
    from src.datasets import AnomalyDataset

    normal, anom = fake_image_data(seed=3)
    ds = AnomalyDataset(
        name="mvtec:metal_nut", modality="image",
        X_train=normal,
        X_test=torch.cat([normal[:12], anom]),
        y_test=torch.tensor([0] * 12 + [1] * len(anom)),
    )
    rows = evaluate_baselines(ds, names=["winclip"],
                              baseline_kwargs={"winclip": {"backend": FakeClipBackend()}})
    assert rows[0]["adaptation"] == "winclip"
    assert rows[0]["auroc"] > 0.95


def test_baseline_input_type_helper():
    from src.baselines import baseline_input_type

    assert baseline_input_type("iforest") == "features"
    assert baseline_input_type("winclip") == "image"
    assert baseline_input_type("anomalygpt") == "image"
    with pytest.raises(KeyError):
        baseline_input_type("nope")


def test_image_baseline_rejects_tabular_dataset():
    ds = synthetic_tabular(d=6, seed=32, name="tab_ds")
    with pytest.raises(ValueError, match="raw images"):
        evaluate_baselines(ds, names=["winclip"],
                           baseline_kwargs={"winclip": {"backend": FakeClipBackend()}})
