"""
Multimodal anomaly-detection experiment pipeline.

The point of the multimodal design: the PC is trained ONCE on a set of source
datasets; integrating a NEW dataset afterwards must not require heavy
retraining.  Three adaptation levels, in increasing cost:

  zero_shot  No PC training at all.  The new dataset only gets a featurizer
             (closed-form preprocessing: statistics / SVD on its normal
             split), then is scored by the existing shared density model.
  leaves     No gradient steps.  Leaf parameters of a COPY of the PC are
             re-initialized on the new latents via the closed-form
             median/MAD fit; sum weights and structure are reused.
  finetune   A few NLL epochs on a COPY of the PC (worst case, still light:
             structure and initialization are inherited).

The shared PC itself is never mutated by evaluation — adaptation happens on
copies, so previously integrated datasets keep their scores.

Direction 2 equivalent: RoutedRawPC.add_modality trains one new sub-circuit
on the shared vtree (structure transfer) without touching existing ones.

CLI:
    python -m src.experiment --train adbench:cardio text:sci \\
                             --test adbench:thyroid mvtec:bottle \\
                             --adapt zero_shot leaves --latent-dim 64
"""
from __future__ import annotations

import argparse
import copy
from typing import Dict, List, Optional, Sequence

import torch

from .datasets import AnomalyDataset, Featurizer, featurizer_for, load_dataset_spec
from .directions import (
    ProbRoutedRawPC,
    RoutedRawPC,
    auroc,
    fit_pc_nll,
)
from .logging_utils import (
    ExperimentLogger,
    aggregate_results,
    format_summary_table,
    save_summary,
    set_global_seed,
)
from .probabilistic_circuits import (
    DensityPC,
    LeafFactory,
    SquaredPC,
    VTREE_METHODS,
    learned_vtree,
    random_balanced_vtree,
)

ADAPTATION_MODES = ("zero_shot", "leaves", "finetune")
DIRECTIONS = ("encoder", "encoder_free")
ROUTING_SIGNALS = ("shift", "entropy", "expected_k", "neg_max_marginal")


class MultimodalPipeline:
    """
    One shared exact density model over a fixed D-dimensional latent space;
    per-dataset featurizers project every modality into that space.

    Usage:
        pipe = MultimodalPipeline(latent_dim=64)
        pipe.fit([ds_a, ds_b])                       # train ONCE
        res = pipe.evaluate(ds_new, "zero_shot")     # new dataset, no retraining
        res = pipe.evaluate(ds_new, "leaves")        # closed-form leaf re-init
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
        use_sos: bool = False,
        vtree_method: str = "chow_liu",
        seed: int = 0,
        image_pretrained: bool = True,
    ):
        self.latent_dim = latent_dim
        self.seed = seed
        self.image_pretrained = image_pretrained
        self.vtree_method = vtree_method

        def make_pc(vt):
            if use_sos:
                return SquaredPC(vt, n_sum_components, leaf_factory, seed=seed)
            return DensityPC(vt, n_sum_components, leaf_factory)

        self._make_pc = make_pc
        if vtree_method == "random":
            self.pc = make_pc(random_balanced_vtree(list(range(latent_dim)), seed=seed))
        elif vtree_method in VTREE_METHODS:
            # learned structure (chow_liu / spectral / orc / forman): built at
            # fit time from the pooled source latents, so dependent latent
            # dims share deep sub-circuits instead of being split at random
            self.pc = None
        else:
            raise KeyError(f"Unknown vtree_method {vtree_method!r} "
                           f"(use one of {VTREE_METHODS})")
        self.featurizers: Dict[str, Featurizer] = {}
        self.trained_on: List[str] = []

    # ── featurization (closed-form, per dataset) ─────────────────────────

    def _featurizer(self, ds: AnomalyDataset) -> Featurizer:
        """Get or fit the featurizer for a dataset (fit on its normal split)."""
        if ds.name not in self.featurizers:
            kwargs = {}
            if ds.modality == "image":
                kwargs["pretrained"] = self.image_pretrained
            f = featurizer_for(ds.modality, self.latent_dim, seed=self.seed, **kwargs)
            f.fit(ds.X_train)
            self.featurizers[ds.name] = f
        return self.featurizers[ds.name]

    def latents(self, ds: AnomalyDataset, split: str = "train") -> torch.Tensor:
        f = self._featurizer(ds)
        return f.transform(ds.X_train if split == "train" else ds.X_test)

    # ── one-time training on the source datasets ─────────────────────────

    def fit(
        self,
        datasets: Sequence[AnomalyDataset],
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: Optional[int] = 256,
        verbose: bool = False,
    ) -> "MultimodalPipeline":
        """Train the shared PC once, on the pooled latents of all sources
        (unsupervised exact-NLL — the primary regime)."""
        Z = torch.cat([self.latents(ds, "train") for ds in datasets], dim=0)
        if self.pc is None:
            self.pc = self._make_pc(
                learned_vtree(Z, method=self.vtree_method, seed=self.seed))
        self.pc.fit_leaves(Z)
        self.train_history = fit_pc_nll(self.pc, Z, epochs=epochs, lr=lr,
                                        batch_size=batch_size, verbose=verbose)
        self.trained_on = [ds.name for ds in datasets]
        return self

    # ── integrating / evaluating a new dataset ──────────────────────────

    def adapt(
        self,
        ds: AnomalyDataset,
        mode: str = "zero_shot",
        finetune_epochs: int = 10,
        lr: float = 0.02,
    ):
        """
        Return the PC used to score `ds` under the given adaptation mode.
        The shared PC is NEVER mutated: 'leaves' and 'finetune' act on a copy.
        """
        if mode not in ADAPTATION_MODES:
            raise KeyError(f"Unknown adaptation mode {mode!r}; use {ADAPTATION_MODES}")
        if self.pc is None:
            raise RuntimeError("Pipeline not fitted yet — the learned (Chow-Liu) "
                               "vtree is built from the source latents in fit()")
        if mode == "zero_shot":
            return self.pc
        pc = copy.deepcopy(self.pc)
        Z = self.latents(ds, "train")
        pc.fit_leaves(Z)  # closed-form median/MAD — no gradient steps
        if mode == "finetune":
            fit_pc_nll(pc, Z, epochs=finetune_epochs, lr=lr)
        return pc

    def evaluate(
        self,
        ds: AnomalyDataset,
        mode: str = "zero_shot",
        finetune_epochs: int = 10,
    ) -> dict:
        """AUROC of the (possibly adapted) shared model on a dataset."""
        pc = self.adapt(ds, mode, finetune_epochs=finetune_epochs)
        with torch.no_grad():
            scores = -pc.log_prob(self.latents(ds, "test"))
        normal = scores[ds.y_test == 0]
        anomal = scores[ds.y_test == 1]
        return {
            "dataset": ds.name,
            "modality": ds.modality,
            "adaptation": mode,
            "auroc": auroc(normal, anomal),
            # held-out NLL on the normal test points (nats): the
            # structure-sensitive metric for vtree comparisons — AUROC alone
            # can hide structure differences that mixtures compensate away
            "nll": float(normal.mean()),
            "n_test": int(ds.y_test.numel()),
        }

    def validate(self) -> None:
        self.pc.validate()


# ─── Direction 2: encoder-free routed PC ─────────────────────────────────────


class EncoderFreePipeline:
    """
    Direction 2 (the research bet): exact density estimation directly over the
    RAW features — no encoder anywhere.  One sub-circuit per modality (dataset),
    routed by modality; structure (the vtree) is SHARED across modalities that
    agree on their feature dimension, so the scope-partition hierarchy transfers
    while leaf/weight parameters stay modality-specific.

    Mirrors MultimodalPipeline's interface (fit / evaluate / validate) so the two
    directions run behind one config + cluster harness and are compared directly.

    Restrictions, baked in by the encoder-free contract:
      * Inputs must be numeric tensors.  Tabular (adbench) features qualify
        directly; text (strings) and full-resolution images do not — those are
        the encoder direction's domain.  A non-tensor / >2-D modality is flagged
        and skipped rather than silently featurized.
      * zero_shot scoring of a held-out dataset needs an existing sub-circuit of
        the SAME dimension (the exact mixture over dim-compatible experts).  A
        held-out dataset of a NEW dimension cannot be zero-shot scored without an
        encoder, so that (dataset, zero_shot) cell is skipped; leaves / finetune
        integrate it via add_modality (structure transfer) instead.
      * router=True (ProbRoutedRawPC) additionally emits the exact ProbMoE
        routing queries as anomaly signals (adaptation tag ``route:<signal>``),
        but only where ≥2 dim-compatible experts exist (single-expert routing is
        degenerate).  The density score is UNCHANGED — routing rides alongside it.
    """

    def __init__(
        self,
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
        use_sos: bool = False,
        vtree_method: str = "chow_liu",
        seed: int = 0,
        router: bool = False,
        k_min: int = 1,
        k_max: Optional[int] = None,
        routing_signals: Sequence[str] = ROUTING_SIGNALS,
    ):
        self.n_sum_components = n_sum_components
        self.leaf_factory = leaf_factory
        self.use_sos = use_sos
        self.vtree_method = vtree_method
        self.seed = seed
        self.router = router
        self.k_min = k_min
        self.k_max = k_max
        self.routing_signals = list(routing_signals)
        self.det: Optional[RoutedRawPC] = None
        self.trained_on: List[str] = []

    # ── raw-feature extraction (no encoder) ──────────────────────────────────

    @staticmethod
    def _raw(ds: AnomalyDataset, split: str = "train") -> torch.Tensor:
        X = ds.X_train if split == "train" else ds.X_test
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                f"encoder-free needs numeric features, but {ds.name!r} "
                f"(modality={ds.modality}) provides {type(X).__name__}; use the "
                f"encoder direction for this modality")
        X = X.float()
        return X.reshape(X.shape[0], -1) if X.dim() > 2 else X

    # ── one-time training on the source datasets ─────────────────────────────

    def fit(
        self,
        datasets: Sequence[AnomalyDataset],
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: Optional[int] = 256,
        verbose: bool = False,
    ) -> "EncoderFreePipeline":
        data = {ds.name: self._raw(ds, "train") for ds in datasets}
        feature_dims = {n: X.shape[1] for n, X in data.items()}
        kwargs = dict(
            n_sum_components=self.n_sum_components,
            leaf_factory=self.leaf_factory,
            use_sos=self.use_sos,
            shared_structure=True,
            vtree_method=self.vtree_method,
            seed=self.seed,
        )
        if self.router:
            self.det = ProbRoutedRawPC(feature_dims, k_min=self.k_min,
                                       k_max=self.k_max, **kwargs)
        else:
            self.det = RoutedRawPC(feature_dims, **kwargs)
        self.det.fit(data, epochs=epochs, lr=lr, batch_size=batch_size,
                     verbose=verbose)
        self.trained_on = list(data)
        return self

    # ── integrating / evaluating a dataset ───────────────────────────────────

    def _adapt(self, ds: AnomalyDataset, mode: str, finetune_epochs: int):
        """Return (detector, score_modality_or_None) for scoring `ds`.

        zero_shot reuses the shared detector and scores by the exact mixture
        over dim-compatible experts (score_modality=None).  leaves / finetune
        integrate `ds` as a NEW modality on a COPY via add_modality — leaves =
        closed-form leaf fit only (0 epochs), finetune = a few NLL epochs — so
        the shared model is never mutated and structure transfers when the
        dimension matches an existing sub-circuit.
        """
        if mode not in ADAPTATION_MODES:
            raise KeyError(f"Unknown adaptation mode {mode!r}; use {ADAPTATION_MODES}")
        if mode == "zero_shot":
            return self.det, None
        det = copy.deepcopy(self.det)
        name = ds.name if ds.name not in det.feature_dims else f"{ds.name}__{mode}"
        det.add_modality(name, self._raw(ds, "train"),
                         epochs=0 if mode == "leaves" else finetune_epochs)
        return det, name

    def evaluate_all(
        self,
        ds: AnomalyDataset,
        mode: str = "zero_shot",
        finetune_epochs: int = 10,
        role: str = "held_out",
    ) -> List[dict]:
        """Density row (always, when the dataset is scorable) plus the exact
        routing-query rows (router=True, ≥2 dim-compatible experts).  Returns an
        empty list when an unseen dimension makes the (dataset, mode) cell
        unscorable — the caller logs the skip."""
        det, score_mod = self._adapt(ds, mode, finetune_epochs)
        Xte = self._raw(ds, "test")
        try:
            with torch.no_grad():
                scores = -det.log_prob(Xte, modality=score_mod)
        except ValueError:
            # zero_shot on a never-seen dimension: no compatible sub-circuit
            return []
        normal = scores[ds.y_test == 0]
        anomal = scores[ds.y_test == 1]
        base = {"dataset": ds.name, "modality": ds.modality, "role": role}
        rows = [{
            **base,
            "adaptation": mode,
            "auroc": auroc(normal, anomal),
            "nll": float(normal.mean()),
            "n_test": int(ds.y_test.numel()),
        }]
        if self.router and isinstance(det, ProbRoutedRawPC):
            rows += self._routing_rows(det, Xte, ds, mode, base)
        return rows

    def _routing_rows(self, det, Xte, ds, mode, base) -> List[dict]:
        """Exact ProbMoE routing signals as anomaly scores (the F1 detection
        kill-shot material), each scored exactly like the density."""
        if len(det._compatible(Xte.shape[1])) < 2:
            return []  # single-expert routing is degenerate
        rows = []
        with torch.no_grad():
            for sig in self.routing_signals:
                rs = det.routing_score(Xte, signal=sig)
                rows.append({
                    **base,
                    "adaptation": f"{mode}+route:{sig}",
                    "auroc": auroc(rs[ds.y_test == 0], rs[ds.y_test == 1]),
                    "nll": float("nan"),  # routing signal, not a density
                    "n_test": int(ds.y_test.numel()),
                })
        return rows

    def validate(self) -> None:
        self.det.validate()


# ─── Seeded single run ───────────────────────────────────────────────────────

def run_experiment(
    train_sets: Sequence[AnomalyDataset],
    test_sets: Sequence[AnomalyDataset],
    seed: int = 0,
    adapt_modes: Sequence[str] = ("zero_shot", "leaves"),
    latent_dim: int = 64,
    n_sum_components: int = 3,
    epochs: int = 100,
    finetune_epochs: int = 10,
    use_sos: bool = False,
    vtree_method: str = "chow_liu",
    image_pretrained: bool = True,
    log_root: Optional[str] = None,
    leaf_factory=None,
    baselines: Sequence[str] = (),
    direction: str = "encoder",
    router: bool = False,
    k_min: int = 1,
    k_max: Optional[int] = None,
    routing_signals: Sequence[str] = ROUTING_SIGNALS,
) -> List[dict]:
    """
    One fully seeded, fully logged execution: pin every RNG, train the shared
    PC once on the sources, integrate each held-out dataset under every
    adaptation mode.  All artifacts land in logs/<seed>/ (run.log,
    config.json, history_train_nll.csv, results.jsonl).  Returns the result
    dicts (each tagged with the seed).
    """
    set_global_seed(seed)
    elog = ExperimentLogger(seed, root=log_root)
    elog.log_config({
        "train": [ds.name for ds in train_sets],
        "test": [ds.name for ds in test_sets],
        "adapt_modes": list(adapt_modes),
        "latent_dim": latent_dim,
        "n_sum_components": n_sum_components,
        "epochs": epochs,
        "finetune_epochs": finetune_epochs,
        "use_sos": use_sos,
        "vtree_method": vtree_method,
        "direction": direction,
        "router": router,
        "k_min": k_min,
        "k_max": k_max,
    })
    try:
        tag = {"vtree": vtree_method, "seed": seed, "direction": direction}
        if direction == "encoder_free":
            pipe = EncoderFreePipeline(
                n_sum_components=n_sum_components,
                leaf_factory=leaf_factory,
                use_sos=use_sos,
                vtree_method=vtree_method,
                seed=seed,
                router=router,
                k_min=k_min,
                k_max=k_max,
                routing_signals=routing_signals,
            )
            elog.info(f"training encoder-free routed PC on {len(train_sets)} "
                      f"source modality/modalities (router={router})")
            pipe.fit(train_sets, epochs=epochs)
            pipe.validate()

            results = []
            for ds in train_sets:  # in-distribution reference
                for r in pipe.evaluate_all(ds, "zero_shot", role="source"):
                    r.update(tag)
                    elog.log_result(r)
                    results.append(r)
            for ds in test_sets:
                for mode in adapt_modes:
                    rows = pipe.evaluate_all(ds, mode,
                                             finetune_epochs=finetune_epochs)
                    if not rows:
                        elog.info(f"{ds.name}: skipping ({mode}) — no "
                                  f"dim-compatible sub-circuit for encoder-free "
                                  f"scoring (dim {pipe._raw(ds, 'test').shape[1]})")
                        continue
                    for r in rows:
                        r.update(tag)
                        elog.log_result(r)
                        results.append(r)
            return results

        pipe = MultimodalPipeline(
            latent_dim=latent_dim,
            n_sum_components=n_sum_components,
            leaf_factory=leaf_factory,
            use_sos=use_sos,
            vtree_method=vtree_method,
            seed=seed,
            image_pretrained=image_pretrained,
        )
        elog.info(f"training shared PC once on {len(train_sets)} source dataset(s)")
        pipe.fit(train_sets, epochs=epochs)
        pipe.validate()
        elog.log_history("train_nll", pipe.train_history)

        results = []
        for ds in train_sets:  # in-distribution reference
            r = {**pipe.evaluate(ds, "zero_shot"), "role": "source",
                 "vtree": vtree_method, "seed": seed}
            elog.log_result(r)
            results.append(r)
        for ds in test_sets:
            for mode in adapt_modes:
                r = {**pipe.evaluate(ds, mode, finetune_epochs=finetune_epochs),
                     "role": "held_out", "vtree": vtree_method, "seed": seed}
                elog.log_result(r)
                results.append(r)

        if baselines:
            from .baselines import baseline_input_type, evaluate_baselines

            for ds in test_sets:
                # image-only baselines (WinCLIP etc.) run only on image
                # datasets; feature baselines use the SAME featurizer as the
                # PC → identical latent space, apples-to-apples comparison
                names = [n for n in baselines
                         if baseline_input_type(n) == "features"
                         or ds.modality == "image"]
                skipped = sorted(set(baselines) - set(names))
                if skipped:
                    elog.info(f"{ds.name}: skipping image-only baseline(s) "
                              f"{skipped} (modality={ds.modality})")
                if not names:
                    continue
                rows = evaluate_baselines(ds, names=names,
                                          featurizer=pipe._featurizer(ds),
                                          seed=seed)
                for r in rows:
                    elog.log_result(r)
                    results.append(r)
        return results
    finally:
        elog.close()


# ─── Config-file-driven execution ────────────────────────────────────────────
#
# Experiments are described by YAML/JSON files under config/, not by CLI
# flags.  Every key has a default (DEFAULT_CONFIG); unknown keys fail loudly.

DEFAULT_CONFIG: dict = {
    "name": "experiment",          # used for logs/summary_<name>.json
    "train": [],                   # dataset specs: "adbench:cardio", or
    "test": [],                    #   {spec: "mvtec:bottle", image_size: 128, ...}
    "adapt": ["zero_shot", "leaves"],
    "seeds": [0],                  # one full execution per seed
    "latent_dim": 64,
    "n_sum_components": 3,
    "epochs": 100,
    "finetune_epochs": 10,
    "use_sos": False,
    "vtree_method": "chow_liu",    # chow_liu | spectral | orc | forman | random
    "vtree_methods": None,         # list → matched-budget vtree ablation: the
                                   #   full pipeline runs once per method (same
                                   #   seeds, same capacity), rows tagged with
                                   #   "vtree" for side-by-side aggregation
    "image_pretrained": True,
    "log_root": None,              # default: ./logs
    "baselines": [],               # literature baselines (src/baselines.py),
                                   #   e.g. [iforest, lof, ocsvm, knn, gmm]
    # ── direction (which of the two project directions to run) ──────────────
    "direction": "encoder",        # encoder (latent PC) | encoder_free (raw PC)
    "router": False,               # encoder_free: ProbRoutedRawPC (ProbMoE
                                   #   routing queries) vs plain RoutedRawPC
    "k_min": 1,                    # router cardinality range (Dynamic-k); FREEZE
    "k_max": None,                 #   on a held-out modality — sweeping = p-hack
    "routing_signals": list(ROUTING_SIGNALS),
}


def load_config(path: str) -> dict:
    """Read a YAML or JSON experiment config and merge it over the defaults.
    Unknown keys raise, so typos cannot silently fall back to defaults."""
    import json

    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            user_cfg = yaml.safe_load(f) or {}
        elif path.endswith(".json"):
            user_cfg = json.load(f)
        else:
            raise ValueError(f"Config must be .yaml/.yml/.json, got {path!r}")

    unknown = set(user_cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"Unknown config key(s) {sorted(unknown)} in {path}; "
                       f"valid keys: {sorted(DEFAULT_CONFIG)}")
    cfg = {**DEFAULT_CONFIG, **user_cfg}
    if not cfg["train"] or not cfg["test"]:
        raise ValueError(f"Config {path} must define non-empty 'train' and 'test' lists")
    bad_modes = set(cfg["adapt"]) - set(ADAPTATION_MODES)
    if bad_modes:
        raise ValueError(f"Unknown adapt mode(s) {sorted(bad_modes)}; use {ADAPTATION_MODES}")
    methods = list(cfg["vtree_methods"] or []) + [cfg["vtree_method"]]
    bad_vtrees = set(methods) - set(VTREE_METHODS)
    if bad_vtrees:
        raise ValueError(f"Unknown vtree method(s) {sorted(bad_vtrees)}; "
                         f"use {VTREE_METHODS}")
    if cfg["direction"] not in DIRECTIONS:
        raise ValueError(f"Unknown direction {cfg['direction']!r}; use {DIRECTIONS}")
    if cfg["router"] and cfg["direction"] != "encoder_free":
        raise ValueError("router=True is only valid with direction=encoder_free")
    bad_sig = set(cfg["routing_signals"]) - set(ROUTING_SIGNALS)
    if bad_sig:
        raise ValueError(f"Unknown routing signal(s) {sorted(bad_sig)}; "
                         f"use {ROUTING_SIGNALS}")
    return cfg


def _load_spec_entry(entry, seed: int) -> AnomalyDataset:
    """A spec entry is either a string ('adbench:cardio') or a mapping with a
    'spec' key whose remaining keys are loader kwargs."""
    if isinstance(entry, str):
        return load_dataset_spec(entry, seed=seed)
    entry = dict(entry)
    spec = entry.pop("spec")
    entry.setdefault("seed", seed)
    return load_dataset_spec(spec, **entry)


def run_from_config(config, seed_override: Optional[int] = None) -> List[dict]:
    """
    Execute an experiment described by a config dict or a path to a config
    file: one fully seeded, fully logged run per seed, then the cross-seed
    mean ± std summary (logs/summary_<name>.json).

    seed_override runs ONE seed instead of the config's list — the hook for
    SLURM array jobs, where each array task executes a single seed and the
    cross-seed summary is produced afterwards by aggregate_from_logs.
    """
    if isinstance(config, str):
        config = load_config(config)
    else:
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise KeyError(f"Unknown config key(s): {sorted(unknown)}")
        config = {**DEFAULT_CONFIG, **config}
    seeds = [seed_override] if seed_override is not None else config["seeds"]
    methods = list(config["vtree_methods"] or [config["vtree_method"]])

    all_results: List[dict] = []
    for seed in seeds:
        # reload per seed: the train/test split of every dataset is itself
        # seeded, so robustness covers data splits, init, and training noise
        train_sets = [_load_spec_entry(s, seed) for s in config["train"]]
        test_sets = [_load_spec_entry(s, seed) for s in config["test"]]
        for i, method in enumerate(methods):
            all_results += run_experiment(
                train_sets, test_sets,
                seed=seed,
                adapt_modes=config["adapt"],
                latent_dim=config["latent_dim"],
                n_sum_components=config["n_sum_components"],
                epochs=config["epochs"],
                finetune_epochs=config["finetune_epochs"],
                use_sos=config["use_sos"],
                vtree_method=method,
                image_pretrained=config["image_pretrained"],
                log_root=config["log_root"],
                # baselines are vtree-independent: run them once per seed
                baselines=config["baselines"] if i == 0 else (),
                direction=config["direction"],
                router=config["router"],
                k_min=config["k_min"],
                k_max=config["k_max"],
                routing_signals=config["routing_signals"],
            )

    aggregated = aggregate_results(all_results)
    suffix = f"_seed{seed_override}" if seed_override is not None else ""
    path = save_summary(aggregated, root=config["log_root"],
                        filename=f"summary_{config['name']}{suffix}.json")
    print(f"\n=== {config['name']}: robustness over seeds {seeds} "
          f"(summary saved to {path}) ===")
    print(format_summary_table(aggregated))
    return all_results


def aggregate_from_logs(config) -> List[dict]:
    """
    Cross-seed aggregation WITHOUT re-running anything: read the per-seed
    logs/<seed>/results.jsonl files written by earlier executions (e.g. SLURM
    array tasks, one seed each), keep the latest entry per
    (dataset, adaptation, role, seed) restricted to this config's datasets,
    and write the usual summary_<name>.json.
    """
    if isinstance(config, str):
        config = load_config(config)
    else:
        config = {**DEFAULT_CONFIG, **config}

    def spec_name(entry) -> str:
        return entry if isinstance(entry, str) else entry["spec"]

    wanted = {spec_name(e) for e in list(config["train"]) + list(config["test"])}

    latest: Dict[tuple, dict] = {}
    for seed in config["seeds"]:
        elog = ExperimentLogger(seed, root=config["log_root"], echo=False)
        for r in elog.read_results():           # file order = chronological
            if r.get("dataset") in wanted:
                latest[(r.get("dataset"), r.get("adaptation"),
                        r.get("role"), r.get("vtree"), r.get("seed"))] = r
        elog.close()
    results = list(latest.values())
    if not results:
        raise RuntimeError(
            f"No logged results found for config {config['name']!r} under "
            f"{config['log_root'] or 'logs/'} (seeds {config['seeds']}) — "
            "run the experiments first")

    aggregated = aggregate_results(results)
    path = save_summary(aggregated, root=config["log_root"],
                        filename=f"summary_{config['name']}.json")
    print(f"\n=== {config['name']}: aggregated from logs, seeds "
          f"{sorted({r['seed'] for r in results})} (saved to {path}) ===")
    print(format_summary_table(aggregated))
    return results


def main(argv: Optional[List[str]] = None) -> List[dict]:
    p = argparse.ArgumentParser(
        description="Run experiments described by config files under config/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("configs", nargs="+",
                   help="path(s) to YAML/JSON experiment config(s), e.g. "
                        "config/adbench_demo.yaml")
    p.add_argument("--seed", type=int, default=None,
                   help="run only this seed (SLURM array hook: one seed per "
                        "array task); the config's seed list is ignored")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip execution; aggregate existing logs/<seed>/ "
                        "results into the cross-seed summary")
    args = p.parse_args(argv)

    results: List[dict] = []
    for path in args.configs:
        print(f"\n##### {path} #####")
        if args.aggregate_only:
            results += aggregate_from_logs(path)
        else:
            results += run_from_config(path, seed_override=args.seed)
    return results


if __name__ == "__main__":
    main()
