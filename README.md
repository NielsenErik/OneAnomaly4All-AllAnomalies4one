# One Anomaly For All — All Anomaly For One

A **multimodal probabilistic circuit (PC)** framework for anomaly detection
across data types (images, text/hallucinations, tabular, …) with **exact**
density estimation and **exact** marginals as the non-negotiable invariant.

Two directions are implemented behind one shared interface so they can be
compared directly:

1. **Direction 1 — encoder + latent PC** (`LatentPCDetector`): per-modality
   encoders map raw inputs into a shared latent space; an exact PC scores the
   latents. Lower risk; the encoder is the only approximation.
2. **Direction 2 — encoder-free PC over raw input** (`RoutedRawPC`): one exact
   sub-circuit per modality directly over raw features, with a **shared vtree**
   across modalities (structure-level transfer — the research bet).

## Layout

```
src/
  datasets.py                 # ADBench / MVTec-AD / text-AD loaders + featurizers
  experiment.py               # multimodal pipeline: train once, integrate new
                              # datasets with zero/cheap adaptation + CLI
  probabilistic_circuits.py   # File 1: all PC components
                              #  - vtree (save/load, random generators, LCA matrix)
                              #  - structure sources (random / single-task / consensus)
                              #  - leaves: Gaussian, GaussianMixture, Categorical, InputNode
                              #  - SumNode / ProductNode, DensityPC
                              #  - exact inference: log-density, marginals, MPE, partition
                              #  - the 4 property validators
                              #  - SOS mode: SquaredPC (subtractive mixtures, exact Z)
  directions.py               # File 2: the two directions + shared scoring interface
                              #  - training loops: unsupervised NLL (primary) and
                              #    supervised contrastive (arXiv:2605.05953, Eq. 4)
tests/
  test_vtree.py               # vtree, structure sources, DensityPC basics
  test_inference.py           # marginals/MPE/partition vs. brute-force grids, SOS, validators
  test_directions.py          # both directions, both training regimes, AUROC
```

## Environment

Everything runs in the `expllm_env` conda environment (torch, numpy, scipy, pytest):

```bash
conda activate expllm_env
cd /path/to/OneAnomalyForAll-AllAnomalyforOne
```

All snippets below assume you run Python from the repo root (imports are `src.*`).

## Running the tests

```bash
# Full suite (~20 min: the exactness tests integrate densities on brute-force grids)
python -m pytest tests/ -q

# Fast development loop (skip the grid-integration file)
python -m pytest tests/test_vtree.py tests/test_directions.py -q

# Only the exact-inference / SOS checks
python -m pytest tests/test_inference.py -q
```

Every constructed circuit can (and should) be checked against the four
structural properties — tests fail loudly on violations.

## File 1: building and querying circuits

### Build a PC from a vtree, validate, score

```python
import torch
from src.probabilistic_circuits import (
    DensityPC, GaussianLeaf, random_balanced_vtree,
)

d = 8
vt = random_balanced_vtree(list(range(d)), seed=0)
pc = DensityPC(vt, n_sum_components=3, leaf_factory=GaussianLeaf)

X = torch.randn(500, d)
pc.fit_leaves(X)          # robust median/MAD init of all leaves
pc.validate()             # smoothness + decomposability + structured decomposability

log_p  = pc.log_prob(X)            # exact log-density, shape (B,)
scores = pc.anomaly_score(X)       # NLL: higher = more anomalous
```

Leaf choices: `GaussianLeaf`, `GaussianMixtureLeaf(idx, n_components)`,
`CategoricalLeaf(idx, n_categories)`, or the default heavy-tailed `InputNode`
(Gaussian/Laplace/Student-t mixture).

### Exact inference

```python
# Marginals: integrate out features {2, 5} exactly (their values are ignored)
log_m = pc.log_marginal(X, marginalized=[2, 5])

# Partition function (≈ 0 by construction for the monotone path)
log_Z = pc.log_partition()

# MPE: most probable completion given partial evidence
assignment, log_val = pc.mpe(evidence={0: 1.5, 3: -0.2})
```

### The four property validators

```python
from src.probabilistic_circuits import (
    validate_smoothness, validate_decomposability,
    validate_determinism, validate_structured_decomposability,
)

validate_smoothness(pc.root)
validate_decomposability(pc.root)
validate_structured_decomposability(pc.root, vt)
validate_determinism(pc.root, X)   # empirical; mixtures of Gaussians correctly fail
```

### SOS mode (sum-of-squares / subtractive mixtures)

```python
from src.probabilistic_circuits import SquaredPC

sos = SquaredPC(vt, n_sum_components=3, leaf_factory=GaussianLeaf, seed=0)
sos.fit_leaves(X)
log_p = sos.log_prob(X)                  # 2·log|c(x)| − log Z, exactly normalized
log_m = sos.log_marginal(X, [1, 4])      # exact marginals via the pairwise recursion
sos.validate()
```

SOS leaves must support pairwise integrals (Gaussian, GMM, Categorical — not
`InputNode`). MPE is not supported in SOS mode.

### Learned (Chow-Liu / HCLT-style) vtree

Instead of a random structure, learn the vtree from data: pairwise mutual
information → Chow-Liu maximum spanning tree → recursive balanced weak-edge
cuts. Dependent features end up deep under the same sub-circuit, which raises
expressiveness at identical capacity while preserving all four circuit
properties:

```python
from src.probabilistic_circuits import chow_liu_vtree, mutual_information_matrix

vt = chow_liu_vtree(X_train)               # data-driven structure
pc = DensityPC(vt, n_sum_components=3, leaf_factory=GaussianLeaf)
```

### Curvature-guided and spectral vtrees

Beyond Chow-Liu, three more structure learners share the same dispatcher
(`learned_vtree(X, method=...)`); every method yields a valid vtree, so the
four circuit properties hold for all of them — only structure quality differs:

| method     | cut criterion | reference |
|------------|---------------|-----------|
| `chow_liu` | weakest MST edge (rank-1 pairwise surrogate) | Chow & Liu 1968; HCLT, Liu & Van den Broeck 2021 |
| `spectral` | recursive normalized cut of the MI matrix (the honest adversary) | Shi & Malik 2000 |
| `orc`      | Ollivier-Ricci bottlenecks: cut the most negatively curved edges of the sparsified MI graph (exact per-edge W₁) | Ollivier 2009; Ni et al. 2019; Topping et al. ICLR 2022 |
| `forman`   | Forman-Ricci variant — closed form, no transport, for high d | Sreejith et al. 2016 |
| `random`   | balanced random split (control) | — |

Rationale for the curvature methods: a product node's modeling error is the
mutual information across its scope cut, and negative curvature marks the
neighborhood-aware bottleneck — the cut crossing the least aggregate
dependence — where the weakest-MST-edge rule only sees one pairwise weight.

```python
from src.probabilistic_circuits import (
    learned_vtree, curvature_vtree, spectral_vtree,
    ollivier_ricci_curvature, forman_curvature,
    sparsify_mi_graph, curvature_sign_stability,
)

vt = learned_vtree(X_train, method="orc")          # or spectral / forman / ...
vt = curvature_vtree(X_train, flow_iters=2)        # Ricci-flow reweighting

# MI estimates are noisy at small n — check edge-sign stability before
# trusting a cut (fraction of bootstrap resamples agreeing per edge):
stab = curvature_sign_stability(X_train, n_boot=20)
```

Both directions and the pipeline accept any of these via `vtree_method=`
(`MultimodalPipeline` defaults to `"chow_liu"`; `"random"` is the control).
For every learned method the structure comes from the training data inside
`fit`, so the PC is built lazily:

```python
det  = RoutedRawPC({"a": d, "b": d}, vtree_method="orc")  # shared learned vtree
pipe = MultimodalPipeline(latent_dim=64)                  # chow_liu by default
```

**Matched-budget vtree ablation** (the decisive structure-vs-capacity
experiment): `config/vtree_ablation.yaml` runs the full pipeline once per
method — identical capacity, seeds 0-4 — and aggregates AUROC **and held-out
NLL** per (dataset, vtree) so the methods can be read side by side:

```bash
python -m src.experiment config/vtree_ablation.yaml          # local
bash cluster_scripts/submit_all.sh config/vtree_ablation.yaml  # SLURM
```

The bar that matters: `orc` must beat `spectral` (not just `chow_liu`) by
more than the cross-seed std — otherwise the geometry is decoration. And if
`random` matches the learned methods, structure is doing no work at this
capacity (mixtures compensate), which gates the structure-transfer claims.

### Vtrees and structure sources

```python
from src.probabilistic_circuits import (
    save_vtree, load_vtree, random_unbalanced_vtree,
    single_task_vtree, consensus_vtree,
)

save_vtree(vt, "vtree.json"); vt2 = load_vtree("vtree.json")

vt_single    = single_task_vtree(pc)                       # negative control
vt_consensus = consensus_vtree([vt, vt_single], n_features=d)  # co-grouping consensus
```

## File 2: the two directions

### Direction 1 — encoder + latent PC

```python
import torch
from src.directions import LatentPCDetector, default_mlp_encoder, evaluate_detector
from src.probabilistic_circuits import GaussianLeaf

X_train = torch.randn(500, 64)                     # normal data, raw dim 64

det = LatentPCDetector(
    default_mlp_encoder(in_dim=64, latent_dim=16),
    latent_dim=16, n_sum_components=3, leaf_factory=GaussianLeaf,
)
det.fit(X_train, epochs=100)                       # UNSUPERVISED (primary regime)
scores = det.score(X_test)                         # higher = more anomalous
```

Multimodal: pass dicts — one encoder per modality, one shared latent PC:

```python
det = LatentPCDetector(
    {"image": img_encoder, "text": txt_encoder}, latent_dim=16,
)
det.fit({"image": X_img, "text": X_txt})
scores = det.score(X_img_test, modality="image")
```

### Direction 2 — encoder-free routed PC (the research bet)

```python
from src.directions import RoutedRawPC

det = RoutedRawPC({"tabular": 8, "sensor": 8})     # equal dims → shared vtree object
det.fit({"tabular": X_tab, "sensor": X_sen}, epochs=100)
det.validate()

s1 = det.score(X_tab_test, modality="tabular")     # routed: exact density of raw data
s2 = det.score(X_unknown)                          # unknown modality: exact mixture
```

Learned shared structure (single-task vtrees → consensus → refit):

```python
from src.directions import build_consensus_routed_pc

det = build_consensus_routed_pc({"a": X_a, "b": X_b}, pretrain_epochs=30, epochs=100)
```

## Training regimes (both directions)

Per arXiv:2605.05953 (Eq. 4):
`L = α·E[−log p(z⁺)] + (1−α)·E[max(0, γ + log p(z⁻) − log p(z⁺))]`

**Unsupervised (α = 1) — the primary target.** Pure exact-NLL maximum
likelihood on normal data only; no labels needed:

```python
det.fit(X_train)                  # alias: det.fit_unsupervised(...)
```

**Supervised contrastive** — when labeled anomalies exist, the margin term
displaces them into low-density regions (encoder trained jointly in
Direction 1; gradients clipped at ‖∇‖₂ ≤ 1):

```python
det.fit_contrastive(X_normal, X_anomalous, alpha=0.5, margin=1.0, epochs=100)

# Direction 2: per-modality; modalities without negatives fall back to NLL
det.fit_contrastive({"a": Xa_pos, "b": Xb_pos}, {"a": Xa_neg})
```

The loops are also available standalone for any PC:
`fit_pc_unsupervised(pc, X)` and `fit_pc_contrastive(pc, X_pos, X_neg, ...)`.

## Real datasets: ADBench, MVTec-AD, text AD

All loaders fetch **lazily** — single files via Python packages, never the
whole benchmark — and cache locally (`data/adbench/`, the HF hub cache,
sklearn's cache):

```python
from src.datasets import load_adbench, load_mvtec, load_text_ad, list_adbench

ds = load_adbench("thyroid")          # one ~MB npz from the ADBench repo
print(list_adbench().keys())          # all available ADBench dataset names

ds = load_mvtec("bottle")             # ONLY this category from HF Voxel51/mvtec-ad
                                      # (manifest first, then just its images)
ds = load_text_ad("sci")              # one-class 20 Newsgroups via sklearn

# every loader returns the same protocol object:
# ds.X_train (normal-only), ds.X_test, ds.y_test (1 = anomaly), ds.modality
```

If MVTec-AD is already on disk in the standard layout, skip the network with
`load_mvtec_local(root, "bottle")`.

## Multimodal pipeline: train once, integrate new datasets cheaply

The shared PC is trained **once**; a new dataset then costs (in increasing
order — the first two involve **no gradient retraining of the PC at all**):

| mode        | what happens                                                         |
|-------------|----------------------------------------------------------------------|
| `zero_shot` | new dataset gets only a closed-form featurizer; PC untouched         |
| `leaves`    | leaf params of a **copy** re-initialized (closed-form median/MAD)    |
| `finetune`  | a few NLL epochs on a **copy** (structure + init inherited)          |

```python
from src.experiment import MultimodalPipeline
from src.datasets import load_adbench, load_text_ad

pipe = MultimodalPipeline(latent_dim=64, n_sum_components=3)
pipe.fit([load_adbench("cardio"), load_text_ad("sci")], epochs=100)   # ONCE

new = load_adbench("thyroid")                       # never seen in training
print(pipe.evaluate(new, "zero_shot"))              # no retraining at all
print(pipe.evaluate(new, "leaves"))                 # closed-form only
```

Or from the command line — experiments are described by **config files under
`config/`**, not CLI flags:

```bash
python -m src.experiment config/adbench_demo.yaml
python -m src.experiment config/multimodal_demo.yaml config/sos_ablation.yaml
```

A config sets dataset specs, adaptation modes, seeds, and model
hyperparameters; unknown keys fail loudly and anything omitted falls back to
a default (see `DEFAULT_CONFIG` in `src/experiment.py`):

```yaml
# config/adbench_demo.yaml
name: adbench_demo
train: [adbench:cardio]
test:  [adbench:thyroid]
adapt: [zero_shot, leaves, finetune]
seeds: [0, 1, 2]
latent_dim: 32
epochs: 40
vtree_method: chow_liu       # chow_liu | spectral | orc | forman | random
```

Setting `vtree_methods: [chow_liu, spectral, orc, forman, random]` instead
runs the whole pipeline once per method at identical capacity (the
matched-budget vtree ablation, `config/vtree_ablation.yaml`); rows are tagged
with the method and the summary table gains `vtree` and held-out `nll`
columns.

Dataset entries can carry loader options (mapping form):

```yaml
test:
  - spec: mvtec:bottle
    image_size: 224
    max_train: 100
```

Direction 2 equivalent — add a modality without touching existing circuits
(the new sub-circuit reuses the shared vtree: structure transfer):

```python
det = RoutedRawPC({"a": 8})
det.fit({"a": X_a})
det.add_modality("b", X_b)        # only the new sub-circuit is trained
```

## Logging, seeds, and robustness over multiple executions

Every execution is fully seeded (`set_global_seed` pins python/numpy/torch —
data splits, featurizer projections, leaf jitter, batch shuffling, init) and
logs all artifacts under `logs/<seed>/`:

```
logs/
  0/
    run.log                 # full text log
    config.json             # run configuration
    history_train_nll.csv   # training curve
    results.jsonl           # one line per (dataset, adaptation) result
  1/ …
  summary.json              # cross-seed mean ± std aggregate
```

Run the same experiment over several seeds and get the robustness table
(per-seed dataset splits are re-drawn too, so the std covers split noise,
initialization, and training stochasticity):

```bash
# seeds come from the config file (seeds: [0, 1, 2, 3, 4])
python -m src.experiment config/adbench_demo.yaml
# === adbench_demo: robustness over seeds [0, 1, 2] (summary saved to logs/summary_adbench_demo.json) ===
# dataset              adaptation    auroc (mean±std)  seeds
# adbench:thyroid      zero_shot     0.861 ±0.012      3
# ...
```

Programmatic use:

```python
from src.experiment import run_experiment
from src.logging_utils import aggregate_results, format_summary_table

results = []
for seed in range(5):
    results += run_experiment(train_sets, test_sets, seed=seed)
print(format_summary_table(aggregate_results(results)))
```

Same seed → bit-identical results; `logs/<seed>/results.jsonl` accumulates
across runs so repeated executions remain auditable.

## Comparing the two directions

```python
from src.directions import evaluate_detector

auc_1 = evaluate_detector(latent_det, X_in_test, X_out_test)
auc_2 = evaluate_detector(routed_det, X_in_test, X_out_test, modality="tabular")
```

`evaluate_detector` returns rank-based AUROC (1.0 = perfect, 0.5 = chance);
both directions share the `AnomalyDetector` interface (`fit`, `log_prob`,
`score`), so swapping them in a benchmark is a one-line change.

## Running on a SLURM cluster

`cluster_scripts/` contains ready-to-submit SLURM scripts (conda env
`expllm_env`, `.env` sourcing for HF_TOKEN, CUDA module; the PC experiments
are CPU-bound — uncomment `--gres=gpu:1` only for configs with image
datasets):

```bash
# everything: per config, one array job (one task per seed) + a chained
# aggregation job (--dependency=afterok) that writes the mean±std summary
bash cluster_scripts/submit_all.sh

# a single config, all seeds sequentially in one job
sbatch cluster_scripts/run_config.sbatch config/adbench_demo.yaml

# seed-parallel by hand: array indices = seeds, then aggregate
sbatch --array=0-2 cluster_scripts/run_config_array.sbatch config/adbench_demo.yaml
sbatch cluster_scripts/aggregate.sbatch config/adbench_demo.yaml
```

The hooks behind the array workflow: `python -m src.experiment <config>
--seed $SLURM_ARRAY_TASK_ID` runs one seed (writing `logs/<seed>/` and a
non-clobbering `summary_<name>_seed<seed>.json`), and `--aggregate-only`
re-reads all `logs/<seed>/results.jsonl` files and produces the final
cross-seed `logs/summary_<name>.json` without re-running anything. SLURM
stdout goes to `logs/slurm_*.txt`.

## Literature baselines

`src/baselines.py` provides 10 standard anomaly-detection baselines behind
the same fit-on-normals / score interface, so they drop into the same
evaluation and logging machinery as the PCs. Add them to any config
(`baselines: [iforest, lof, ocsvm, knn, gmm]`) and they are scored **in the
same latent space** as the shared PC (same featurizer) — an apples-to-apples
comparison in the summary table (role `baseline`).

| name          | method                          | reference |
|---------------|---------------------------------|-----------|
| `iforest`     | Isolation Forest                | Liu, Ting, Zhou — ICDM 2008 |
| `lof`         | Local Outlier Factor            | Breunig, Kriegel, Ng, Sander — SIGMOD 2000 |
| `ocsvm`       | One-Class SVM                   | Schölkopf et al. — Neural Computation 2001 |
| `knn`         | k-NN distance                   | Ramaswamy, Rastogi, Shim — SIGMOD 2000 |
| `gmm`         | Gaussian Mixture NLL (EM)       | Dempster, Laird, Rubin — JRSS-B 1977; ADBench (Han et al., NeurIPS 2022) |
| `mahalanobis` | Mahalanobis distance            | Mahalanobis 1936; Lee et al. — NeurIPS 2018 |
| `pca`         | Principal Component Classifier  | Shyu, Chen, Sarinnapakorn, Chang — ICDM-FDM 2003 |
| `ecod`        | Empirical-CDF tail probabilities| Li et al. — IEEE TKDE 2022 (arXiv:2201.00382) |
| `ae`          | Autoencoder reconstruction      | Hawkins et al. — DaWaK 2002; Sakurada & Yairi — MLSDA 2014 |
| `deep_svdd`   | Deep SVDD (one-class deep)      | Ruff et al. — ICML 2018 |
| `winclip`     | Zero-/few-shot CLIP prompt ensemble (image-only, raw images) | Jeong et al. — CVPR 2023 (arXiv:2303.14814) |
| `anomalyclip` | Object-agnostic CLIP prompts — *lite*: fixed prompts, no prompt training (lower bound of the full method) | Zhou et al. — ICLR 2024 (arXiv:2310.18961) |
| `anomalygpt`  | LVLM industrial AD — integration adapter; needs the official checkpoint (not re-implemented) | Gu et al. — AAAI 2024 (arXiv:2308.15366) |

The three CLIP/LVLM baselines have `input_type="image"`: they consume **raw
images** (the evaluation glue routes them automatically and passes the
dataset's class name to WinCLIP's prompts), they download CLIP weights
lazily via `transformers` on first use, and they error with clear
instructions on non-image datasets. `anomalygpt` only runs if you provide
the official checkpoint (`baseline_kwargs={"anomalygpt": {"checkpoint_path": …}}`).

Programmatic use:

```python
from src.baselines import make_baseline, evaluate_baselines

bl = make_baseline("iforest", seed=0).fit(X_train_normal)
scores = bl.score(X_test)                       # higher = more anomalous
rows = evaluate_baselines(ds, names=["iforest", "gmm"], seed=0)
```

Full citations (DOIs/arXiv) are in the `src/baselines.py` module docstring.
Survey context: Han et al., *ADBench* (NeurIPS 2022); Ruff et al., *A
Unifying Review of Deep and Shallow Anomaly Detection* (Proc. IEEE 2021).

## References

- Choi, Vergari, Van den Broeck — *Probabilistic Circuits: A Unifying Framework* (properties)
- Loconte, Mengel, Vergari — *Sum of Squares Circuits*, AAAI 2025 (SOS mode)
- Nielsen, Cunegatti, Vukojevic, Iacca — *Hallucination as an Anomaly: Dynamic
  Intervention via Probabilistic Circuits*, arXiv:2605.05953 (training objective)
- Full reference list in `CLAUDE.md`.
