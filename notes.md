# Implementation Notes

---

## Stage 0a + 0b — Vtree Infrastructure & Structure Sources — 2026-05-25

### What was built

#### `src/vtree.py` — Core data structures

- **`VtreeLeaf(feature_idx)`** and **`VtreeInternal(left, right)`**: frozen dataclasses representing the variable scope-partition tree.  `VtreeInternal` exposes `.scope`, `.left_scope`, `.right_scope` as derived `frozenset[int]` properties.
- **`save_vtree` / `load_vtree`**: full JSON round-trip.  Sets are serialised as sorted lists to guarantee deterministic output across runs.
- **`vtree_depth`, `vtree_leaves`, `vtree_nodes`**: tree-inspection utilities.
- **`lca_depth_matrix(root)`**: returns a dict mapping each feature pair `(i, j)` → depth of their lowest-common ancestor.  This is the raw signal used by the consensus algorithm: deeper LCA = features were kept together longer across the partition hierarchy.
- **`random_balanced_vtree(features, seed)`**: shuffles features then splits at the midpoint recursively.  Produces a balanced binary tree.
- **`random_unbalanced_vtree(features, seed)`**: shuffles then splits at a random point at each level.  Can be highly skewed; used as a second random baseline.

#### `src/probabilistic_circuits.py` — Node classes + DensityPC

All original node classes are retained unchanged (JIT kernels, `InputNode`, `SumNode`, `ProductNode`, `ClassifierNode`).

New additions:

- **`DensityPC(vtree_root, n_sum_components)`**: a single structured-decomposable PC built top-down from a vtree.  At each `VtreeInternal` node, `n_sum_components` product-node sub-circuits are created (one per mixture component) and wrapped in a `SumNode`.  At each `VtreeLeaf`, an `InputNode` is created.  This construction guarantees smoothness and decomposability by construction.
  - `forward(z)` → `log p(z)` of shape `(B,)` for input `z ∈ ℝ^{B×d}`.
  - `fit_leaves(X)`: median/MAD initialisation of all `InputNode`s from data.
  - `anomaly_score(z)` → `-log p(z)`.
  - `validate()`: convenience wrapper around `validate_circuit`.

- **`validate_circuit(root)`**: traverses the circuit and asserts (i) decomposability — product-node children have pairwise-disjoint scopes — and (ii) smoothness — sum-node children all share the same scope.  Raises `AssertionError` with an actionable message on first violation.  Uses scope memoisation by node `id` to handle shared sub-trees efficiently.

- **`extract_vtree(root)`**: walks the circuit and reconstructs the implied `VtreeNode` tree.  `InputNode` → `VtreeLeaf`; `ProductNode` → `VtreeInternal`; `SumNode`/`ClassifierNode` are transparent (delegate to first child).  Raises `ValueError` if a `ProductNode` has ≠ 2 children.

- **`PCNet`** (legacy): the original random-structure builder from `bak/`, refactored to track `scope_map` and `vtree_map` dictionaries during `init_network`.  Now records `self.vtree` (best-effort extraction) alongside `self.root`.  Not guaranteed to be smooth/decomposable — use `DensityPC` for formally valid circuits.

#### `src/vtree_sources.py` — Three structure sources (Stage 0b)

- **Source 1 (random baseline)**: `random_balanced_vtree` and `random_unbalanced_vtree` re-exported.  No domain information; any positive transfer result must beat this.
- **Source 2 (single-task / negative control)**: `single_task_vtree(pc)` wraps `extract_vtree(pc.root)`.  Expected to overfit the source domain and fail to transfer.
- **Source 3 (consensus / candidate contribution)**: `consensus_vtree(vtrees, n_features)` implements the co-grouping matrix algorithm from dev.md §3.2:
  1. `cogroup_matrix` computes `M[i,j]` = average normalised LCA-depth of `(i,j)` across source vtrees.
  2. Distance = `1 − M`.
  3. Average-linkage hierarchical clustering (scipy) on the distance matrix.
  4. Dendrogram converted to a `VtreeInternal` / `VtreeLeaf` tree.
  - Degrades gracefully: returns the single source vtree if only one is provided; falls back to a random balanced vtree if the list is empty.
  - **Critical alignment note** (dev.md §3.2): consensus is only meaningful when source-domain latent dimensions are aligned (shared `fϕ` or explicit alignment step before `cogroup_matrix` is called).

#### `tests/test_vtree.py` — 32 tests, all passing

- `VtreeLeaf`/`VtreeInternal` scope properties
- JSON round-trip save/load, including check that no raw Python `set` objects appear in serialised output
- `random_balanced_vtree` covers all features, is reproducible per seed, and differs across seeds
- `DensityPC` builds successfully; `n_sum_components=1` produces no `SumNode`s
- `validate_circuit` passes on a valid `DensityPC`; raises `AssertionError` with "Decomposability" / "Smoothness" on hand-crafted violations
- `extract_vtree` scope round-trips through `DensityPC`
- `forward` output shape `(B,)` and all-finite values
- Anomaly score higher for N(10, 0.1) data than N(0,1) after `fit_leaves` — basic density property
- `lca_depth_matrix` gives deeper LCA for within-block pairs than cross-block pairs
- `cogroup_matrix` within-block average > cross-block average
- **Synthetic case 11 (dev.md)**: 10 source vtrees all encoding the same `{0,1,2} | {3,4,5}` block structure → consensus vtree covers all 6 features, passes `validate_circuit`, and co-grouping matrix confirms within-block > cross-block signal
- Consensus edge cases: single-vtree passthrough, empty-list fallback
- d=128 smoke test: builds, forward pass, validates — ready for the transfer study

### What is NOT done yet (explicit scope boundary)

- Projection `fϕ` (MLP bottleneck to fixed d=128): not yet implemented.
- Latent alignment decision (shared `fϕ` vs. explicit alignment): unresolved; **must be decided before running Stage 1 consensus experiments** (see dev.md §3.2 critical dependency).
- Training loop (contrastive NLL + margin objective).
- Stage 1 transplant experiment (refit weights/leaves on target domain, measure AUROC at n ∈ {10, 20, 50, 100}).
- DoSE-style statistic-of-density score (Stage 2).
- Threshold calibration and transfer.

### Key design decisions made

- `DensityPC` creates one fresh `InputNode` per product-component per leaf position (tree form, no DAG sharing).  This gives `n_sum_components^depth` leaf instances per feature — more parameters but simpler gradient flow and no shared-node aliasing issues.
- `validate_circuit` memoises by `id(node)`, so it handles circuits with internal DAG sharing (e.g., residual-path nodes from legacy `PCNet`) without double-traversal.
- `consensus_vtree` uses average-linkage (not single or complete) because it is less sensitive to outlier source vtrees, which is appropriate when source domains have heterogeneous quality.
- Attentional sum-node variants (`AttentionalSumNode`, `MultiHeadAttentionalSumNode`) from the original code are **excluded from `DensityPC`** because input-dependent weights break normalisation and therefore exact marginals — they become discriminative scorers, not valid density models.  Kept in `bak/` for potential ablation use.
