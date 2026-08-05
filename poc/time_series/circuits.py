"""
Circuit models for the time-series PoC, built on the region-graph DAG.

Two models, one density each:

  WindowPC   — exact density over a flattened (window × channel) block.  The
               anomaly score is −log p(x); everything else it can answer
               (per-channel conditionals, scoring under dead sensors) is a
               different query on the SAME trained object.

  SurvivalPC — exact joint density over (window, τ) where τ is the discretised
               time-to-failure.  Trained with the EXACT right-censored
               likelihood: an observed failure contributes log p(x, τ=k), a
               unit still alive at bin c contributes log P(x, τ ≥ c), which is
               an axis-aligned box query — not an approximation, not a
               separate hazard head.  At test time the same circuit gives
               p(τ|x), E[τ|x] and S(t|x) = P(τ > t | x).

Neither model is representable in the tree layout at these widths, which is
what `bench_scaling.py` demonstrates.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .progress import Phase, track
from src.probabilistic_circuits import (
    CategoricalLeaf,
    RegionNode,
    chain_region_graph,
    delta_window_transform,
    permute_region_graph,
    timestep_block_permutation,
    learned_region_graph,
    move_circuit_,
    GaussianLeaf,
    GaussianMixtureLeaf,
    RegionGraphPC,
    SquaredPC,
    VtreeInternal,
    VtreeLeaf,
    VtreeNode,
    learned_vtree,
    random_balanced_vtree,
    time_channel_vtree,
)

VTREE_CHOICES = (
    # binary vtrees
    "time", "channel", "channel_groups", "chow_liu", "spectral",
    "orc", "forman", "random",
    # region graphs: n-ary curvature / spectral, and the HMM-shaped chain
    "orc_rg", "forman_rg", "spectral_rg", "orc_rg_multi", "forman_rg_multi",
    "chain", "chain_grouped", "chain_full",
    # LAYOUT CONTROLS — the same chain circuit with the variable->position map
    # broken.  Capacity, arity, depth and parameter count are identical, so a
    # difference is attributable to the layout and nothing else (§B.3).
    "chain_perm_blocks", "chain_perm_features",
)


# ═══════════════════════════════════════════════════════════════════════════
# Device handling
# ═══════════════════════════════════════════════════════════════════════════

def resolve_device(device: Optional[str] = None) -> torch.device:
    """
    'auto' -> cuda when present, else mps, else cpu.

    What the flag is worth depends entirely on the EVALUATOR (see
    `evaluator=` on WindowPC, and poc/time_series/bench_device.py):

      * recursive (per-node) — the GPU loses to the CPU at every batch size
        measured, 0.19-0.54×.  The circuit is launch-latency bound, and the
        Python recursion costs the same on both devices.
      * layered (compiled, the default) — the GPU wins from batch ~128 up,
        ~2× at batch 2048+, and keeps scaling where the CPU flattens.

    So "a circuit is GPU-hostile" was a fact about the recursion, not about
    circuits.  Measure per machine — `--device` is a knob, not an upgrade —
    but measure it with the layered evaluator.
    """
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class DegenerateModelError(RuntimeError):
    """
    Raised when a trained circuit is provably carrying no information.

    Three separate silent degeneracies have each produced a confident WRONG
    result in this project (leaf jitter not covering every leaf type; uniform
    sum-node weights in the DAG layout; τ attached at the root making
    E[τ|x] constant).  All three were invisible in the training loss and
    surfaced only in a query — after the conclusions had been drawn.  The fix
    is to make degeneracy LOUD: a predictive that does not depend on its input
    is refused rather than returned.
    """


# ═══════════════════════════════════════════════════════════════════════════
# Vtree selection
# ═══════════════════════════════════════════════════════════════════════════

def build_window_vtree(
    method: str,
    window: int,
    n_channels: int,
    X: Optional[torch.Tensor] = None,
    channel_groups: Optional[Sequence[Sequence[int]]] = None,
    seed: int = 0,
    max_arity: int = 4,
    n_partitions: int = 3,
):
    """
    Vtree over a flattened window.  Hand-built temporal structures and learned
    structures are deliberately interchangeable here — that substitution IS the
    T4 ablation, and every option yields a valid vtree, so all four circuit
    properties hold regardless of which one wins.
    """
    d = window * n_channels
    if method == "time":
        return time_channel_vtree(window, n_channels, mode="time")
    if method == "channel":
        return time_channel_vtree(window, n_channels, mode="channel")
    if method == "channel_groups":
        return time_channel_vtree(window, n_channels, mode="channel",
                                  channel_groups=channel_groups)
    if method == "random":
        return random_balanced_vtree(list(range(d)), seed=seed)

    # ── region graphs (n-ary, optionally multi-partition) ────────────────
    if method == "chain":              # HMM-shaped: the order-sensitive one
        return chain_region_graph(window, n_channels, emission="factorized")
    if method == "chain_grouped":
        return chain_region_graph(window, n_channels, emission="grouped",
                                  channel_groups=channel_groups)
    if method == "chain_full":
        return chain_region_graph(window, n_channels, emission="chain")
    if method in ("chain_perm_blocks", "chain_perm_features"):
        base = chain_region_graph(window, n_channels, emission="factorized")
        if method == "chain_perm_blocks":
            # timestep ORDER destroyed, same-timestep channels still together
            perm = timestep_block_permutation(window, n_channels, seed=seed)
        else:
            # blocking destroyed as well
            perm = np.random.default_rng(seed).permutation(
                window * n_channels).tolist()
        return permute_region_graph(base, perm)
    if method.endswith("_rg") or method.endswith("_rg_multi"):
        base = method.replace("_rg_multi", "").replace("_rg", "")
        if X is None:
            raise ValueError(f"{method!r} is learned from data; pass X")
        return learned_region_graph(
            X, method=base, seed=seed, max_arity=max_arity,
            n_partitions=n_partitions if method.endswith("_multi") else 1)

    if X is None:
        raise ValueError(f"vtree method {method!r} is learned from data; pass X")
    return learned_vtree(X, method=method, seed=seed)


def attach_variable(base, idx: int, where: str = "root"):
    """
    Extend a structure with one extra variable (here: τ).  Accepts a vtree OR a
    region graph, so the chain/HMM structure can carry a RUL label.

      where="root": τ becomes the sibling of the whole window.  The joint is
          p(x, τ) = Σ_{i,j} w_ij p_i(x) p_j(τ) — a full K×K coupling, so p(τ|x)
          is any convex combination of K learned RUL profiles.  On a CHAIN this
          is the principled choice: the K window units are the K hidden states
          at the head of the chain, i.e. summaries of the whole window, so the
          coupling is "which degradation mode is this unit in" → "how long does
          that mode have left".
      where="deep": τ is coupled low in the structure and the dependence
          propagates upward through every product above it.  More expressive,
          slower.  On a region graph this attaches τ to the deepest suffix
          region (for a chain, the LAST timestep — the one nearest failure).
    """
    tau_leaf = RegionNode(frozenset({idx}))

    if isinstance(base, RegionNode):
        full = base.scope | {idx}
        if where == "root":
            return RegionNode(frozenset(full), [(base, tau_leaf)])
        if where == "deep":
            cache: Dict[frozenset, RegionNode] = {}

            def rebuild(r: RegionNode) -> RegionNode:
                key = r.scope
                hit = cache.get(key)
                if hit is not None:
                    return hit
                new = RegionNode(frozenset(r.scope | {idx}))
                cache[key] = new
                if r.is_leaf:
                    new.partitions = [(RegionNode(r.scope), tau_leaf)]
                else:
                    # descend into the LAST child of the first partition: for a
                    # chain that is the deepest suffix, i.e. the newest data
                    part = list(r.partitions[0])
                    part[-1] = rebuild(part[-1])
                    new.partitions = [tuple(part)]
                return new

            return rebuild(base)
        raise KeyError(f"unknown attachment {where!r} (use 'root' or 'deep')")

    if where == "root":
        return VtreeInternal(base, VtreeLeaf(idx))
    if where == "deep":
        def rebuild_vt(node: VtreeNode) -> VtreeNode:
            if isinstance(node, VtreeLeaf):
                return VtreeInternal(node, VtreeLeaf(idx))
            return VtreeInternal(node.left, rebuild_vt(node.right))
        return rebuild_vt(base)
    raise KeyError(f"unknown attachment {where!r} (use 'root' or 'deep')")


def window_leaf(i: int, n_components: int, mixture_at_1: bool = False) -> nn.Module:
    """
    THE leaf choice for a sensor feature, in one place.

    `n_components` has always selected three things at once — the leaf CLASS,
    its initialisation RULE and the component COUNT — so "does a second
    component help?" has never been answerable: c=1 gives a `GaussianLeaf`
    seeded at MAD·1.4826, c=2 gives a `GaussianMixtureLeaf` seeded at std/n,
    and the 25.65 -> 20.22 jump could be either (hand-off §A.4).

    `mixture_at_1` is the fix, deliberately OPT-IN: it builds a 1-component
    `GaussianMixtureLeaf`, which is mathematically a single Gaussian and so
    isolates the class/init from the count.  Opt-in because switching the
    default would move every recorded 1-component number, and the whole point
    of the arm is to compare against them.  Read it as §A.5 says:

        GMLeaf(n=1) ~ 20.2  ->  the win was the INITIALISATION, and the fix is
                                to seed GaussianLeaf at std; mixture leaves are
                                not needed at all
        GMLeaf(n=1) ~ 25.6  ->  the win is real capacity from the second
                                component, and mixture leaves earn their place
    """
    if n_components > 1 or (mixture_at_1 and n_components == 1):
        return GaussianMixtureLeaf(i, n_components=n_components)
    return GaussianLeaf(i)


def mixed_leaf_factory(
    tau_idx: int, n_bins: int, n_components: int = 1,
    mixture_at_1: bool = False,
) -> "callable":
    """
    Per-feature leaf factory: Gaussian(-mixture) on the sensor window,
    Categorical on τ.  τ MUST get a leaf with a closed-form interval mass —
    that is what makes the survival/censoring query exact (the heavy-tailed
    InputNode has no closed-form CDF and raises if boxed).
    """
    def factory(i: int) -> nn.Module:
        if i == tau_idx:
            return CategoricalLeaf(i, n_categories=n_bins)
        return window_leaf(i, n_components, mixture_at_1)
    return factory


# ═══════════════════════════════════════════════════════════════════════════
# Model 1 — window density for anomaly detection
# ═══════════════════════════════════════════════════════════════════════════

def _compile_or_fallback(model, x_probe: torch.Tensor, box_feature=None):
    """
    Use the compiled layer-parallel evaluator when the circuit supports it,
    the recursion when it does not.  Shared by WindowPC and SurvivalPC.

    `box_feature` (SurvivalPC's τ) additionally gates the BOX query, because
    that is the term carrying the censored likelihood: a compiled evaluator
    that is right on densities and wrong on intervals would silently corrupt
    exactly the quantity the survival model exists to compute.

    Signed/SOS circuits are refused by the compiler and keep running on the
    recursion — slower and correct, which is the right way round.
    """
    model.compiled = None
    if getattr(model, "evaluator", "layered") == "recursive" \
            or getattr(model, "use_sos", False):
        return model.pc
    try:
        comp = model.pc.compile_(x_probe=x_probe, device=model.device)
        if box_feature is not None:
            lo = torch.full((len(x_probe),), 1.0, device=x_probe.device)
            comp.assert_matches_reference(
                model.pc.root, x_probe,
                boxes={int(box_feature): (lo, float("inf"))})
        model.compiled = comp
        return comp
    except (NotImplementedError, RuntimeError) as exc:
        print(f"    [pc] compiled evaluator unavailable ({exc}); "
              "falling back to the recursion")
        model.pc.use_recursive()
        model.compiled = None
        return model.pc


class WindowPC:
    """Exact window density; anomaly score = −log p(x)."""

    def __init__(
        self,
        window: int,
        n_channels: int,
        vtree_method: str = "time",
        n_sum_components: int = 6,
        leaf_components: int = 1,
        channel_groups: Optional[Sequence[Sequence[int]]] = None,
        use_sos: bool = False,
        delta: bool = False,
        seed: int = 0,
        device: Optional[str] = None,
        weight_jitter: float = 0.5,
        evaluator: str = "layered",
        mixture_at_1: bool = False,
    ):
        self.window, self.n_channels = window, n_channels
        self.d = window * n_channels
        self.vtree_method = vtree_method
        self.K = n_sum_components
        self.leaf_components = leaf_components
        self.mixture_at_1 = mixture_at_1
        self.channel_groups = channel_groups
        self.use_sos = use_sos
        self.delta = delta
        self.seed = seed
        self.device = resolve_device(device)
        self.weight_jitter = weight_jitter
        # "layered" = the compiled layer-parallel evaluator (default);
        # "recursive" = the per-node reference.  Same numbers either way — the
        # knob exists so the A/B is one flag, not a code change.
        self.evaluator = evaluator
        self.compiled = None
        self.pc = None
        self.history: List[float] = []

    def _prep(self, X: torch.Tensor) -> torch.Tensor:
        """Optional first-difference reparameterisation.  Unit-determinant, so
        the density stays exact and log p is directly comparable."""
        X = X.to(self.device)
        if not self.delta:
            return X
        return delta_window_transform(X, self.window, self.n_channels)

    def _leaf_factory(self):
        c, m = self.leaf_components, self.mixture_at_1
        return lambda i: window_leaf(i, c, m)

    def fit(self, X: torch.Tensor, epochs: int = 60, lr: float = 0.05,
            batch_size: int = 256, verbose: bool = False,
            log_every: int = 0) -> "WindowPC":
        """
        Structure learning and closed-form leaf initialisation happen on CPU
        (they are numpy/statistics, not tensor algebra); only the gradient loop
        runs on `self.device`.  The per-epoch NLL is kept in `self.history` so
        the pipeline can write a training curve for every run.
        """
        torch.manual_seed(self.seed)
        X_cpu = X.detach().cpu()
        if self.delta:
            X_cpu = delta_window_transform(X_cpu, self.window, self.n_channels)
        vt = build_window_vtree(self.vtree_method, self.window, self.n_channels,
                                X=X_cpu, channel_groups=self.channel_groups,
                                seed=self.seed)
        if self.use_sos:
            # SOS / squared circuit: subtractive mixtures, exactly normalised by
            # the pairwise construction.  region_graph=True is required — the
            # tree layout has the same K^depth blowup the rebuild removed.
            self.pc = SquaredPC(vt, n_sum_components=self.K,
                                leaf_factory=self._leaf_factory(),
                                seed=self.seed, region_graph=True)
        else:
            self.pc = RegionGraphPC(vt, n_sum_components=self.K,
                                    leaf_factory=self._leaf_factory(),
                                    weight_jitter=self.weight_jitter, seed=self.seed)
        self.pc.validate()
        self.pc.fit_leaves(X_cpu)
        move_circuit_(self.pc, self.device)   # DAG-safe; .to() is exponential here

        Xd = X_cpu.to(self.device)
        # Compile to the layer-parallel evaluator, gated against the recursive
        # reference on a real batch.  This is the difference between one Python
        # frame per NODE (10^3-10^5 per step) and one per LAYER (~16), and it is
        # what makes the GPU worth using at all — see poc/time_series/bench_device.
        model = self._compile_or_fallback(Xd[: min(len(Xd), 64)])
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        n = len(Xd)
        self.history = []
        for ep in track(range(epochs), f"fit {self.vtree_method} K={self.K}",
                        total=epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for s in range(0, n, batch_size):
                xb = Xd[perm[s:s + batch_size]]
                loss = -model.log_prob(xb).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += float(loss.detach()) * len(xb)
            self.history.append(tot / max(n, 1))
            every = log_every or (max(epochs // 6, 1) if verbose else 0)
            if every and ep % every == 0:
                print(f"    [pc] epoch {ep:3d}  nll {self.history[-1]:8.3f}")
        if self.compiled is not None:
            # the DAG owns the semantics (validate/MPE/serialisation): give it
            # the trained values back before anything else reads it
            self.compiled.write_back()
        return self

    def _compile_or_fallback(self, x_probe: torch.Tensor, box_feature=None):
        return _compile_or_fallback(self, x_probe, box_feature)

    # ── degeneracy guardrail ─────────────────────────────────────────────

    @torch.no_grad()
    def channel_sensitivity(self, X: torch.Tensor, shift: float = 3.0
                            ) -> np.ndarray:
        """
        Mean |Δ −log p| when one whole channel is shifted by `shift` sd.

        The PARTIAL-collapse detector.  `assert_informative` below only rejects
        a density that is constant in x, and a circuit reading 2 of 6 channels
        is not constant — it has an ordinary NLL curve, an ordinary score
        spread, and no per-channel claim it makes is worth anything.  That is
        the partial version of degeneracies #1 and #2 and nothing looked for it
        until 2026-08-05.
        """
        base = float(self.score(X).mean())
        out = np.zeros(self.n_channels)
        for c in range(self.n_channels):
            Xp = X.clone()
            Xp[:, [t * self.n_channels + c for t in range(self.window)]] += shift
            out[c] = abs(float(self.score(Xp).mean()) - base)
        return out

    @torch.no_grad()
    def assert_informative(self, X: torch.Tensor, min_sd: float = 1e-3,
                           check_channels: bool = True,
                           min_channel_frac: float = 0.10) -> float:
        """
        Refuse a circuit that is provably carrying no information — in either
        of the two ways it can happen.

        1. the score is constant in x (total collapse), and
        2. the score ignores whole CHANNELS (partial collapse) — measured
           against the median channel's influence rather than an absolute nat
           threshold, because the nat scale moves with window length, channel
           count and fit quality.  A fixed constant here would be the same
           literal-instead-of-state mistake that made the `@floor` diagnostic
           structurally unable to fire (hand-off §A.6).

        Every silent degeneracy this project has hit (see DegenerateModelError)
        showed up in one of these two and nowhere else: the training loss
        looked fine while the model had collapsed to a product of marginals or
        worse.  Cheap, so it is checked on every fitted model rather than
        trusted.
        """
        Xs = X[: min(len(X), 512)]
        s = self.score(Xs)
        sd = float(s.std())
        if not np.isfinite(sd) or sd < min_sd:
            raise DegenerateModelError(
                f"degenerate density: -log p(x) has sd {sd:.3e} over "
                f"{len(Xs)} inputs, i.e. the score barely depends on x. "
                "Check weight_jitter (must be > 0), leaf jitter and the vtree "
                "before trusting any number from this model.")

        if check_channels and self.n_channels > 2:
            sens = self.channel_sensitivity(Xs[: min(len(Xs), 128)])
            med = float(np.median(sens))
            blind = [c for c, v in enumerate(sens) if v < min_channel_frac * med]
            if blind and med > 0:
                raise DegenerateModelError(
                    f"partially degenerate density: channels {blind} move "
                    f"-log p(x) by under {min_channel_frac:.0%} of the median "
                    f"channel under a 3-sd shift "
                    f"(sensitivities {np.round(sens, 2).tolist()}). The score "
                    "still varies with x, so a constant-score check passes and "
                    "every per-channel claim about these channels is empty.")
        return sd

    @torch.no_grad()
    def score(self, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """−log p(x).  Always returned on CPU: every consumer (metrics, plots,
        sklearn baselines) is numpy-side, and a stray device tensor there is a
        crash at the end of a long run rather than at the start."""
        X = self._prep(X)
        out = [-self.pc.log_prob(X[s:s + batch_size]) for s in range(0, len(X), batch_size)]
        return torch.cat(out).cpu()

    @torch.no_grad()
    def score_with_missing(self, X: torch.Tensor, dead_channels: Sequence[int],
                           batch_size: int = 512) -> torch.Tensor:
        """
        −log p(observed part) with whole channels marginalised OUT exactly.
        No imputation: the dead sensors simply leave the query.  This is the
        query a reconstruction-based detector cannot express.
        """
        marg = [t * self.n_channels + c
                for t in range(self.window) for c in dead_channels]
        X = self._prep(X)
        out = [-self.pc.log_marginal(X[s:s + batch_size], marg)
               for s in range(0, len(X), batch_size)]
        return torch.cat(out).cpu()

    @torch.no_grad()
    def typed_scores(self, X: torch.Tensor, batch_size: int = 512
                     ) -> Dict[str, torch.Tensor]:
        """
        Exact per-channel decomposition of the anomaly:

          marginal_c   = −log p(x_c)              "is this channel odd on its own?"
          conditional_c= −log p(x_c | x_{−c})     "is it odd GIVEN the others?"
          structural_c = conditional_c − marginal_c

        A purely univariate anomaly has structural ≈ 0; a broken cross-channel
        relation shows up as large structural even when every channel is
        individually unremarkable.  Both terms are exact marginals of one
        circuit, so the decomposition costs two extra passes, not a new model.
        """
        C, W = self.n_channels, self.window
        X = self._prep(X)
        all_feats = set(range(self.d))
        marg_out, cond_out = [], []
        for c in track(range(C), "typed decomposition (per channel)", total=C):
            chan = [t * C + c for t in range(W)]
            others = sorted(all_feats - set(chan))
            m, cd = [], []
            for s in range(0, len(X), batch_size):
                xb = X[s:s + batch_size]
                lp_c = self.pc.log_marginal(xb, others)          # log p(x_c)
                lp_o = self.pc.log_marginal(xb, chan)            # log p(x_-c)
                lp_j = self.pc.log_prob(xb)                      # log p(x)
                m.append(-lp_c.cpu())
                cd.append(-(lp_j - lp_o).cpu())                  # −log p(x_c|x_-c)
            marg_out.append(torch.cat(m)); cond_out.append(torch.cat(cd))
        marginal = torch.stack(marg_out, dim=1)
        conditional = torch.stack(cond_out, dim=1)
        return {"marginal": marginal, "conditional": conditional,
                "structural": conditional - marginal}

    @torch.no_grad()
    def time_channel_attribution(self, X: torch.Tensor,
                                 batch_size: int = 512) -> torch.Tensor:
        """
        Exact per-(t, c) conditional surprise −log p(x_tc | everything else),
        shape (N, window, C).  The heat-map an operator reads: WHEN and WHERE.
        """
        Xp = self._prep(X)
        W, C = self.window, self.n_channels
        grid = torch.zeros(len(Xp), W, C)
        for t in track(range(W), f"(t,c) attribution grid {W}x{C}", total=W):
            for c in range(C):
                f = t * C + c
                others = [i for i in range(self.d) if i != f]
                cols = []
                for s in range(0, len(Xp), batch_size):
                    xb = Xp[s:s + batch_size]
                    cols.append((self.pc.log_marginal(xb, [f])
                                 - self.pc.log_prob(xb)).cpu())
                grid[:, t, c] = torch.cat(cols)
        return grid

    @torch.no_grad()
    def chain_rule_attribution(self, X: torch.Tensor, order=None,
                               batch_size: int = 512) -> torch.Tensor:
        """
        EXACTLY COMPLETE attribution, shape (N, d).

        For any ordering pi of the variables the chain rule gives
            log p(x) = sum_i log p(x_{pi_i} | x_{pi_1..pi_{i-1}}),
        and every term is a difference of two exact marginals of this circuit.
        So the attributions sum to the score with ZERO residual — completeness
        is a theorem here, not an approximation target.  SHAP only satisfies
        completeness up to its estimation error, and its conditional variant
        needs conditionals that are unavailable for the models it explains.

        Averaging this over random orderings converges to the exact conditional
        Shapley value: only the ordering average is sampled, never the
        conditional itself.
        """
        Xp = self._prep(X)
        d = self.d
        order = list(range(d)) if order is None else list(order)
        out = torch.zeros(len(Xp), d)
        n_batches = max((len(Xp) + batch_size - 1) // batch_size, 1)
        for s in track(range(0, len(Xp), batch_size),
                       f"chain-rule attribution ({d} exact conditionals)",
                       total=n_batches):
            xb = Xp[s:s + batch_size]
            # prefix marginal: everything from position i onward integrated out
            prev = self.pc.log_marginal(xb, order)          # = 0 (all marginalised)
            for i, f in enumerate(order):
                rest = order[i + 1:]
                cur = self.pc.log_marginal(xb, rest) if rest else self.pc.log_prob(xb)
                out[s:s + len(xb), f] = (cur - prev).cpu()
                prev = cur
        return out

    @torch.no_grad()
    def shapley_channels(self, X: torch.Tensor, n_orders: int = 8,
                         seed: int = 0) -> torch.Tensor:
        """
        Conditional Shapley values over CHANNELS, (N, C), averaged over
        `n_orders` random channel orderings.  Each term is exact; only the
        ordering average is Monte-Carlo, which is the reverse of KernelSHAP
        (whose conditional itself is approximated).
        """
        Xp = self._prep(X)
        W, C = self.window, self.n_channels
        g = torch.Generator().manual_seed(seed)
        chan = lambda c: [t * C + c for t in range(W)]
        acc = torch.zeros(len(Xp), C)
        for _ in track(range(n_orders), f"conditional Shapley ({n_orders} orders)",
                       total=n_orders):
            perm = torch.randperm(C, generator=g).tolist()
            marg = [f for c in perm for f in chan(c)]
            prev = self.pc.log_marginal(Xp, marg)
            for i, c in enumerate(perm):
                rest = [f for cc in perm[i + 1:] for f in chan(cc)]
                cur = (self.pc.log_marginal(Xp, rest) if rest
                       else self.pc.log_prob(Xp))
                acc[:, c] += (cur - prev).cpu()
                prev = cur
        return -(acc / n_orders)          # higher = more responsible

    def size(self) -> Dict[str, int]:
        from src.probabilistic_circuits import circuit_size
        return circuit_size(self.pc.root)


# ═══════════════════════════════════════════════════════════════════════════
# Model 2 — joint (window, τ) circuit with exact censored likelihood
# ═══════════════════════════════════════════════════════════════════════════

class SurvivalPC:
    """
    Joint exact density over (window, τ).  The contribution is the training
    objective and the query set, not the architecture:

        observed failure (δ=1):  ℓ = log p(x, τ = k)
        right-censored  (δ=0):   ℓ = log P(x, τ ≥ c)   ← exact box query

    Both terms come from the same circuit and the same partition function, so
    censored and uncensored units are combined on a single likelihood scale.
    """

    def __init__(
        self,
        window: int,
        n_channels: int,
        n_bins: int,
        cap: float,
        vtree_method: str = "time",
        n_sum_components: int = 8,
        leaf_components: int = 1,
        mixture_at_1: bool = False,
        tau_where: str = "deep",
        channel_groups: Optional[Sequence[Sequence[int]]] = None,
        delta: bool = False,
        seed: int = 0,
        device: Optional[str] = None,
        weight_jitter: float = 0.5,
        evaluator: str = "layered",
    ):
        self.window, self.n_channels = window, n_channels
        self.d = window * n_channels
        self.tau_idx = self.d
        self.n_bins, self.cap = n_bins, cap
        self.vtree_method, self.K = vtree_method, n_sum_components
        self.leaf_components = leaf_components
        self.mixture_at_1 = mixture_at_1
        self.tau_where = tau_where
        self.channel_groups = channel_groups
        self.delta = delta
        self.seed = seed
        self.device = resolve_device(device)
        self.weight_jitter = weight_jitter
        self.evaluator = evaluator
        self.compiled = None
        self.pc: Optional[RegionGraphPC] = None
        self.history: List[float] = []
        self.target_sd = 0.0

    # ── construction / training ──────────────────────────────────────────

    def _prep(self, X: torch.Tensor) -> torch.Tensor:
        """First-difference the WINDOW only.  Unit determinant, so the joint
        density over (window, tau) stays exactly normalised; tau is a separate
        variable and must never be differenced."""
        X = X.to(self.device)
        if not self.delta:
            return X
        return delta_window_transform(X, self.window, self.n_channels)

    def _augment(self, X: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        Xp = self._prep(X)
        return torch.cat([Xp, tau.reshape(-1, 1).to(dtype=Xp.dtype,
                                                    device=Xp.device)], dim=1)

    def fit(
        self,
        X: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        epochs: int = 60,
        lr: float = 0.05,
        batch_size: int = 256,
        use_censored: bool = True,
        verbose: bool = False,
        log_every: int = 0,
    ) -> "SurvivalPC":
        """
        use_censored=False drops the censored units entirely — the standard
        practice this PoC is meant to beat.  It is the ablation, not a bug.
        """
        torch.manual_seed(self.seed)
        if not use_censored:
            keep = delta == 1
            X, tau, delta = X[keep], tau[keep], delta[keep]

        # structure + leaf init on CPU (numpy statistics), training on device
        X_cpu = X.detach().cpu()
        base_in = (delta_window_transform(X_cpu, self.window, self.n_channels)
                   if self.delta else X_cpu)
        base = build_window_vtree(self.vtree_method, self.window, self.n_channels,
                                  X=base_in, channel_groups=self.channel_groups,
                                  seed=self.seed)
        vt = attach_variable(base, self.tau_idx, where=self.tau_where)
        self.pc = RegionGraphPC(
            vt, n_sum_components=self.K,
            leaf_factory=mixed_leaf_factory(self.tau_idx, self.n_bins,
                                            self.leaf_components,
                                            self.mixture_at_1),
            weight_jitter=self.weight_jitter,
            seed=self.seed)
        self.pc.validate()
        # what "constant" means has to be relative to the target, not the cap
        # (hand-off §B.5); recorded here so `predict` has a reference
        self.target_sd = float(
            (tau.detach().cpu().float() + 0.5) .mul(self.cap / self.n_bins).std())
        self.pc.fit_leaves(torch.cat(
            [base_in, tau.detach().cpu().reshape(-1, 1).to(base_in.dtype)], dim=1))
        move_circuit_(self.pc, self.device)   # DAG-safe; .to() is exponential here

        Xd = X_cpu.to(self.device)
        taud = tau.to(self.device)
        deltad = delta.to(self.device)
        # Compile, gated on BOTH query types this loop uses: the plain density
        # for observed failures and the box query for censored units.  A box
        # bug would corrupt exactly the censored term — the one thing this
        # model exists to get right — so it is checked, not assumed.
        inf = float("inf")
        probe = self._augment(Xd[: min(len(Xd), 64)],
                              taud[: min(len(Xd), 64)].to(Xd.dtype))
        model = _compile_or_fallback(self, probe, box_feature=self.tau_idx)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        n = len(Xd)
        self.history = []
        for ep in track(range(epochs),
                        f"survival fit ({'censored' if use_censored else 'observed'})",
                        total=epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb, tb, db = Xd[idx], taud[idx], deltad[idx]
                zb = self._augment(xb, tb)
                obs = db == 1
                terms = []
                if obs.any():
                    terms.append(model.log_prob(zb[obs]))
                if (~obs).any():
                    # P(τ ≥ c) per sample: ONE circuit pass, per-sample interval
                    lo = tb[~obs].to(zb.dtype)
                    terms.append(model.log_box(
                        zb[~obs], {self.tau_idx: (lo, inf)}))
                loss = -torch.cat(terms).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            self.history.append(tot / max(n, 1))
            every = log_every or (max(epochs // 6, 1) if verbose else 0)
            if every and ep % every == 0:
                print(f"    [surv] epoch {ep:3d}  censored-NLL {self.history[-1]:8.3f}")
        if self.compiled is not None:
            self.compiled.write_back()
        return self

    # ── exact queries ────────────────────────────────────────────────────

    @torch.no_grad()
    def log_pmf(self, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """
        Exact log p(τ = k | x) for every bin, shape (N, n_bins).  Computed as
        n_bins joint evaluations renormalised — the conditional is a ratio of
        two exact quantities, so no normalisation constant is ever estimated.
        """
        out = []
        n_batches = max((len(X) + batch_size - 1) // batch_size, 1)
        for s in track(range(0, len(X), batch_size),
                       f"log p(tau|x) · {self.n_bins} bins", total=n_batches):
            xb = X[s:s + batch_size]
            cols = []
            for k in range(self.n_bins):
                tau_k = torch.full((len(xb),), float(k), device=self.device)
                cols.append(self.pc.log_prob(self._augment(xb, tau_k)))
            joint = torch.stack(cols, dim=1)
            out.append((joint - torch.logsumexp(joint, dim=1, keepdim=True)).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def log_survival(self, X: torch.Tensor, t_bin: float,
                     batch_size: int = 512) -> torch.Tensor:
        """
        Exact log S(t | x) = log P(τ > t_bin | x), one box query per batch
        divided by the exact marginal p(x).  `t_bin` may be a scalar or a
        per-sample tensor.
        """
        inf = float("inf")
        out = []
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            zb = self._augment(xb, torch.zeros(len(xb), device=self.device))
            tb = (t_bin[s:s + batch_size].to(self.device)
                  if isinstance(t_bin, torch.Tensor)
                  else torch.full((len(xb),), float(t_bin), device=self.device))
            num = self.pc.log_box(zb, {self.tau_idx: (tb + 1.0, inf)})
            den = self.pc.log_marginal(zb, [self.tau_idx])
            out.append((num - den).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def anomaly_score(self, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """
        −log p(x) with τ marginalised out: the SAME circuit that predicts RUL
        also detects anomalies, with no second model and no extra training.
        """
        out = []
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            zb = self._augment(xb, torch.zeros(len(xb), device=self.device))
            out.append(-self.pc.log_marginal(zb, [self.tau_idx]).cpu())
        return torch.cat(out)

    def bin_edges(self) -> torch.Tensor:
        return torch.linspace(0, self.cap, self.n_bins + 1)

    def bin_centers(self) -> torch.Tensor:
        edges = self.bin_edges()
        return 0.5 * (edges[:-1] + edges[1:])

    @torch.no_grad()
    def predict(self, X: torch.Tensor, check_degenerate: bool = True,
                min_sd_frac: float = 0.05) -> Dict[str, torch.Tensor]:
        """
        Point + distributional RUL predictions in CYCLES.

        GUARDRAIL (hand-off §3).  A predictive that is constant across inputs
        is refused instead of returned.  This is not defensive programming for
        its own sake: `tau_where='root'` once made `predict` emit the same
        102.4 cycles for all 851 test windows (sd 0.0), and an entire
        pre-registered gate was run and reported on that model before anyone
        noticed.  The training loss showed nothing wrong.  Everything the
        circuit does downstream — CRPS, PICP, the censoring ablation — is
        meaningless once p(τ|x) stops depending on x, so it is checked here,
        at the one place every consumer passes through.
        """
        logp = self.log_pmf(X)
        p = logp.exp()
        centers = self.bin_centers().to(p.device)
        mean = (p * centers).sum(1)
        mode = centers[p.argmax(1)]
        cdf = p.cumsum(1)

        if check_degenerate and len(mean) > 1:
            sd = float(mean.std())
            # The threshold is a fraction of the TARGET's own spread, not of
            # the cap.  It used to be 1e-3*cap = 0.13 cycles against a target
            # whose sd is ~31 cycles — 315x too loose, so a predictive with 8%
            # of the target's spread sailed through (hand-off §B.5).  `cap/4`
            # is the fallback when no training spread was recorded: RUL is
            # capped-uniform-ish, so sd ~ cap/4 is the right order.
            ref = float(getattr(self, "target_sd", 0.0)) or max(self.cap, 1.0) / 4.0
            if not np.isfinite(sd) or sd < min_sd_frac * ref:
                raise DegenerateModelError(
                    f"degenerate predictive: E[tau|x] has sd {sd:.3g} cycles "
                    f"over {len(mean)} inputs, under {min_sd_frac:.0%} of the "
                    f"target's own spread ({ref:.3g}) — p(tau|x) barely "
                    "depends on x. Check tau_where (use 'deep'; 'root' caps "
                    "the coupling at a KxK latent and collapses at most K, "
                    "non-monotonically), weight_jitter (must be > 0) and leaf "
                    "jitter before trusting any result from this model.")

        # ── quantiles, in two conventions ─────────────────────────────────
        # `q_idx(level)` is the discrete quantile: the first bin whose
        # cumulative mass reaches `level`.  What you then REPORT for that bin
        # is a choice, and it is not a cosmetic one:
        #
        #   centres  q05/q95   the bin's midpoint.  Right for a point summary
        #                      and for comparing against another discretised
        #                      quantity.  WRONG as the endpoint of an interval
        #                      that has to cover a continuous target: it gives
        #                      away half a bin at each end, for reasons that
        #                      have nothing to do with the model.
        #   edges    q05_edge  the OUTER edges of the same bins — the lower
        #            q95_edge  edge of the low bin, the upper edge of the high
        #                      one.  This is the interval whose coverage the
        #                      pmf actually claims.
        #
        # Measured (hand-off §B.2, bins=25 cap=130): PICP 0.616 on centres vs
        # 0.929 on edges, for one extra bin width of MPIW, with the PIT
        # variance already at 1/12 — i.e. most of the recorded under-coverage
        # was the convention, not the density.  BOTH are returned and neither
        # is renamed: `q05`/`q95` keep their meaning so every number recorded
        # against them stays comparable, which is the same one-thing-at-a-time
        # rule that the sigma-floor flag broke (§A.1).
        edges = self.bin_edges().to(p.device)

        def q_idx(level: float) -> torch.Tensor:
            return (cdf < level).sum(1).clamp(max=self.n_bins - 1)

        lo_i, hi_i = q_idx(0.05), q_idx(0.95)
        return {"pmf": p, "mean": mean, "mode": mode,
                "q05": centers[lo_i], "q50": centers[q_idx(0.50)],
                "q95": centers[hi_i],
                "q05_edge": edges[lo_i], "q95_edge": edges[hi_i + 1]}

    def size(self) -> Dict[str, int]:
        return self.pc.size()
