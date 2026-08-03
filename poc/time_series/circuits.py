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

from src.probabilistic_circuits import (
    CategoricalLeaf,
    RegionNode,
    chain_region_graph,
    delta_window_transform,
    learned_region_graph,
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
)


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


def mixed_leaf_factory(
    tau_idx: int, n_bins: int, n_components: int = 1
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
        if n_components > 1:
            return GaussianMixtureLeaf(i, n_components=n_components)
        return GaussianLeaf(i)
    return factory


# ═══════════════════════════════════════════════════════════════════════════
# Model 1 — window density for anomaly detection
# ═══════════════════════════════════════════════════════════════════════════

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
    ):
        self.window, self.n_channels = window, n_channels
        self.d = window * n_channels
        self.vtree_method = vtree_method
        self.K = n_sum_components
        self.leaf_components = leaf_components
        self.channel_groups = channel_groups
        self.use_sos = use_sos
        self.delta = delta
        self.seed = seed
        self.pc = None

    def _prep(self, X: torch.Tensor) -> torch.Tensor:
        """Optional first-difference reparameterisation.  Unit-determinant, so
        the density stays exact and log p is directly comparable."""
        if not self.delta:
            return X
        return delta_window_transform(X, self.window, self.n_channels)

    def _leaf_factory(self):
        c = self.leaf_components
        return (lambda i: GaussianMixtureLeaf(i, n_components=c)) if c > 1 else GaussianLeaf

    def fit(self, X: torch.Tensor, epochs: int = 60, lr: float = 0.05,
            batch_size: int = 256, verbose: bool = False) -> "WindowPC":
        torch.manual_seed(self.seed)
        X = self._prep(X)
        vt = build_window_vtree(self.vtree_method, self.window, self.n_channels,
                                X=X, channel_groups=self.channel_groups, seed=self.seed)
        if self.use_sos:
            # SOS / squared circuit: subtractive mixtures, exactly normalised by
            # the pairwise construction.  region_graph=True is required — the
            # tree layout has the same K^depth blowup the rebuild removed.
            self.pc = SquaredPC(vt, n_sum_components=self.K,
                                leaf_factory=self._leaf_factory(),
                                seed=self.seed, region_graph=True)
        else:
            self.pc = RegionGraphPC(vt, n_sum_components=self.K,
                                    leaf_factory=self._leaf_factory(), seed=self.seed)
        self.pc.validate()
        self.pc.fit_leaves(X)
        opt = torch.optim.Adam(self.pc.parameters(), lr=lr)
        n = len(X)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, batch_size):
                xb = X[perm[s:s + batch_size]]
                loss = -self.pc.log_prob(xb).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pc.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(xb)
            if verbose and ep % max(epochs // 6, 1) == 0:
                print(f"    [pc] epoch {ep:3d}  nll {tot / n:8.3f}")
        return self

    @torch.no_grad()
    def score(self, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        X = self._prep(X)
        out = [-self.pc.log_prob(X[s:s + batch_size]) for s in range(0, len(X), batch_size)]
        return torch.cat(out)

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
        return torch.cat(out)

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
        for c in range(C):
            chan = [t * C + c for t in range(W)]
            others = sorted(all_feats - set(chan))
            m, cd = [], []
            for s in range(0, len(X), batch_size):
                xb = X[s:s + batch_size]
                lp_c = self.pc.log_marginal(xb, others)          # log p(x_c)
                lp_o = self.pc.log_marginal(xb, chan)            # log p(x_-c)
                lp_j = self.pc.log_prob(xb)                      # log p(x)
                m.append(-lp_c)
                cd.append(-(lp_j - lp_o))                        # −log p(x_c|x_-c)
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
        for t in range(W):
            for c in range(C):
                f = t * C + c
                others = [i for i in range(self.d) if i != f]
                cols = []
                for s in range(0, len(Xp), batch_size):
                    xb = Xp[s:s + batch_size]
                    cols.append(self.pc.log_marginal(xb, [f]) - self.pc.log_prob(xb))
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
        for s in range(0, len(Xp), batch_size):
            xb = Xp[s:s + batch_size]
            # prefix marginal: everything from position i onward integrated out
            prev = self.pc.log_marginal(xb, order)          # = 0 (all marginalised)
            for i, f in enumerate(order):
                rest = order[i + 1:]
                cur = self.pc.log_marginal(xb, rest) if rest else self.pc.log_prob(xb)
                out[s:s + len(xb), f] = cur - prev
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
        for _ in range(n_orders):
            perm = torch.randperm(C, generator=g).tolist()
            marg = [f for c in perm for f in chan(c)]
            prev = self.pc.log_marginal(Xp, marg)
            for i, c in enumerate(perm):
                rest = [f for cc in perm[i + 1:] for f in chan(cc)]
                cur = (self.pc.log_marginal(Xp, rest) if rest
                       else self.pc.log_prob(Xp))
                acc[:, c] += cur - prev
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
        tau_where: str = "root",
        channel_groups: Optional[Sequence[Sequence[int]]] = None,
        delta: bool = False,
        seed: int = 0,
    ):
        self.window, self.n_channels = window, n_channels
        self.d = window * n_channels
        self.tau_idx = self.d
        self.n_bins, self.cap = n_bins, cap
        self.vtree_method, self.K = vtree_method, n_sum_components
        self.leaf_components = leaf_components
        self.tau_where = tau_where
        self.channel_groups = channel_groups
        self.delta = delta
        self.seed = seed
        self.pc: Optional[RegionGraphPC] = None

    # ── construction / training ──────────────────────────────────────────

    def _prep(self, X: torch.Tensor) -> torch.Tensor:
        """First-difference the WINDOW only.  Unit determinant, so the joint
        density over (window, tau) stays exactly normalised; tau is a separate
        variable and must never be differenced."""
        if not self.delta:
            return X
        return delta_window_transform(X, self.window, self.n_channels)

    def _augment(self, X: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._prep(X), tau.reshape(-1, 1).to(X.dtype)], dim=1)

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
    ) -> "SurvivalPC":
        """
        use_censored=False drops the censored units entirely — the standard
        practice this PoC is meant to beat.  It is the ablation, not a bug.
        """
        torch.manual_seed(self.seed)
        if not use_censored:
            keep = delta == 1
            X, tau, delta = X[keep], tau[keep], delta[keep]

        base = build_window_vtree(self.vtree_method, self.window, self.n_channels,
                                  X=self._prep(X), channel_groups=self.channel_groups,
                                  seed=self.seed)
        vt = attach_variable(base, self.tau_idx, where=self.tau_where)
        self.pc = RegionGraphPC(
            vt, n_sum_components=self.K,
            leaf_factory=mixed_leaf_factory(self.tau_idx, self.n_bins,
                                            self.leaf_components),
            seed=self.seed)
        self.pc.validate()
        self.pc.fit_leaves(self._augment(X, tau))

        opt = torch.optim.Adam(self.pc.parameters(), lr=lr)
        n = len(X)
        inf = float("inf")
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb, tb, db = X[idx], tau[idx], delta[idx]
                zb = self._augment(xb, tb)
                obs = db == 1
                terms = []
                if obs.any():
                    terms.append(self.pc.log_prob(zb[obs]))
                if (~obs).any():
                    # P(τ ≥ c) per sample: ONE circuit pass, per-sample interval
                    lo = tb[~obs].to(zb.dtype)
                    terms.append(self.pc.log_box(
                        zb[~obs], {self.tau_idx: (lo, inf)}))
                loss = -torch.cat(terms).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pc.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            if verbose and ep % max(epochs // 6, 1) == 0:
                print(f"    [surv] epoch {ep:3d}  censored-NLL {tot / n:8.3f}")
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
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            cols = []
            for k in range(self.n_bins):
                tau_k = torch.full((len(xb),), float(k))
                cols.append(self.pc.log_prob(self._augment(xb, tau_k)))
            joint = torch.stack(cols, dim=1)
            out.append(joint - torch.logsumexp(joint, dim=1, keepdim=True))
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
            zb = self._augment(xb, torch.zeros(len(xb)))
            tb = (t_bin[s:s + batch_size] if isinstance(t_bin, torch.Tensor)
                  else torch.full((len(xb),), float(t_bin)))
            num = self.pc.log_box(zb, {self.tau_idx: (tb + 1.0, inf)})
            den = self.pc.log_marginal(zb, [self.tau_idx])
            out.append(num - den)
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
            zb = self._augment(xb, torch.zeros(len(xb)))
            out.append(-self.pc.log_marginal(zb, [self.tau_idx]))
        return torch.cat(out)

    def bin_centers(self) -> torch.Tensor:
        edges = torch.linspace(0, self.cap, self.n_bins + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Point + distributional RUL predictions in CYCLES."""
        logp = self.log_pmf(X)
        p = logp.exp()
        centers = self.bin_centers()
        mean = (p * centers).sum(1)
        mode = centers[p.argmax(1)]
        cdf = p.cumsum(1)
        def q(level: float) -> torch.Tensor:
            idx = (cdf < level).sum(1).clamp(max=self.n_bins - 1)
            return centers[idx]
        return {"pmf": p, "mean": mean, "mode": mode,
                "q05": q(0.05), "q50": q(0.50), "q95": q(0.95)}

    def size(self) -> Dict[str, int]:
        return self.pc.size()
