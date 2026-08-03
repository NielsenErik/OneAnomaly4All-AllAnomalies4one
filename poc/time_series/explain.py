"""
Explanation layer: attribution methods, faithfulness metrics, plots, and
worked examples.

The thesis this file exists to test: on this data the circuit only *ties* the
best detectors on AUROC, so its value has to be that it can say WHY — exactly,
completely, and in a form an operator can act on — where the methods tying it
cannot.

Three things are measured, because "explainable" is three separate claims:

  1. CORRECTNESS   does the explanation point at the channels that were
                   actually corrupted?  (ground truth from the generator)
  2. COMPLETENESS  do the attributions account for the whole score, with no
                   residual?  For the circuit this is a theorem (chain rule on
                   exact marginals); for SHAP it is an estimation target.
  3. FAITHFULNESS  if you remove what the explanation blames, does the anomaly
                   actually go away?  (deletion curve — model-agnostic, so
                   every method can be scored on it)

A method can win (1) by luck and (2) by construction, so (3) is the one that
keeps everyone honest.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .metrics import auroc


# ═══════════════════════════════════════════════════════════════════════════
# 1. Attribution methods — each returns (N, C), higher = more responsible
# ═══════════════════════════════════════════════════════════════════════════

def pc_attributions(pc, X: torch.Tensor, shapley_orders: int = 0
                    ) -> Dict[str, np.ndarray]:
    """
    Every exact view the circuit offers, from ONE trained model:

      marginal      −log p(x_c)            is this sensor odd on its own?
      conditional   −log p(x_c | x_-c)     is it odd GIVEN the others?
      structural    conditional − marginal  dependence-breaking mass only
      shapley       conditional Shapley over channels (exact terms, sampled
                    ordering) — the completeness-respecting credit assignment
    """
    td = pc.typed_scores(X)
    out = {
        "PC conditional (exact)": td["conditional"].numpy(),
        "PC marginal (exact)": td["marginal"].numpy(),
        "PC structural (exact)": td["structural"].numpy(),
    }
    if shapley_orders > 0:
        out["PC Shapley (exact conditionals)"] = \
            pc.shapley_channels(X, n_orders=shapley_orders).numpy()
    return out


class GaussianConditional:
    """
    Exact conditional attribution under a full multivariate Gaussian:
    for precision Λ, the conditional surprise of coordinate i given the rest is
    (Λx̃)_i² / (2Λ_ii) up to a constant.

    Included deliberately as a STRONG adversary.  Conditional attribution is
    tractable for a Gaussian, so the circuit's claim can never be "only PCs do
    conditionals" — it has to be "PCs do conditionals for a model class that
    actually fits the data".  If this row wins, the data was Gaussian and the
    circuit was unnecessary.
    """

    name = "Gaussian conditional (exact)"

    def __init__(self, window: int, n_channels: int, ridge: float = 1e-3):
        self.w, self.C, self.ridge = window, n_channels, ridge

    def fit(self, X: torch.Tensor) -> "GaussianConditional":
        A = np.asarray(X, dtype=np.float64)
        self.mu = A.mean(0)
        S = np.cov(A, rowvar=False) + self.ridge * np.eye(A.shape[1])
        self.Lam = np.linalg.inv(S)
        return self

    def attribute(self, X: torch.Tensor) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64) - self.mu
        G = A @ self.Lam
        contrib = (G ** 2) / (2.0 * np.diag(self.Lam))
        return contrib.reshape(len(A), self.w, self.C).sum(1)


def ae_channel_error(ae, X: torch.Tensor) -> np.ndarray:
    """Per-channel reconstruction error — the standard deep attribution."""
    with torch.no_grad():
        A = ae._reshape(X)
        return ((ae.net(A) - A) ** 2).mean(dim=2).numpy()


def sampling_shap(ae, X: torch.Tensor, background: torch.Tensor,
                  n_samples: int = 32, seed: int = 0) -> np.ndarray:
    """
    Marginal-sampling attribution on the AE score: expected change in
    reconstruction error when a channel is replaced by background draws.

    This is what KernelSHAP / permutation-SHAP actually estimate — the value
    function uses the MARGINAL distribution, because the conditional is
    unavailable for the model being explained.  Its Monte-Carlo error is
    exactly what the circuit's closed form removes.
    """
    rng = np.random.default_rng(seed)
    N, w, C = len(X), ae.w, ae.C
    Xr = X.reshape(N, w, C)
    Br = background.reshape(-1, w, C)
    with torch.no_grad():
        base = ((ae.net(ae._reshape(X)) - ae._reshape(X)) ** 2).mean(dim=(1, 2)).numpy()
    out = np.zeros((N, C))
    for c in range(C):
        acc = np.zeros(N)
        for _ in range(n_samples):
            idx = rng.integers(0, len(Br), size=N)
            pert = Xr.clone()
            pert[:, :, c] = Br[idx][:, :, c]
            flat = pert.reshape(N, -1)
            with torch.no_grad():
                e = ((ae.net(ae._reshape(flat)) - ae._reshape(flat)) ** 2
                     ).mean(dim=(1, 2)).numpy()
            acc += np.abs(e - base)
        out[:, c] = acc / n_samples
    return out


def zscore_channel(zs, X: torch.Tensor) -> np.ndarray:
    A = np.asarray(X).reshape(-1, zs.w, zs.C)
    return np.abs((A - zs.mu) / zs.sd).max(axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Metrics
# ═══════════════════════════════════════════════════════════════════════════

def localization_report(attr: np.ndarray, affected: List[List[int]],
                        kinds: List[str], keep: Sequence[str]) -> Dict[str, float]:
    """
    CLAIM 1 (correctness).  AUROC and precision@k of channel attribution
    against ground truth, over windows that have a localised truth at all.
    `organic` and `normal` windows are excluded: fleet-wide degradation is not
    a sensor fault, so there is no correct channel to point at and scoring it
    would reward noise.
    """
    scores, labels, prec = [], [], []
    for i, (a, k) in enumerate(zip(affected, kinds)):
        if not a or k not in keep:
            continue
        C = attr.shape[1]
        y = np.zeros(C)
        y[list(a)] = 1.0
        scores.append(attr[i])
        labels.append(y)
        top = np.argsort(-attr[i])[:len(a)]
        prec.append(len(set(top.tolist()) & set(a)) / len(a))
    if not scores:
        return {"auroc": float("nan"), "prec_at_k": float("nan"), "n": 0}
    return {"auroc": auroc(np.concatenate(scores), np.concatenate(labels)),
            "prec_at_k": float(np.mean(prec)), "n": len(prec)}


def completeness_error(pc, X: torch.Tensor, order=None) -> Dict[str, float]:
    """
    CLAIM 2 (completeness).  The chain-rule attributions must sum to log p(x)
    with no residual.  Reports the max and mean absolute residual in nats —
    for the circuit these are float32 round-off, not approximation error.
    """
    contrib = pc.chain_rule_attribution(X, order=order)
    with torch.no_grad():
        total = pc.pc.log_prob(pc._prep(X)).cpu()
    resid = (contrib.sum(1) - total).abs()
    return {"max_residual_nats": float(resid.max()),
            "mean_residual_nats": float(resid.mean())}


def deletion_curve(score_fn, X: torch.Tensor, attr: np.ndarray, window: int,
                   n_channels: int, reference: Optional[torch.Tensor] = None,
                   ) -> Tuple[np.ndarray, float]:
    """
    CLAIM 3 (faithfulness).  Repeatedly neutralise the highest-attributed
    channel (replace it with the training mean) and re-score.  A faithful
    explanation makes the anomaly score fall fast.

    Model-agnostic, so every method is scored on the same footing and no method
    can win by being self-consistent.  Returns (mean score after k deletions,
    normalised area under the curve — LOWER is better).
    """
    N = len(X)
    Xr = X.reshape(N, window, n_channels).clone()
    ref = (reference.reshape(-1, window, n_channels).mean(0)
           if reference is not None else torch.zeros(window, n_channels))
    order = np.argsort(-attr, axis=1)
    curve = [float(score_fn(Xr.reshape(N, -1)).mean())]
    for k in range(n_channels):
        rows = np.arange(N)
        Xr[rows, :, order[:, k]] = ref[:, order[:, k]].T
        curve.append(float(score_fn(Xr.reshape(N, -1)).mean()))
    curve = np.asarray(curve)
    # normalise by the curve's own RANGE, not by its drop from the start: an
    # explanation whose "deletions" make the score go UP has no drop, and
    # dividing by it produced a meaningless 1e11.  This form is bounded in
    # [0, 1] for every method, so a rising curve scores near 1 (bad) instead of
    # exploding.
    lo, hi = curve.min(), curve.max()
    auc = float(np.mean((curve - lo) / max(hi - lo, 1e-9)))
    return curve, auc


# ═══════════════════════════════════════════════════════════════════════════
# 3. Worked examples — the "does it make sense to a human" check
# ═══════════════════════════════════════════════════════════════════════════

def explain_window(pc, x: torch.Tensor, channel_names: Optional[List[str]] = None,
                   top: int = 3) -> Dict:
    """
    Full exact explanation of ONE window, in the form an operator would read.

    Returns the marginal / conditional / structural score per channel, the
    (t, c) surprise grid, and a plain-language verdict per flagged channel:

      "out of range"        large marginal, small structural — the sensor value
                            is simply unusual; a per-channel alarm catches this.
      "broken relationship" small marginal, large structural — every value is
                            individually plausible, but they no longer agree
                            with the other sensors.  Invisible to per-channel
                            monitoring, and the case a joint density exists for.
    """
    xb = x.reshape(1, -1)
    td = pc.typed_scores(xb)
    grid = pc.time_channel_attribution(xb)[0]
    marg = td["marginal"][0]
    cond = td["conditional"][0]
    struct = td["structural"][0]
    names = channel_names or [f"ch{c}" for c in range(len(marg))]

    with torch.no_grad():
        total = float(-pc.pc.log_prob(pc._prep(xb))[0].cpu())

    ranked = torch.argsort(cond, descending=True)[:top].tolist()
    verdicts = []
    for c in ranked:
        m, s = float(marg[c]), float(struct[c])
        kind = ("broken relationship" if s > max(1.0, 0.5 * abs(m))
                else "out of range" if m > 5.0 else "unremarkable")
        t_star = int(torch.argmax(grid[:, c]))
        verdicts.append({"channel": names[c], "idx": c, "verdict": kind,
                         "marginal": m, "structural": s,
                         "conditional": float(cond[c]), "worst_step": t_star})
    return {"score_nats": total, "marginal": marg.numpy(),
            "conditional": cond.numpy(), "structural": struct.numpy(),
            "grid": grid.numpy(), "verdicts": verdicts}


def format_explanation(exp: Dict, truth: Optional[List[int]] = None,
                       kind: str = "") -> str:
    """One window's explanation as readable text."""
    lines = [f"  anomaly score {exp['score_nats']:.1f} nats"
             + (f"   [injected: {kind}, channels {truth}]" if truth is not None else "")]
    for v in exp["verdicts"]:
        hit = "" if truth is None else ("  <- CORRECT" if v["idx"] in truth
                                        else "  <- miss")
        lines.append(
            f"    {v['channel']:>5}  {v['verdict']:<20} "
            f"marginal {v['marginal']:7.2f}  structural {v['structural']:7.2f}"
            f"  worst at t={v['worst_step']}{hit}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Plots
# ═══════════════════════════════════════════════════════════════════════════

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_case_study(exp: Dict, x: torch.Tensor, window: int, n_channels: int,
                    truth: Optional[List[int]], kind: str, path: str) -> None:
    """
    Three panels for one window: the raw signal, the exact (t, c) surprise
    heat-map, and the marginal-vs-structural split per channel.  Ground-truth
    corrupted channels are marked so the reader can check the claim by eye.
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    W = x.reshape(window, n_channels).numpy()

    axes[0].imshow(W.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    axes[0].set_title(f"window ({kind})")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("channel")

    im = axes[1].imshow(exp["grid"].T, aspect="auto", cmap="magma")
    axes[1].set_title("exact surprise  −log p(x$_{tc}$ | rest)")
    axes[1].set_xlabel("t")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    idx = np.arange(n_channels)
    axes[2].barh(idx - 0.2, exp["marginal"], height=0.4, label="marginal")
    axes[2].barh(idx + 0.2, exp["structural"], height=0.4, label="structural")
    axes[2].set_title("marginal vs structural")
    axes[2].set_ylabel("channel"); axes[2].legend(fontsize=8)
    if truth:
        for c in truth:
            for ax in (axes[0], axes[1]):
                ax.axhline(c, color="lime", lw=1.0, ls="--", alpha=0.8)
            axes[2].axhline(c, color="lime", lw=1.0, ls="--", alpha=0.8)
    fig.suptitle("green dashed = ground-truth corrupted channel", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_localization(summary: Dict[str, Dict[str, float]], path: str) -> None:
    """Ranked bar chart of localisation AUROC with seed error bars."""
    plt = _mpl()
    names = sorted(summary, key=lambda n: summary[n]["auroc"])
    vals = [summary[n]["auroc"] for n in names]
    errs = [summary[n].get("auroc_sd", 0.0) for n in names]
    colors = ["#2a6fb0" if n.startswith("PC") else "#b0632a" for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.6))
    ax.barh(names, vals, xerr=errs, color=colors)
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="chance")
    ax.set_xlabel("channel-localisation AUROC vs ground truth")
    ax.set_xlim(0.4, 1.0); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_deletion_curves(curves: Dict[str, np.ndarray], path: str) -> None:
    """Faithfulness: score as the most-blamed channels are neutralised."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, c in curves.items():
        ax.plot(np.arange(len(c)), c, marker="o", ms=3,
                lw=2 if name.startswith("PC") else 1.2,
                ls="-" if name.startswith("PC") else "--", label=name)
    ax.set_xlabel("channels neutralised (highest attribution first)")
    ax.set_ylabel("mean anomaly score")
    ax.set_title("deletion curve — steeper is more faithful")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
