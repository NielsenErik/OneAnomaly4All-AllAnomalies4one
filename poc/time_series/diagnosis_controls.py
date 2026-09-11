"""Known-law relational control with independent windows and exact oracle.

The nominal law is Gaussian with separable temporal/channel covariance.
Replacing one complete channel with an independent draw preserves that
channel's trajectory marginal exactly in distribution. This is a mechanism
control, not evidence about physical sensor failures.
"""
import torch

from .data import ADTask


def control_covariance(window, channels, temporal=.6, cross=.8):
    if window < 1 or channels < 2 or not 0 <= temporal < 1 or not 0 <= cross < 1:
        raise ValueError("positive window, >=2 channels and correlations in [0,1) required")
    t = torch.arange(window)
    time = temporal ** (t[:, None] - t[None, :]).abs().double()
    channel = (1 - cross) * torch.eye(channels, dtype=torch.double) + cross
    return torch.kron(time, channel)


def make_control_task(window=4, channels=4, n_train=512, n_val=300, seed=0,
                      temporal=.6, cross=.8):
    covariance = control_covariance(window, channels, temporal, cross)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randn(n_train + n_val, window * channels, dtype=torch.double,
                          generator=generator) @ torch.linalg.cholesky(covariance).T
    X = samples.float()
    task = ADTask(X_train=X[:n_train], X_val=X[n_train:],
        X_test=torch.empty(0, window * channels), y_test=torch.empty(0),
        kind_test=[], affected_test=[], window=window, n_channels=channels,
        channel_groups=[list(range(channels))],
        unit_train=torch.arange(n_train), unit_val=torch.arange(n_train, n_train + n_val),
        meta={"control": "known_gaussian", "independent_unit": "window"})
    return task, covariance


def independent_replacement(X, covariance, channels, seed=0, target=0):
    """Independent donor from the true nominal law, not a finite donor pool."""
    generator = torch.Generator().manual_seed(seed)
    donor = torch.randn(X.shape, generator=generator, dtype=torch.double) @ torch.linalg.cholesky(covariance).T
    result = X.clone()
    result[:, target::channels] = donor[:, target::channels].to(X.dtype)
    return result


def gaussian_oracle_map(X, covariance, channels, mask=None):
    """Analytic reference law, using exact covariance submatrices.

    This intentionally simple oracle is for statistical correctness, not a
    competitive latency implementation of all Gaussian conditionals.
    """
    x = X.double()
    d = x.shape[1]
    mask = torch.ones(channels, dtype=torch.bool) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
    if mask.shape != (channels,):
        raise ValueError("oracle expects one channel mask")

    def lp(features):
        if not features:
            return torch.zeros(len(x), dtype=x.dtype)
        cov = covariance[features][:, features]
        law = torch.distributions.MultivariateNormal(torch.zeros(len(features), dtype=x.dtype), covariance_matrix=cov)
        return law.log_prob(x[:, features])

    observed = [i for i in range(d) if mask[i % channels]]
    joint = lp(observed)
    marginal = torch.full((len(x), channels), float("nan"), dtype=x.dtype)
    conditional = marginal.clone()
    for c in range(channels):
        if mask[c]:
            marginal[:, c] = -lp(list(range(c, d, channels)))
            conditional[:, c] = lp([i for i in observed if i % channels != c]) - joint
    return dict(log_px=joint, marginal=marginal, conditional=conditional,
                R=conditional - marginal, structural=conditional - marginal,
                mask=mask.expand(len(x), -1))


# ═══════════════════════════════════════════════════════════════════════════
# Known dependency graph with designated witnesses (step 6)
# ═══════════════════════════════════════════════════════════════════════════
#
# The availability question is not "does the score drop when sensors go
# missing" — it is whether the evidence the statistic needs was ever there.
# A control that answers it has to make the answer KNOWN in advance, so the
# construction below fixes WHICH channels carry the target's dependence:
#
#     target_c0  = α·f            + ε
#     witness_w  = β·f  + γ·g     + ε
#     other_o    =        δ·g     + ε
#
# with f ⟂ g.  Cov(target, other) = 0 exactly, so every path from the target
# to the rest of the fleet runs through a witness.  Observing the witnesses
# leaves the target identifiable; removing all of them leaves it EXACTLY
# independent of what remains, and the correct relational score is then zero —
# not small, zero.  A method that still reports a signal there is reporting
# numerical noise, and a method that abstains is behaving correctly.

def witness_covariance(window, channels, witnesses=(1,), target=0, temporal=.6,
                       alpha=.9, beta=.7, gamma=.5, delta=.8, noise=.35):
    """Separable temporal × channel covariance whose channel graph is known.

    Returns the (window·channels, window·channels) covariance in the flattened
    `t * C + c` layout, so it drops straight into `gaussian_oracle_map`.
    """
    witnesses = sorted({int(w) for w in witnesses})
    target = int(target)
    if channels < 3 or not witnesses or target in witnesses:
        raise ValueError("need >=3 channels and a nonempty witness set excluding the target")
    if any(not 0 <= w < channels for w in witnesses) or not 0 <= target < channels:
        raise ValueError("witness and target indices must be channels")
    if noise <= 0 or not 0 <= temporal < 1:
        raise ValueError("positive noise and temporal correlation in [0, 1) required")
    loadings = torch.zeros(channels, 2, dtype=torch.double)
    for c in range(channels):
        if c == target:
            loadings[c, 0] = alpha
        elif c in witnesses:
            loadings[c, 0], loadings[c, 1] = beta, gamma
        else:
            loadings[c, 1] = delta
    channel_cov = loadings @ loadings.T + noise * torch.eye(channels, dtype=torch.double)
    scale = torch.sqrt(torch.diagonal(channel_cov))
    channel_cov = channel_cov / scale[:, None] / scale[None, :]     # unit variances
    t = torch.arange(window)
    time = temporal ** (t[:, None] - t[None, :]).abs().double()
    return torch.kron(time, channel_cov)


def witness_task(window=4, channels=4, witnesses=(1,), target=0, n_train=512,
                 n_val=300, seed=0, **kwargs):
    """`make_control_task` on the witness law, carrying the graph in `meta`."""
    covariance = witness_covariance(window, channels, witnesses, target, **kwargs)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randn(n_train + n_val, window * channels, dtype=torch.double,
                          generator=generator) @ torch.linalg.cholesky(covariance).T
    X = samples.float()
    task = ADTask(X_train=X[:n_train], X_val=X[n_train:],
        X_test=torch.empty(0, window * channels), y_test=torch.empty(0),
        kind_test=[], affected_test=[], window=window, n_channels=channels,
        channel_groups=[list(range(channels))],
        unit_train=torch.arange(n_train), unit_val=torch.arange(n_train, n_train + n_val),
        meta={"control": "known_witness_graph", "independent_unit": "window",
              "target": int(target), "witnesses": sorted(int(w) for w in witnesses)})
    return task, covariance


def witness_masks(n_channels, target=0, witnesses=(1,)):
    """Progressive witness removal, plus the two degenerate references.

    Named so a results table says which regime produced a row:
      `witnesses_all`     every witness present — the identifiable case
      `witnesses_k`       k witnesses removed
      `witnesses_none`    no witness present — the target is exactly
                          independent of the rest and zero is the right answer
      `target_hidden`     the target is not observed at all — every remaining
                          query must be numerically unchanged by its value
    """
    witnesses = sorted({int(w) for w in witnesses})
    target = int(target)
    out = {}
    for k in range(len(witnesses) + 1):
        mask = torch.ones(n_channels, dtype=torch.bool)
        mask[witnesses[:k]] = False
        name = ("witnesses_all" if k == 0 else
                "witnesses_none" if k == len(witnesses) else f"witnesses_drop{k}")
        out[name] = mask
    hidden = torch.ones(n_channels, dtype=torch.bool)
    hidden[target] = False
    out["target_hidden"] = hidden
    return out


def witness_availability(mask, target=0, witnesses=(1,)):
    """The three availability facts every step-6 row has to carry."""
    mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    present = [int(w) for w in witnesses if bool(mask[int(w)])]
    return {"target_visible": bool(mask[int(target)]),
            "witnesses_available": len(present),
            "witnesses_present": present,
            "identifiable": bool(mask[int(target)]) and bool(present)}
