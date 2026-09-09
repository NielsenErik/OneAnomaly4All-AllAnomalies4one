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
