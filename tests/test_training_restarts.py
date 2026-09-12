"""Contracts for restart selection and the learning-rate schedules.

The thing being protected here is not speed, it is what the selection is
allowed to see.  Restarts exist because initialisation dominated architecture
on FD001 (five seeds of one architecture: checkpoint NLL 20.2-59.1, and the
detection advantage tracked it), and the only defensible way to spend them is
to keep the attempt with the best CHECKPOINT loss — the quantity early
stopping already uses, computed on engines the evaluation never touches.
"""
import numpy as np
import pytest
import torch

from poc.time_series.circuits import WindowPC


def data(n=64, window=2, channels=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, window * channels, generator=g)
    return x[: n // 2], x[n // 2:]


def fitted(restarts=1, epochs=6, seed=5, **kw):
    X, Xv = data()
    pc = WindowPC(2, 3, vtree_method="channel_blocked", n_sum_components=2,
                  device="cpu", seed=seed)
    pc.fit(X, epochs=epochs, lr=.05, batch_size=16, X_val=Xv,
           restarts=restarts, **kw)
    return pc


def test_restarts_need_checkpoint_data_and_valid_settings():
    X, Xv = data()
    pc = WindowPC(2, 3, vtree_method="channel_blocked", n_sum_components=2,
                  device="cpu", seed=1)
    with pytest.raises(ValueError, match="checkpoint data"):
        pc.fit(X, epochs=2, batch_size=16, restarts=3)
    with pytest.raises(ValueError, match="positive integer"):
        pc.fit(X, epochs=2, batch_size=16, X_val=Xv, restarts=0)
    with pytest.raises(ValueError, match="lr_schedule"):
        pc.fit(X, epochs=2, batch_size=16, X_val=Xv, lr_schedule="exponential")
    with pytest.raises(ValueError, match="checkpoint loss"):
        pc.fit(X, epochs=2, batch_size=16, lr_schedule="plateau")


def test_selected_restart_is_the_best_checkpoint_loss_and_beats_one_restart():
    many = fitted(restarts=3)
    losses = many.restart_selection_losses
    assert len(losses) == 3 and len(set(losses)) > 1      # the attempts differ
    assert many.selected_restart == int(np.argmin(losses))
    assert many.best_selection_loss == pytest.approx(min(losses))
    # Restart 0 IS the single-restart run: same init seed, same stream. So more
    # restarts can never select a worse fit than one restart would have.
    one = fitted(restarts=1)
    assert one.best_selection_loss == pytest.approx(losses[0])
    assert many.best_selection_loss <= one.best_selection_loss


def test_restarts_vary_initialisation_only_not_structure():
    one, many = fitted(restarts=1), fitted(restarts=3)
    assert one.size()["parameters"] == many.size()["parameters"]
    seeds = [t["init_seed"] for t in many.restart_traces]
    assert len(set(seeds)) == 3 and seeds[0] == many.seed


def test_restart_fits_are_reproducible_and_cost_is_reported_in_full():
    a, b = fitted(restarts=2), fitted(restarts=2)
    assert a.restart_selection_losses == pytest.approx(b.restart_selection_losses)
    assert a.optimizer_steps == sum(t["optimizer_steps"] for t in a.restart_traces)
    assert a.selected_optimizer_steps <= a.optimizer_steps


def test_selected_parameters_are_the_selected_restarts_parameters():
    pc = fitted(restarts=3)
    X, Xv = data()
    with torch.no_grad():
        nll = float(-pc.pc.log_prob(Xv).mean())
    # The winner's rolled-back checkpoint, not whichever attempt ran last.
    assert nll == pytest.approx(pc.best_val_nll, abs=1e-4)
    pc.pc.validate()
    assert abs(float(pc.pc.log_partition().detach())) < 1e-4


def test_abandoned_restarts_are_never_selected():
    pc = fitted(restarts=3, epochs=8, restart_abandon_margin=1e-6)
    assert any(t["abandoned"] for t in pc.restart_traces)
    assert not pc.restart_traces[pc.selected_restart]["abandoned"]


def test_cosine_schedule_decays_the_step_size_to_the_floor():
    pc = fitted(epochs=10, lr_schedule="cosine", lr_min_factor=.1)
    final = pc.restart_traces[0]["final_lr"]
    assert .05 * .1 <= final < .05
    flat = fitted(epochs=10)
    assert flat.restart_traces[0]["final_lr"] == pytest.approx(.05)
