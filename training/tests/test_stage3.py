"""Checks for ``eegmr.stage3`` (CPU only, small synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eegmr.contrastive import mean_percentile_rank
from eegmr.stage3 import N_CHANS, Encoder, Stage3Config, augment, train_encoder


@pytest.mark.parametrize("model", ["linear", "mlp", "resmlp", "eegnet", "shallow"])
def test_encoder_output_is_unit_norm(model):
    cfg = Stage3Config(model, hidden=32, window=(0, 60))
    enc = Encoder(cfg, n_times=60, out_dim=16).eval()
    out = enc(torch.randn(5, N_CHANS, 60))
    assert out.shape == (5, 16)
    assert torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5)


def test_augment_preserves_shape_and_is_seeded():
    x = torch.randn(8, N_CHANS, 40)
    cfg = Stage3Config(augment=True)
    a = augment(x, cfg, torch.Generator().manual_seed(0))
    b = augment(x, cfg, torch.Generator().manual_seed(0))
    assert a.shape == x.shape and torch.equal(a, b) and not torch.equal(a, x)


def test_shift_only_is_a_circular_roll():
    x = torch.arange(N_CHANS * 10, dtype=torch.float32).view(1, N_CHANS, 10)
    cfg = Stage3Config(augment=True, shift=2, chan_drop=0.0, noise=0.0)
    y = augment(x, cfg, torch.Generator().manual_seed(3))
    assert any(torch.equal(y, torch.roll(x, s, dims=2)) for s in range(-2, 3))


@pytest.mark.parametrize("model", ["linear", "mlp"])
def test_training_beats_chance(model):
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(200, 6))
    X = torch.tensor((latent @ rng.normal(size=(6, N_CHANS * 10))).reshape(200, N_CHANS, 10)
                     + rng.normal(size=(200, N_CHANS, 10)), dtype=torch.float32)
    V = torch.tensor(latent @ rng.normal(size=(6, 16)) + 0.3 * rng.normal(size=(200, 16)), dtype=torch.float32)
    cfg = Stage3Config(model, hidden=32, dropout=0.1, lr=1e-2, window=(0, 10), max_steps=300)
    enc = train_encoder(X[:160], V[:160], cfg)
    assert mean_percentile_rank(enc.similarity(X[160:], V[160:])) < 0.35  # chance = 0.5
