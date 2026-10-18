"""Checks for ``eegmr.contrastive``: ridge warm start, expected top-k, and a smoke test of training."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge

from eegmr.contrastive import ContrastiveConfig, expected_topk, mean_percentile_rank, ridge_weights, train_head

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def data() -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(200, 6))
    X = latent @ rng.normal(size=(6, 40)) + rng.normal(size=(200, 40))
    V = latent @ rng.normal(size=(6, 16)) + 0.3 * rng.normal(size=(200, 16))
    X, V = (X - X.mean(0)) / X.std(0), (V - V.mean(0)) / V.std(0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(V, dtype=torch.float32)


def test_ridge_weights_match_sklearn(data):
    X, V = data
    W = ridge_weights(X, V, 10.0).numpy()
    ref = Ridge(alpha=10.0, fit_intercept=False).fit(X.numpy().astype(np.float64), V.numpy()).coef_.T
    assert np.abs(W - ref).max() < 1e-5


def test_expected_topk_limits():
    assert expected_topk(np.array([1]), n_pool=190, k=1) == pytest.approx(1.0)
    assert expected_topk(np.array([190]), n_pool=190, k=5) == pytest.approx(0.0)
    # true video ranked 2nd of 190: top-1 of 102 iff the one better video is not drawn
    assert expected_topk(np.array([2]), n_pool=190, k=1) == pytest.approx(1 - 101 / 189)
    assert expected_topk(np.array([102]), n_pool=102, k=1) == pytest.approx(0.0)  # pool == candidate set


@pytest.mark.parametrize("device", DEVICES)
def test_warm_start_without_steps_is_ridge(data, device):
    X, V = (t.to(device) for t in data)
    head = train_head(X, V, ContrastiveConfig("warm", lr=0.0, max_steps=0), ridge_alpha=10.0)
    W = ridge_weights(X, V, 10.0).float()
    assert head.steps == 0 and torch.allclose(head.W_e, W, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["linear", "two_tower"])
def test_training_beats_chance(data, device, kind):
    X, V = (t.to(device) for t in data)
    head = train_head(X[:160], V[:160], ContrastiveConfig(kind, dim=8, lr=1e-2, max_steps=300), ridge_alpha=10.0)
    assert mean_percentile_rank(head.similarity(X[160:], V[160:])) < 0.35  # chance = 0.5
    assert 0.01 <= float(head.log_tau.exp()) <= 1.0
