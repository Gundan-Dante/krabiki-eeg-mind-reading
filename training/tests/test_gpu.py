"""Parity of ``eegmr.gpu`` (torch) with the sklearn/numpy reference implementations."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from eegmr import gpu
from eegmr.linear import _feature_transform, correlation_similarity

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ALPHAS = np.array([1.0, 30.0, 1e3, 1e5])


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(240, 8))
    features = latent @ rng.normal(size=(8, 60)) + 0.5 * rng.normal(size=(240, 60))  # decaying spectrum
    eeg = (latent @ rng.normal(size=(8, 3 * 20))).reshape(240, 3, 20) + rng.normal(size=(240, 3, 20))
    return features, eeg


def numpy_cv_curve(features, eeg, alphas, window, pca, whiten, erp_subtract=False):
    """The original numpy/sklearn retrieval CV, kept here as the reference."""
    scores = np.zeros(len(alphas))
    eeg = eeg[..., window]
    for trn, val in KFold(5, shuffle=True, random_state=0).split(features):
        t = _feature_transform(pca, whiten).fit(features[trn])
        Zt, Zv = t.transform(features[trn]), t.transform(features[val])
        Y = eeg[trn].reshape(len(trn), -1)
        ym = Y.mean(0)
        U, s, Vt = np.linalg.svd(Zt, full_matrices=False)
        UtY, ZV = U.T @ (Y - ym), Zv @ Vt.T
        erp = ym.reshape(eeg.shape[1:]) if erp_subtract else 0.0
        for k, a in enumerate(alphas):
            p = (ZV * (s / (s**2 + a))) @ UtY + ym
            sim = correlation_similarity(eeg[val] - erp, p.reshape(len(val), *eeg.shape[1:]) - erp, slice(None))
            scores[k] += np.sum(((sim >= np.diag(sim)[:, None]).sum(1) - 1) / (len(val) - 1))
    return scores / len(features)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pca,whiten", [(0.95, False), (0.8, False), (None, False), (0.95, True)])
def test_ridge_predict_matches_sklearn(data, device, pca, whiten):
    features, eeg = data
    Y = eeg.reshape(len(eeg), -1)
    basis = gpu.ridge_basis(features[:200], pca, whiten, device)
    pred = gpu.ridge_predict(basis, Y[:200], features[200:], ALPHAS).cpu().numpy()
    for i, alpha in enumerate(ALPHAS):
        ref_pipe = _feature_transform(pca, whiten)
        Z = ref_pipe.fit_transform(features[:200])
        ref = Ridge(alpha=alpha).fit(Z, Y[:200]).predict(ref_pipe.transform(features[200:]))
        assert basis.U.shape[1] == Z.shape[1], "component count differs from sklearn PCA"
        assert np.abs(pred[i] - ref).max() < 1e-4 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("device", DEVICES)
def test_correlation_similarity_matches_numpy(data, device):
    _, eeg = data
    ref = correlation_similarity(eeg[:50], eeg[50:120], slice(2, 15))
    out = gpu.correlation_similarity(eeg[:50], eeg[50:120], slice(2, 15), device=device).cpu().numpy()
    assert np.abs(out - ref).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pca,whiten", [(0.95, False), (None, False), (0.95, True)])
def test_retrieval_cv_matches_numpy(data, device, pca, whiten):
    features, eeg = data
    windows = [slice(2, 15), slice(0, 20)]
    out = gpu.retrieval_cv_scores(features, eeg, ALPHAS, windows, pca_variance=pca, whiten=whiten, device=device)
    for i, w in enumerate(windows):
        ref = numpy_cv_curve(features, eeg, ALPHAS, w, pca, whiten)
        assert np.abs(out[i] - ref).max() < 2e-3  # float32 can flip near-ties in a few ranks
        if not whiten:  # whitened curves are flat, so their argmin is arbitrary
            assert out[i].argmin() == ref.argmin()


@pytest.mark.parametrize("device", DEVICES)
def test_retrieval_cv_erp_subtract_matches_numpy(data, device):
    features, eeg = data
    out = gpu.retrieval_cv_scores(features, eeg, ALPHAS, [slice(2, 15)], erp_subtract=True, device=device)[0]
    ref = numpy_cv_curve(features, eeg, ALPHAS, slice(2, 15), 0.95, False, erp_subtract=True)
    assert np.abs(out - ref).max() < 2e-3 and out.argmin() == ref.argmin()


@pytest.mark.parametrize("device", DEVICES)
def test_matching_fit_predict_matches_sklearn(data, device):
    from eegmr.matching import fit_predict

    features, eeg = data
    pred = fit_predict(features[:200], eeg[:200], features[200:], 1e3, 0.95, False, device)
    pipe = _feature_transform(0.95)
    ref = Ridge(alpha=1e3).fit(pipe.fit_transform(features[:200]), eeg[:200].reshape(200, -1))
    ref = ref.predict(pipe.transform(features[200:])).reshape(pred.shape)
    assert np.abs(pred - ref).max() < 1e-4 * max(1.0, np.abs(ref).max())
