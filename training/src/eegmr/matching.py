"""Matching variants: how a measured EEG trial is compared with a predicted one.

Every statistic (mean response, reliability weights, noise covariances) is estimated from the subject's TRAINING
data only. The reliability-aware variants additionally need out-of-fold (OOF) predictions of the training trials, so
that "prediction error" is measured on trials the model did not see.

    erp_subtract   remove the training-mean response from measured and predicted EEG
    weighted       weight each (channel, time) by max(r, 0)^p, r = corr(measured, OOF prediction) over training trials
    noise          whiten the spatial noise (OOF residuals, Ledoit-Wolf covariance), then correlation or -distance^2

The OOF predictions and the weighted / noise statistics run in torch (CUDA if available, else CPU). The plain
correlation paths (``kind="corr"``) use :func:`eegmr.linear.correlation_similarity` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, Literal

import numpy as np
import torch
from sklearn.model_selection import KFold

from eegmr import gpu
from eegmr.linear import DEFAULT_WINDOW, correlation_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _t(x: np.ndarray | torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, device=device).to(dtype)


def fit_predict(features_trn: np.ndarray, eeg_trn: np.ndarray, features_new: np.ndarray, alpha: float,
                pca_variance: float = 0.95, whiten: bool = False, device: str = DEVICE) -> np.ndarray:
    """Standardize -> PCA(``pca_variance``) -> (standardize if ``whiten``) -> ridge(``alpha``); returns the predicted
    EEG ``(n_new, channels, time)``. Uses the shared torch ridge in :mod:`eegmr.gpu` (same as the sklearn pipeline in
    ``eegmr.linear``)."""
    basis = gpu.ridge_basis(features_trn, pca_variance, whiten, device)
    pred = gpu.ridge_predict(basis, eeg_trn.reshape(len(eeg_trn), -1), features_new, np.array([alpha]))[0]
    return pred.reshape(len(features_new), *eeg_trn.shape[1:]).cpu().numpy()


def oof_predictions(features: np.ndarray, eeg: np.ndarray, alpha: float, pca_variance: float = 0.95,
                    whiten: bool = False, n_folds: int = 5, seed: int = 0, device: str = DEVICE) -> np.ndarray:
    """Out-of-fold prediction of every training trial. Same folds as ``retrieval_cv_curve``; the feature transform
    is refit on each fold's training part; ``alpha`` is fixed."""
    out = np.zeros_like(eeg)
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        out[val] = fit_predict(features[trn], eeg[trn], features[val], alpha, pca_variance, whiten, device)
    return out


def _columnwise_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation over dim 0 for every remaining index; 0 where either input is constant."""
    a, b = a - a.mean(0), b - b.mean(0)
    denom = ((a**2).sum(0) * (b**2).sum(0)).sqrt()
    return torch.where(denom > 0, (a * b).sum(0) / denom.clamp_min(1e-300), torch.zeros_like(denom))


def ledoit_wolf_covariance(X: torch.Tensor) -> torch.Tensor:
    """Ledoit-Wolf shrunk covariance of a batch ``X`` of shape ``(batch, n, p)`` (samples centred per batch item).

    Closed-form shrinkage towards ``mu * I``, identical to ``sklearn.covariance.LedoitWolf`` (its
    ``ledoit_wolf_shrinkage`` formula)."""
    _, n, p = X.shape
    X = X - X.mean(1, keepdim=True)
    emp = X.transpose(1, 2) @ X / n
    trace = emp.diagonal(dim1=1, dim2=2).sum(-1)
    mu = trace / p
    delta_ = (emp**2).sum((1, 2))
    X2 = X**2
    beta_ = ((X2.transpose(1, 2) @ X2).sum((1, 2)) / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace + p * mu**2) / p
    beta = torch.minimum(beta_, delta)
    shrink = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)[:, None, None]
    return (1 - shrink) * emp + shrink * mu[:, None, None] * torch.eye(p, dtype=X.dtype, device=X.device)


def _inv_sqrt(cov: torch.Tensor) -> torch.Tensor:
    """Symmetric inverse square root of a batch of positive-definite matrices ``(batch, p, p)``."""
    vals, vecs = torch.linalg.eigh(cov)
    return (vecs / vals.clamp_min(1e-12).sqrt()[:, None, :]) @ vecs.transpose(1, 2)


@dataclass
class Matcher:
    """Similarity between measured and predicted EEG, both ``(n, channels, time)``; returns ``(n_measured, n_pred)``.

    ``kind="corr"`` is exactly :func:`correlation_similarity` (on EEG minus the mean response if ``erp_subtract``).
    Call :meth:`fit` with the subject's training EEG (and, for ``weighted`` / ``noise``, the OOF predictions of
    those trials) before use. ``fit`` accepts a ``cache`` dict to share statistics between matchers fitted on the
    same data (the correlation map for all powers, the covariances for both similarity measures).
    """

    kind: Literal["corr", "weighted", "noise"] = "corr"
    erp_subtract: bool = False
    power: float = 1.0  # weighted: w = max(r, 0) ** power
    noise_cov: Literal["per_time", "pooled"] = "per_time"  # noise: one covariance per timepoint, or one for all
    noise_similarity: Literal["corr", "euclid"] = "corr"  # noise: correlation or negative squared distance
    window: slice = field(default_factory=lambda: DEFAULT_WINDOW)
    device: str = DEVICE
    erp_: np.ndarray | None = None  # (channels, time) mean training response
    weights_: torch.Tensor | None = None  # (channels, window time)
    whiteners_: torch.Tensor | None = None  # (window time or 1, channels, channels)

    def fit(self, eeg_trn: np.ndarray, oof_pred: np.ndarray | None = None, cache: dict | None = None) -> "Matcher":
        """``eeg_trn``, ``oof_pred``: ``(n_train, channels, time)`` (same crop / decimation as the test data)."""
        self.erp_ = eeg_trn.mean(axis=0) if self.erp_subtract else None
        if self.kind == "corr":
            return self
        assert oof_pred is not None and oof_pred.shape == eeg_trn.shape, f"{self.kind} matching needs OOF predictions"
        cache = {} if cache is None else cache
        key = "r" if self.kind == "weighted" else ("cov", self.noise_cov)
        if key not in cache:
            Y = _t(eeg_trn[..., self.window], self.device, torch.float64)
            P = _t(oof_pred[..., self.window], self.device, torch.float64)
            if self.kind == "weighted":
                cache[key] = _columnwise_corr(Y, P)
            else:
                E = (Y - P).permute(2, 0, 1)  # (time, n, channels): residuals, one batch item per timepoint
                if self.noise_cov == "pooled":
                    E = E.reshape(1, -1, E.shape[-1])
                cache[key] = _inv_sqrt(ledoit_wolf_covariance(E))
        if self.kind == "weighted":
            self.weights_ = cache[key].clamp_min(0.0) ** self.power
        else:
            self.whiteners_ = cache[key]
        return self

    def _prepare(self, x: np.ndarray) -> torch.Tensor:
        """Subtract the mean response, crop to the window and apply the weights / whitening (float64)."""
        if self.erp_ is not None:
            x = x - self.erp_
        x = _t(x[..., self.window], self.device, torch.float64)
        if self.kind == "weighted":
            return x * self.weights_
        W = self.whiteners_.expand(x.shape[-1], -1, -1)
        return torch.einsum("tcd,ndt->nct", W, x)

    def similarity(self, measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        if self.kind == "corr":  # subtraction commutes with windowing; keep the original code path
            if self.erp_ is None:
                return correlation_similarity(measured, predicted, self.window)
            return correlation_similarity(measured - self.erp_, predicted - self.erp_, self.window)
        m, p = self._prepare(measured), self._prepare(predicted)
        m, p = m.reshape(len(m), -1), p.reshape(len(p), -1)
        if self.kind == "noise" and self.noise_similarity == "euclid":
            return -torch.cdist(m, p).pow(2).cpu().numpy()
        m, p = m - m.mean(1, keepdim=True), p - p.mean(1, keepdim=True)
        return ((m / m.norm(dim=1, keepdim=True)) @ (p / p.norm(dim=1, keepdim=True)).T).cpu().numpy()


def nested_cv_scores(features: np.ndarray, eeg: np.ndarray, alpha: float, matchers: dict[str, Matcher],
                     pca_variance: float = 0.95, whiten: bool = False, n_folds: int = 5, seed: int = 0,
                     device: str = DEVICE) -> dict[str, float]:
    """Training-only score of each matcher: mean percentile rank (0 = best, 0.5 = chance) of the true video among the
    held-out videos of the outer folds (same folds as ``retrieval_cv_curve``).

    The matcher's parameters (mean response, weights, covariances) are estimated inside each outer fold's training
    part, from OOF predictions made by an inner ``n_folds``-fold CV on that part, so nothing is estimated from the
    trials being scored. ``alpha`` is fixed.
    """
    totals = dict.fromkeys(matchers, 0.0)
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        pred_val = fit_predict(features[trn], eeg[trn], features[val], alpha, pca_variance, whiten, device)
        need_oof = any(m.kind != "corr" for m in matchers.values())
        oof = (oof_predictions(features[trn], eeg[trn], alpha, pca_variance, whiten, n_folds, seed, device)
               if need_oof else None)
        cache: dict = {}
        for name, matcher in matchers.items():
            sim = replace(matcher, device=device).fit(eeg[trn], oof, cache).similarity(eeg[val], pred_val)
            ranks = (sim >= np.diag(sim)[:, None]).sum(axis=1)
            totals[name] += np.sum((ranks - 1) / (len(val) - 1))
    return {name: total / len(features) for name, total in totals.items()}


def nested_fold_similarities(features: np.ndarray, eeg: np.ndarray, alpha: float, matcher: Matcher,
                             pca_variance: float = 0.95, whiten: bool = False, n_folds: int = 5, seed: int = 0,
                             device: str = DEVICE) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(val_indices, similarity)`` for each outer fold: held-out recordings x held-out predictions, matched
    with ``matcher`` whose parameters come from the fold's training part only (as in :func:`nested_cv_scores`)."""
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        pred_val = fit_predict(features[trn], eeg[trn], features[val], alpha, pca_variance, whiten, device)
        oof = (oof_predictions(features[trn], eeg[trn], alpha, pca_variance, whiten, n_folds, seed, device)
               if matcher.kind != "corr" else None)
        yield val, replace(matcher, device=device).fit(eeg[trn], oof).similarity(eeg[val], pred_val)
