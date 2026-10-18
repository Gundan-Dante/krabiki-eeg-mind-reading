"""Decoding-direction variants: noise-whitened EEG input and reliability-weighted cosine.

The decoder maps the EEG window (channels x time, flattened, standardized) to standardized stimulus features with
ridge, and ranks candidates by cosine similarity (``eegmr.linear.DecodingModel`` is the plain version). Two
optional changes, both estimated from the subject's TRAINING data only:

    whiten_input        multiply each timepoint's channel vector by Sigma_t^-1/2, where Sigma_t is the Ledoit-Wolf
                        covariance of the encoding model's out-of-fold residuals (as in ``eegmr.matching``)
    reliability_power   scale each feature dimension by max(r_d, 0)^p before the cosine, where r_d is the
                        correlation between out-of-fold decoded and true values of dimension d on training trials
                        (same convention as ``matching.Matcher(kind="weighted")``)

:func:`nested_cv_scores` scores every combination without touching the trials being scored: the whiteners and the
reliability weights are re-estimated inside each outer fold's training part from an inner CV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.model_selection import KFold

from eegmr import gpu
from eegmr.linear import DEFAULT_ALPHAS, DEFAULT_WINDOW
from eegmr.matching import _inv_sqrt, ledoit_wolf_covariance, oof_predictions


def noise_whiteners(features: np.ndarray, eeg: np.ndarray, enc_alpha: float, window: slice,
                    pca_variance: float = 0.95, n_folds: int = 5, seed: int = 0,
                    device: str | None = None) -> torch.Tensor:
    """Per-timepoint Sigma^-1/2 ``(window time, C, C)`` from the encoding model's OOF residuals on these trials."""
    device = device or gpu.default_device()
    oof = oof_predictions(features, eeg, enc_alpha, pca_variance, n_folds=n_folds, seed=seed, device=device)
    resid = gpu.as_tensor(eeg[..., window] - oof[..., window], device, torch.float64).permute(2, 0, 1)
    return _inv_sqrt(ledoit_wolf_covariance(resid))


def decoder_input(eeg: np.ndarray | torch.Tensor, window: slice, whiteners: torch.Tensor | None = None,
                  device: str | None = None) -> torch.Tensor:
    """The flattened EEG window ``(n, C * T_w)`` (float64), optionally noise-whitened per timepoint."""
    x = gpu.as_tensor(eeg, device, torch.float64)[..., window]
    if whiteners is not None:
        x = torch.einsum("tcd,ndt->nct", whiteners.to(x.device), x)
    return x.flatten(1)


def reliability(X: torch.Tensor, targets: torch.Tensor, alphas: np.ndarray, n_folds: int = 5,
                seed: int = 0) -> torch.Tensor:
    """Per-dimension correlation ``(alphas, D)`` between OOF-decoded and true targets on these trials."""
    pred = torch.zeros(len(alphas), *targets.shape, dtype=torch.float64, device=X.device)
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(len(X))):
        basis = gpu.ridge_basis(X[trn], pca_variance=None, device=X.device)
        pred[:, val] = gpu.ridge_predict(basis, gpu.standardize(targets[trn], targets[trn]), X[val], alphas).double()
    a, b = pred - pred.mean(1, keepdim=True), targets - targets.mean(0)
    return (a * b).sum(1) / ((a**2).sum(1) * (b**2).sum(0)).sqrt().clamp_min(1e-300)


def _percentile_ranks(sim: torch.Tensor) -> torch.Tensor:
    """Sum over rows of (rank of the diagonal - 1) / (n - 1); ``sim``: ``(..., n, n)``."""
    ranks = (sim >= sim.diagonal(dim1=-2, dim2=-1)[..., None]).sum(dim=-1)
    return ((ranks - 1) / (sim.shape[-1] - 1)).sum(dim=-1).double()


def nested_cv_scores(
    features: np.ndarray,
    eeg: np.ndarray,
    alphas: np.ndarray,
    enc_alpha: float,
    powers: tuple[float, ...] = (0.5, 1.0, 2.0),
    window: slice = DEFAULT_WINDOW,
    inputs: tuple[str, ...] = ("raw", "whitened"),
    pca_variance: float = 0.95,
    n_folds: int = 5,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Training-only retrieval scores (mean percentile rank, 0.5 = chance) for every decoder variant.

    Returns a curve over ``alphas`` per variant: ``"<input>"`` (plain cosine) and ``"<input>_p<power>"`` (weighted
    cosine, weights estimated at the same alpha), with ``<input>`` in {raw, whitened}. ``enc_alpha`` is the encoding
    model's alpha (for the OOF residuals). Same outer folds as every other retrieval CV in the package.
    """
    device = device or gpu.default_device()
    F_all = gpu.as_tensor(features, device, torch.float64)
    totals: dict[str, torch.Tensor] = {}
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        F_trn, F_val = gpu.standardize(F_all[trn], F_all[trn]), gpu.standardize(F_all[val], F_all[trn])
        whiteners = {"raw": lambda: None,
                     "whitened": lambda: noise_whiteners(features[trn], eeg[trn], enc_alpha, window, pca_variance,
                                                         n_folds, seed, device)}
        for name in inputs:
            W = whiteners[name]()
            X_trn, X_val = decoder_input(eeg[trn], window, W, device), decoder_input(eeg[val], window, W, device)
            basis = gpu.ridge_basis(X_trn, pca_variance=None, device=device)
            pred = gpu.ridge_predict(basis, F_trn, X_val, alphas)  # (alphas, val, D)
            totals[name] = totals.get(name, 0) + _percentile_ranks(gpu.cosine_similarity(pred, F_val.float()))
            if not powers:
                continue
            r = reliability(X_trn, F_all[trn], alphas, n_folds, seed).float()  # (alphas, D)
            for power in powers:
                w = (r.clamp_min(0) ** power)[:, None, :]
                sim = torch.nn.functional.normalize(pred * w, dim=-1) @ \
                    torch.nn.functional.normalize(F_val.float() * w, dim=-1).transpose(1, 2)
                key = f"{name}_p{power:g}"
                totals[key] = totals.get(key, 0) + _percentile_ranks(sim)
    return {k: (v / len(features)).cpu().numpy() for k, v in totals.items()}


@dataclass
class DecodingVariant:
    """Decoder with optional noise-whitened input and reliability-weighted cosine (see module docstring).

    ``fit`` needs ``enc_alpha`` (encoding alpha for the residuals) when ``whiten_input``. ``dec_alpha`` is the ridge
    alpha, chosen beforehand together with the variant by :func:`nested_cv_scores`.
    """

    dec_alpha: float
    whiten_input: bool = False
    reliability_power: float | None = None
    enc_alpha: float | None = None
    window: slice = field(default_factory=lambda: DEFAULT_WINDOW)
    pca_variance: float = 0.95
    n_folds: int = 5
    seed: int = 0
    device: str | None = None
    cv_curve: np.ndarray | None = None  # optional: the CV curve the alpha was chosen from (for reporting)

    @property
    def alpha(self) -> float:
        return self.dec_alpha

    @property
    def n_components(self) -> int:
        return int(self.basis_.U.shape[1])

    def fit(self, features: np.ndarray, eeg: np.ndarray) -> "DecodingVariant":
        dev = self.device or gpu.default_device()
        self.whiteners_ = (noise_whiteners(features, eeg, self.enc_alpha, self.window, self.pca_variance,
                                           self.n_folds, self.seed, dev) if self.whiten_input else None)
        X = decoder_input(eeg, self.window, self.whiteners_, dev)
        self.features_trn_ = gpu.as_tensor(features, dev, torch.float64)
        self.basis_ = gpu.ridge_basis(X, pca_variance=None, device=dev)
        self.weights_ = None
        if self.reliability_power is not None:
            r = reliability(X, self.features_trn_, np.array([self.dec_alpha]), self.n_folds, self.seed)[0].float()
            self.weights_ = r.clamp_min(0) ** self.reliability_power
        return self

    def similarity(self, eeg: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
        dev = self.basis_.U.device
        X = decoder_input(eeg, self.window, self.whiteners_, dev)
        targets = gpu.standardize(self.features_trn_, self.features_trn_)
        pred = gpu.ridge_predict(self.basis_, targets, X, np.array([self.dec_alpha]))[0]
        cand = gpu.standardize(gpu.as_tensor(candidate_features, dev, torch.float64), self.features_trn_).float()
        if self.weights_ is not None:
            pred, cand = pred * self.weights_, cand * self.weights_
        return gpu.cosine_similarity(pred, cand).cpu().numpy().astype(np.float64)
