"""Torch linear algebra shared by all tracks: multi-alpha ridge, correlation similarity, retrieval CV.

Every function takes ``device`` (default: CUDA if available, else CPU) and accepts numpy arrays or tensors.
Results match the sklearn/numpy reference (``eegmr.linear``) up to float32 rounding; see ``tests/test_gpu.py``.

The ridge here is the full feature pipeline ``StandardScaler -> PCA(variance) -> Ridge`` for many alphas at once.
It needs a single SVD of the standardized training features, X = U S V^T: the PCA scores are U_k S_k, whose own
SVD is (U_k, S_k, I), so the prediction for new rows is

    X_new V_k diag(S_k / (S_k^2 + alpha)) U_k^T (Y - y_mean) + y_mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import KFold

ArrayLike = np.ndarray | torch.Tensor


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def as_tensor(x: ArrayLike, device: str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
    return torch.as_tensor(x, dtype=dtype, device=device or default_device())


@dataclass
class RidgeBasis:
    """One SVD of the standardized training features, reusable for any alpha (and any targets)."""

    mean: torch.Tensor  # (D,) feature mean
    std: torch.Tensor  # (D,) feature std (0 replaced by 1, as sklearn)
    U: torch.Tensor  # (n, k)
    S: torch.Tensor  # (k,)
    V: torch.Tensor  # (D, k)
    whiten: bool

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """PCA scores of new rows in the SVD basis of the training scores, ``(m, k)``."""
        Z = ((X - self.mean) / self.std) @ self.V
        return Z * (len(self.U) ** 0.5 / self.S) if self.whiten else Z

    @property
    def singular_values(self) -> torch.Tensor:
        return torch.full_like(self.S, len(self.U) ** 0.5) if self.whiten else self.S


def ridge_basis(X_trn: ArrayLike, pca_variance: float | None = 0.95, whiten: bool = False,
                device: str | None = None) -> RidgeBasis:
    """Standardize, SVD (float64) and keep the components sklearn's ``PCA(n_components=pca_variance)`` would keep.

    ``pca_variance=None`` keeps every component (plain ridge on standardized features). ``whiten=True`` reproduces
    the old scaler -> PCA -> scaler pipeline, where all singular values equal sqrt(n).
    """
    X = as_tensor(X_trn, device, torch.float64)
    mean, std = X.mean(dim=0), X.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    U, S, Vh = torch.linalg.svd((X - mean) / std, full_matrices=False)
    k = len(S)
    if pca_variance:
        ratio_cum = torch.cumsum(S**2, dim=0) / (S**2).sum()
        threshold = torch.tensor(pca_variance, dtype=S.dtype, device=S.device)
        k = min(int(torch.searchsorted(ratio_cum, threshold, right=True)) + 1, k)
    return RidgeBasis(mean, std, U[:, :k], S[:k], Vh[:k].T, whiten)


def ridge_predict(basis: RidgeBasis, Y_trn: ArrayLike, X_new: ArrayLike, alphas: ArrayLike) -> torch.Tensor:
    """Ridge predictions for every alpha, ``(n_alphas, m, T)``; ``Y_trn`` is ``(n, T)``. Returns float32."""
    dev = basis.U.device
    Y = as_tensor(Y_trn, dev, torch.float64)
    y_mean = Y.mean(dim=0)
    UtY = (basis.U.T @ (Y - y_mean)).float()
    Z = basis.project(as_tensor(X_new, dev, torch.float64)).float()
    s = basis.singular_values.float()
    shrink = s / (s**2 + as_tensor(alphas, dev)[:, None])  # (alphas, k)
    return torch.einsum("mk,ak,kt->amt", Z, shrink, UtY) + y_mean.float()


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    """Flatten the last two axes (channels x time) and scale each row to zero mean, unit norm."""
    x = x.flatten(-2)
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True)


def correlation_similarity(measured: ArrayLike, predicted: ArrayLike, window: slice = slice(None),
                           device: str | None = None) -> torch.Tensor:
    """Pearson correlation over ``window`` (channels x time flattened) between all measured and predicted EEGs.

    ``measured``: ``(n, C, T)``; ``predicted``: ``(..., m, C, T)`` with optional leading batch axes (e.g. alphas).
    Returns ``(..., n, m)``.
    """
    dev = predicted.device if isinstance(predicted, torch.Tensor) else device
    m = row_zscore(as_tensor(measured, dev)[..., window])
    p = row_zscore(as_tensor(predicted, dev)[..., window])
    return m @ p.transpose(-1, -2)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between the rows of ``a`` (``(..., n, D)``) and ``b`` (``(m, D)``): ``(..., n, m)``."""
    return torch.nn.functional.normalize(a, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).T


def standardize(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Z-score the columns of ``x`` with the mean/std of ``ref`` (std 0 -> 1)."""
    std = ref.std(dim=0, unbiased=False)
    return (x - ref.mean(dim=0)) / torch.where(std == 0, torch.ones_like(std), std)


def decoding_cv_scores(
    eeg: ArrayLike,
    features: ArrayLike,
    alphas: ArrayLike,
    windows: list[slice],
    n_folds: int = 5,
    seed: int = 0,
    device: str | None = None,
) -> np.ndarray:
    """Retrieval CV for the decoding direction (EEG window -> standardized features, cosine matching).

    ``eeg``: ``(n, C, T)``; ``features``: ``(n, D)``. Same folds and score as :func:`retrieval_cv_scores` (mean
    percentile rank of the true video among the held-out videos). Returns ``(len(windows), len(alphas))``.
    """
    device = device or default_device()
    eeg = as_tensor(eeg, device, torch.float64)
    features = as_tensor(features, device, torch.float64)
    scores = torch.zeros(len(windows), len(alphas), dtype=torch.float64, device=device)
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(len(features))):
        F_trn, F_val = standardize(features[trn], features[trn]), standardize(features[val], features[trn])
        for i, w in enumerate(windows):
            basis = ridge_basis(eeg[trn][..., w].flatten(1), pca_variance=None, device=device)
            pred = ridge_predict(basis, F_trn, eeg[val][..., w].flatten(1), alphas)  # (alphas, val, D)
            sim = cosine_similarity(pred, F_val.float())
            ranks = (sim >= sim.diagonal(dim1=1, dim2=2)[..., None]).sum(dim=-1)
            scores[i] += ((ranks - 1) / (len(val) - 1)).sum(dim=-1).double()
    return (scores / len(features)).cpu().numpy()


def retrieval_cv_scores(
    features: ArrayLike,
    eeg: ArrayLike,
    alphas: ArrayLike,
    windows: list[slice],
    pca_variance: float | None = 0.95,
    n_folds: int = 5,
    seed: int = 0,
    whiten: bool = False,
    erp_subtract: bool = False,
    device: str | None = None,
) -> np.ndarray:
    """Mean percentile rank (0 = always first, 0.5 = chance) of the true video on held-out training folds.

    ``features``: ``(n, D)``; ``eeg``: ``(n, C, T)``. Returns ``(len(windows), len(alphas))``. Ridge solves each EEG
    target independently, so one prediction over the union of the windows serves every window. Held-out videos are
    retrieved among each other with the same correlation matching used at test time. With ``erp_subtract`` the
    fold's training-mean response is removed from the held-out and predicted EEG first.
    """
    device = device or default_device()
    lo, hi = min(w.start for w in windows), max(w.stop for w in windows)
    eeg = as_tensor(eeg, device)[..., lo:hi]
    local = [slice(w.start - lo, w.stop - lo) for w in windows]
    features = as_tensor(features, device, torch.float64)
    scores = torch.zeros(len(windows), len(alphas), dtype=torch.float64, device=device)
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(len(features))):
        basis = ridge_basis(features[trn], pca_variance, whiten, device)
        pred = ridge_predict(basis, eeg[trn].flatten(1), features[val], alphas).view(len(alphas), len(val), *eeg.shape[1:])
        measured = eeg[val]
        if erp_subtract:
            erp = eeg[trn].mean(dim=0)
            measured, pred = measured - erp, pred - erp
        for i, w in enumerate(local):
            sim = correlation_similarity(measured, pred, w)  # (alphas, val, val)
            ranks = (sim >= sim.diagonal(dim1=1, dim2=2)[..., None]).sum(dim=-1)  # true video is on the diagonal
            scores[i] += ((ranks - 1) / (len(val) - 1)).sum(dim=-1).double()
        del pred
    return (scores / len(features)).cpu().numpy()
