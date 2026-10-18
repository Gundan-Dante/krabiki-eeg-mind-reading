"""Linear encoding model: stimulus features -> predicted EEG, retrieval by correlation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from eegmr.gpu import retrieval_cv_scores

DEFAULT_ALPHAS = np.logspace(0, 9, 37)  # quarter-decade steps
DEFAULT_WINDOW = slice(5, 60)  # 50-600 ms at 100 Hz


def correlation_similarity(measured: np.ndarray, predicted: np.ndarray, window: slice) -> np.ndarray:
    """Pearson correlation between every measured and every predicted EEG over ``window`` (channels x time flattened).

    Returns ``(n_measured, n_predicted)``.
    """

    def zscore(x: np.ndarray) -> np.ndarray:
        x = x[..., window].reshape(len(x), -1)
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return zscore(measured) @ zscore(predicted).T


def _feature_transform(pca_variance: float | None, whiten: bool = False) -> Pipeline:
    """Standardize -> PCA (skipped if ``pca_variance`` is None), optionally followed by a second standardization.

    ``whiten=True`` makes every PCA component unit-variance, so Z^T Z = n*I and all singular values are equal. Ridge
    then shrinks every direction by the same factor, alpha only rescales predictions, and correlation matching
    ignores it. Kept only to reproduce runs logged before 2026-10-18.
    """
    steps = [StandardScaler()] + ([PCA(n_components=pca_variance, svd_solver="full")] if pca_variance else [])
    return make_pipeline(*steps, *([StandardScaler()] if whiten else []))


def retrieval_cv_curve(features: np.ndarray, eeg: np.ndarray, alphas: np.ndarray, window: slice,
                       **kwargs) -> np.ndarray:
    """:func:`eegmr.gpu.retrieval_cv_scores` for a single window: one score per alpha."""
    return retrieval_cv_scores(features, eeg, alphas, [window], **kwargs)[0]


@dataclass
class EncodingModel:
    """Standardize -> PCA (-> standardize if ``whiten``) -> ridge, predicting the full (channels x time) EEG.

    ``alpha_selection``:
        ``"gcv"``        RidgeCV's generalized cross-validation (minimizes squared error over all EEG targets).
        ``"retrieval"``  k-fold CV on the training videos, maximizing retrieval within ``window``.
    """

    pca_variance: float | None = 0.95
    alphas: np.ndarray = field(default_factory=lambda: DEFAULT_ALPHAS)
    alpha_selection: Literal["gcv", "retrieval"] = "gcv"
    window: slice = field(default_factory=lambda: DEFAULT_WINDOW)
    n_folds: int = 5
    seed: int = 0
    whiten: bool = False
    erp_subtract: bool = False  # alpha selection only: match on EEG minus the (fold-)training mean response
    pipeline: Pipeline | None = None
    eeg_shape: tuple[int, ...] = ()
    cv_curve: np.ndarray | None = None

    def fit(self, features: np.ndarray, eeg: np.ndarray) -> "EncodingModel":
        """``features``: ``(n, D)``; ``eeg``: ``(n, channels, time)``."""
        self.eeg_shape = eeg.shape[1:]
        if self.alpha_selection == "gcv":
            ridge = RidgeCV(alphas=self.alphas)
        else:
            self.cv_curve = retrieval_cv_curve(features, eeg, self.alphas, self.window,
                                               pca_variance=self.pca_variance, n_folds=self.n_folds,
                                               seed=self.seed, whiten=self.whiten, erp_subtract=self.erp_subtract)
            ridge = Ridge(alpha=float(self.alphas[np.argmin(self.cv_curve)]))
        self.pipeline = Pipeline(_feature_transform(self.pca_variance, self.whiten).steps + [("ridge", ridge)])
        self.pipeline.fit(features, eeg.reshape(len(eeg), -1))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(features).reshape(len(features), *self.eeg_shape)

    @property
    def alpha(self) -> float:
        ridge = self.pipeline[-1]
        return float(ridge.alpha_ if isinstance(ridge, RidgeCV) else ridge.alpha)

    @property
    def n_components(self) -> int:
        """Number of feature dimensions entering ridge."""
        return int(self.pipeline[-1].coef_.shape[1])


@dataclass
class DecodingModel:
    """Ridge from the EEG window (channels x time flattened, standardized) to standardized stimulus features.

    Candidates are ranked by cosine similarity between the predicted and the candidate features. Feature and EEG
    standardization use the training statistics. Alpha is chosen by retrieval CV on the training videos.
    """

    alphas: np.ndarray = field(default_factory=lambda: DEFAULT_ALPHAS)
    window: slice = field(default_factory=lambda: DEFAULT_WINDOW)
    n_folds: int = 5
    seed: int = 0
    device: str | None = None
    alpha: float | None = None
    cv_curve: np.ndarray | None = None

    def fit(self, features: np.ndarray, eeg: np.ndarray) -> "DecodingModel":
        """``features``: ``(n, D)``; ``eeg``: ``(n, channels, time)``."""
        from eegmr import gpu

        self.cv_curve = gpu.decoding_cv_scores(eeg, features, self.alphas, [self.window], self.n_folds, self.seed,
                                               self.device)[0]
        self.alpha = float(self.alphas[np.argmin(self.cv_curve)])
        dev = self.device or gpu.default_device()
        self._features_trn = gpu.as_tensor(features, dev, gpu.torch.float64)
        self._basis = gpu.ridge_basis(gpu.as_tensor(eeg[..., self.window], dev).flatten(1).double(), None, device=dev)
        self._targets = gpu.standardize(self._features_trn, self._features_trn)
        return self

    def similarity(self, eeg: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
        """Cosine similarity ``(n_recordings, n_candidates)`` between decoded and candidate features."""
        from eegmr import gpu

        dev = self._basis.U.device
        X = gpu.as_tensor(eeg[..., self.window], dev).flatten(1).double()
        pred = gpu.ridge_predict(self._basis, self._targets, X, np.array([self.alpha]))[0]
        cand = gpu.standardize(gpu.as_tensor(candidate_features, dev, gpu.torch.float64), self._features_trn)
        return gpu.cosine_similarity(pred, cand.float()).cpu().numpy().astype(np.float64)

    @property
    def n_components(self) -> int:
        return int(self._basis.U.shape[1])
