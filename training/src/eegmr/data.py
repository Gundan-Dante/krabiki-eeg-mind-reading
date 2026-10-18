"""Loading and preprocessing of the challenge EEG data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import decimate as _decimate

DEFAULT_ROOT = Path("eeg-mind-reading-challenge")
SFREQ = 500
TRAIN_IDS = np.arange(1, 1001)
TEST_IDS = np.arange(1001, 1103)
ALL_IDS = np.concatenate([TRAIN_IDS, TEST_IDS])
DEV_SUBJECTS = (1, 2, 3, 4)
HELDOUT_SUBJECTS = (5, 6)


def load_train(subject: int, root: Path = DEFAULT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Return training EEG ``(n, 17, 1500)`` and the matching video IDs ``(n,)``, sorted by ID."""
    with np.load(Path(root) / "trn" / f"sub-{subject:02d}_trn.npz") as trn:
        ids = np.array(sorted(trn.files, key=int), dtype=int)
        X = np.stack([trn[str(i)] for i in ids]).astype(np.float32)
    return X, ids


def load_test(subject: int, root: Path = DEFAULT_ROOT) -> np.ndarray:
    """Return test EEG ``(102, 17, 1500)`` in the original (shuffled) file order."""
    return np.load(Path(root) / "tst" / f"sub-{subject:02d}_tst.npy").astype(np.float32)


def load_test_labels(subject: int, root: Path = DEFAULT_ROOT) -> np.ndarray | None:
    """Return true video IDs ``(102,)`` for the test recordings, or ``None`` for held-out subjects."""
    path = Path(root) / "tst_stim" / f"sub-{subject:02d}_tst_stim.npy"
    return np.load(path).astype(int) if path.exists() else None


def decimate(X: np.ndarray, factor: int = 5) -> np.ndarray:
    """Downsample along the last axis with a zero-phase FIR anti-aliasing filter (500 Hz -> 100 Hz)."""
    return _decimate(X, factor, ftype="fir", zero_phase=True, axis=-1).astype(np.float32)


def preprocess(X: np.ndarray, max_seconds: float | None) -> np.ndarray:
    """Crop the raw 500 Hz EEG (before filtering, so no later samples leak in), then decimate to 100 Hz."""
    if max_seconds is not None:
        X = X[..., : int(round(max_seconds * SFREQ))]
    return decimate(X)


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of ``(n, channels, time)`` EEG over trials and time, shaped ``(channels, 1)``."""
    return X.mean(axis=(0, 2))[:, None], X.std(axis=(0, 2))[:, None]


def zscore_channels(X: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Z-score each channel with precomputed (training) statistics from :func:`channel_stats`."""
    mean, std = stats
    return ((X - mean) / std).astype(np.float32)


def video_path(video_id: int, root: Path = DEFAULT_ROOT) -> Path:
    """Path of the ``.mp4`` for a video ID (train: 1..1000, test: 1001..1102)."""
    folder = "stimuli_train" if video_id <= 1000 else "stimuli_test"
    return Path(root) / folder / f"{video_id:04d}.mp4"
