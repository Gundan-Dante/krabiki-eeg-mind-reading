"""Low-level visual features of the stimulus videos, computed locally with deterministic code (no external data).

Posterior EEG channels respond mainly to low-level image structure (contrast, edges, spatial frequency, colour, motion),
which semantic embeddings (CLIP, DINOv2, V-JEPA) retain only weakly. Two families, both cached as
``features/<name>_f<n>.npz`` (``ids``, ``mean``) so ``--member decw:<name>:<n>`` works in ``scripts/ensemble.py``:

    AlexNet maps       torchvision AlexNet, weights IMAGENET1K_V1, pre-ReLU output of conv layer ``c1..c5``
                       (``features[0, 3, 6, 8, 10]``), frames resized to 224 x 224 (area interpolation), ImageNet mean/std,
                       8 frames at the centres of 8 equal segments (as ``features.read_frames``), average-pooled to P x P
                       per channel, then averaged over the 8 frames.  ``alexnet_p<P>`` is conv2 derived from the cached
                       ``alexnet_f8`` (192 x 27 x 27, the mean of the same 8 frames); ``alexnet_c<k>p<P>`` is extracted fresh.
    Hand-crafted       ``lowlevel_{color,freq,edge,motion}`` and their concatenation ``lowlevel_all`` (see :func:`video_descriptors`).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from eegmr.data import ALL_IDS, DEFAULT_ROOT, video_path
from eegmr.features import DEFAULT_FEATURE_DIR, feature_path, read_frames

ALEX_TAPS = {"c1": 0, "c2": 3, "c3": 6, "c4": 8, "c5": 10}  # indices of the conv layers in torchvision AlexNet.features
GROUPS = ("color", "freq", "edge", "motion")


# ---------------------------------------------------------------------------------------------------- AlexNet maps
def pool_cached_alexnet_conv2(pool: int, n_frames: int = 8, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    """``alexnet_p<pool>_f<n>``: average-pool the cached conv2 maps (``alexnet_f<n>``, 192 x 27 x 27 per video) to pool x pool.

    The cache holds the mean over the frames of the pre-ReLU ``features.3`` output; pooling is linear, so this equals
    pooling every frame first and then averaging."""
    with np.load(feature_path("alexnet", n_frames, feature_dir)) as f:
        ids, mean = f["ids"], f["mean"]
    maps = torch.from_numpy(mean).float().view(len(mean), 192, 27, 27)
    pooled = torch.nn.functional.adaptive_avg_pool2d(maps, pool).flatten(1).numpy()
    out = feature_path(f"alexnet_p{pool}", n_frames, feature_dir)
    np.savez(out, ids=ids, mean=pooled)
    return out


def extract_alexnet_layers(layers: list[str], pool: int, n_frames: int = 8, root: Path = DEFAULT_ROOT,
                           feature_dir: Path = DEFAULT_FEATURE_DIR,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu") -> list[Path]:
    """Extract ``alexnet_<layer>p<pool>_f<n>`` for conv layers such as ``["c1", "c3", "c4", "c5"]`` (one pass over the videos)."""
    from torchvision.models import AlexNet_Weights, alexnet

    net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features.eval().to(device)
    taps = {ALEX_TAPS[name]: name for name in layers}
    last = max(taps)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out: dict[str, list[np.ndarray]] = {name: [] for name in layers}
    with torch.no_grad():
        for vid in tqdm(ALL_IDS, desc="alexnet layers"):
            frames = read_frames(video_path(int(vid), root), n_frames)
            x = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in frames])
            h = (torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            for i, layer in enumerate(net):
                h = layer(h)
                if i in taps:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(h, pool).flatten(1)
                    out[taps[i]].append(pooled.mean(dim=0).cpu().numpy())
                if i == last:
                    break
    paths = []
    for name, rows in out.items():
        path = feature_path(f"alexnet_{name}p{pool}", n_frames, feature_dir)
        np.savez(path, ids=ALL_IDS, mean=np.stack(rows).astype(np.float32))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------------------------- hand-crafted descriptors
def _read_all_gray(path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """All frames of a video: ``(n, size, size)`` float32 gray in [0, 1] and the matching BGR frames resized to 64 x 64."""
    cap = cv2.VideoCapture(str(path))
    gray, small = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA))
        small.append(cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA))
    cap.release()
    if not gray:
        raise RuntimeError(f"could not decode {path}")
    return np.stack(gray).astype(np.float32) / 255.0, np.stack(small)


def _grid_mean(x: np.ndarray, cells: int) -> np.ndarray:
    """Mean of a ``(H, W)`` or ``(H, W, C)`` map over a cells x cells grid -> ``(cells * cells[, C])`` (H, W divisible by cells)."""
    h, w = x.shape[:2]
    y = x.reshape(cells, h // cells, cells, w // cells, *x.shape[2:]).mean(axis=(1, 3))
    return y.reshape(cells * cells, *x.shape[2:]) if x.ndim == 2 else y.reshape(cells * cells, -1)


def _power_bands(gray: np.ndarray, edges: list[int]) -> np.ndarray:
    """Log mean power in radial-frequency bands (cycles / image, ``edges``) x 4 orientations of a Hann-windowed square image."""
    n = gray.shape[0]
    win = np.hanning(n)[:, None] * np.hanning(n)[None]
    power = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))) ** 2
    fy, fx = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")
    radius, angle = np.hypot(fx, fy), np.mod(np.degrees(np.arctan2(fy, fx)), 180.0)
    vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for o in range(4):
            sel = (radius >= lo) & (radius < hi) & (angle >= o * 45 - 22.5) & (angle < o * 45 + 22.5)
            sel |= (o == 0) & (radius >= lo) & (radius < hi) & (angle >= 157.5)  # wrap-around of the 0 degree bin
            vals.append(np.log(power[sel].mean() + 1e-8))
    return np.array(vals)


def video_descriptors(path: Path, n_static: int = 8) -> dict[str, np.ndarray]:
    """Deterministic hand-crafted descriptors of one video, by group (all computed from the decoded frames only):

    color   (78)  Lab mean per cell and L std per cell of a 4 x 4 grid (64); 12-bin saturation-weighted hue histogram; mean S, mean V.
    freq    (104) log power in 6 log-spaced radial bands (1-64 cycles/image) x 4 orientations of the 128 x 128 gray frame (24), and in
                  5 bands (1-32) x 4 orientations for each 64 x 64 quadrant (80); average over ``n_static`` frames.
    edge    (24)  Sobel magnitude per 4 x 4 grid cell (16); mean |gx| and |gy| per 2 x 2 cell (8); average over ``n_static`` frames.
    motion  (49)  mean absolute difference of consecutive frames per 4 x 4 cell (16) and overall (1); Farneback flow between frames
                  t and t + 2 (64 x 64 gray): mean magnitude per 4 x 4 cell (16), mean dx, dy per 2 x 2 cell (8), 8-bin log
                  magnitude histogram (8); averaged over the clip.
    """
    gray128, small = _read_all_gray(path, 128)
    n = len(gray128)
    idx = ((np.arange(n_static) + 0.5) * n / n_static).astype(int)  # same frame choice as features.read_frames
    color, freq, edge = [], [], []
    for i in idx:
        lab = cv2.cvtColor(small[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(small[i], cv2.COLOR_BGR2HSV).astype(np.float32)
        l_ch = lab[..., 0]
        grid_std = _grid_mean(l_ch**2, 4).ravel() - _grid_mean(l_ch, 4).ravel() ** 2
        hue = np.histogram(hsv[..., 0], bins=12, range=(0, 180), weights=hsv[..., 1] / 255.0)[0]
        color.append(np.concatenate([_grid_mean(lab, 4).ravel(), np.sqrt(np.maximum(grid_std, 0)), hue / (hue.sum() + 1e-8),
                                     [hsv[..., 1].mean() / 255.0, hsv[..., 2].mean() / 255.0]]))
        g = gray128[i]
        quads = [g[r:r + 64, c:c + 64] for r in (0, 64) for c in (0, 64)]
        freq.append(np.concatenate([_power_bands(g, [1, 2, 4, 8, 16, 32, 64])] + [_power_bands(q, [1, 2, 4, 8, 16, 32]) for q in quads]))
        g64 = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
        gx, gy = cv2.Sobel(g64, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g64, cv2.CV_32F, 0, 1, ksize=3)
        edge.append(np.concatenate([_grid_mean(np.hypot(gx, gy), 4).ravel(), _grid_mean(np.abs(gx), 2).ravel(),
                                    _grid_mean(np.abs(gy), 2).ravel()]))
    g64_all = np.stack([cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA) for g in gray128])
    diffs = np.abs(np.diff(g64_all, axis=0)) if n > 1 else np.zeros((1, 64, 64), np.float32)
    mag_grid, flow_dir, hist = [], [], np.zeros(8)
    bins = np.logspace(-2, 1, 9)  # magnitude histogram edges in pixels (64 x 64 frame)
    pairs = [(t, t + 2) for t in range(0, n - 2)] or [(0, 0)]
    for t0, t1 in pairs:
        flow = cv2.calcOpticalFlowFarneback((g64_all[t0] * 255).astype(np.uint8), (g64_all[t1] * 255).astype(np.uint8), None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        mag_grid.append(_grid_mean(mag, 4).ravel())
        flow_dir.append(np.concatenate([_grid_mean(flow[..., 0], 2).ravel(), _grid_mean(flow[..., 1], 2).ravel()]))
        hist += np.histogram(mag, bins=bins)[0] / mag.size
    motion = np.concatenate([np.mean([_grid_mean(d, 4).ravel() for d in diffs], axis=0), [diffs.mean()],
                             np.mean(mag_grid, axis=0), np.mean(flow_dir, axis=0), hist / len(pairs)])
    return {"color": np.mean(color, axis=0), "freq": np.mean(freq, axis=0), "edge": np.mean(edge, axis=0), "motion": motion}


def _describe(args: tuple[int, Path]) -> dict[str, np.ndarray]:
    cv2.setNumThreads(1)
    return video_descriptors(video_path(args[0], args[1]))


def extract_descriptors(root: Path = DEFAULT_ROOT, feature_dir: Path = DEFAULT_FEATURE_DIR, workers: int = 8) -> list[Path]:
    """Cache ``lowlevel_{color,freq,edge,motion}_f1`` and ``lowlevel_all_f1`` (the groups concatenated) for all 1,102 videos."""
    from multiprocessing import get_context

    with get_context("spawn").Pool(workers) as pool:
        rows = list(tqdm(pool.imap(_describe, [(int(v), root) for v in ALL_IDS], chunksize=8), total=len(ALL_IDS),
                         desc="descriptors"))
    paths = []
    for group in GROUPS:
        path = feature_path(f"lowlevel_{group}", 1, feature_dir)
        np.savez(path, ids=ALL_IDS, mean=np.stack([r[group] for r in rows]).astype(np.float32))
        paths.append(path)
    path = feature_path("lowlevel_all", 1, feature_dir)
    np.savez(path, ids=ALL_IDS, mean=np.stack([np.concatenate([r[g] for g in GROUPS]) for r in rows]).astype(np.float32))
    return paths + [path]
