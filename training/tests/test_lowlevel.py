"""Checks of ``eegmr.lowlevel``: the pooled-AlexNet derivation and the deterministic hand-crafted descriptors."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from eegmr import lowlevel
from eegmr.features import feature_path


def test_pool_cached_alexnet_conv2_matches_frame_pooling(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(5, 8, 192, 27, 27)).astype(np.float32)  # videos x frames x C x H x W
    np.savez(feature_path("alexnet", 8, tmp_path), ids=np.arange(5), mean=frames.mean(axis=1).reshape(5, -1))
    lowlevel.pool_cached_alexnet_conv2(6, 8, tmp_path)
    with np.load(feature_path("alexnet_p6", 8, tmp_path)) as f:
        got = f["mean"]
    expected = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(frames).flatten(0, 1), 6).view(5, 8, -1).mean(1).numpy()
    assert got.shape == (5, 192 * 36)
    np.testing.assert_allclose(got, expected, atol=1e-5)  # pooling is linear: pool-then-mean == mean-then-pool


@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 96))
    if not writer.isOpened():
        pytest.skip("cv2 cannot write mp4 here")
    for t in range(30):  # a bright bar moving to the right on a colour gradient
        img = np.zeros((96, 96, 3), np.uint8)
        img[..., 2] = np.linspace(0, 255, 96, dtype=np.uint8)[None]
        img[:, 3 * t:3 * t + 12] = (255, 255, 255)
        writer.write(img)
    writer.release()
    return path


def test_video_descriptors_shapes_determinism_and_motion(synthetic_video: Path) -> None:
    a = lowlevel.video_descriptors(synthetic_video)
    b = lowlevel.video_descriptors(synthetic_video)
    assert {k: v.shape for k, v in a.items()} == {"color": (78,), "freq": (104,), "edge": (24,), "motion": (49,)}
    assert all(np.isfinite(v).all() for v in a.values())
    assert all(np.array_equal(a[k], b[k]) for k in a)  # deterministic
    assert a["motion"][16] > 0  # overall mean absolute frame difference: the bar moves
