"""Checks that the committed model files reproduce the committed predictions.

    python -m pytest inference/test_predict.py -q

The integrity and unit tests need nothing but this folder. The reproduction tests also need the challenge data
(``tst/`` for sub-05/06, ``tst_stim/`` for the development subjects): pass ``EEG_DATA_ROOT=/path/to/eeg-mind-reading-challenge``
(default: ``eeg-mind-reading-challenge`` next to the repository root); they are skipped when it is missing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODEL = HERE / "model"
DATA = Path(os.environ.get("EEG_DATA_ROOT", REPO / "eeg-mind-reading-challenge"))
REFERENCE = REPO / "submissions" / "final_v4b_csls"  # committed predictions of this model (sub-05, sub-06)

spec = importlib.util.spec_from_file_location("predict", HERE / "predict.py")
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)

META = json.loads((MODEL / "meta.json").read_text())
needs_data = pytest.mark.skipif(not (DATA / "tst").is_dir(), reason=f"challenge data not found at {DATA} (set EEG_DATA_ROOT)")


def test_every_model_file_matches_its_hash_and_is_small() -> None:
    listed = set(META["files"])
    on_disk = {str(p.relative_to(MODEL)) for p in MODEL.rglob("*") if p.is_file() and p.name not in ("meta.json", "README.md")}
    assert listed == on_disk, f"unlisted: {sorted(on_disk - listed)}, missing: {sorted(listed - on_disk)}"
    for rel, info in META["files"].items():
        path = MODEL / rel
        assert path.stat().st_size == info["bytes"] < 50 * 2**20, rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], rel


def test_meta_is_consistent_with_the_files() -> None:
    n = len(META["members"])
    for s in range(1, 7):
        assert len(META["weights"][str(s)]) == n and abs(sum(META["weights"][str(s)]) - 1) < 1e-12
        assert f"sub-{s:02d}/train_eeg_mean.npy" in META["files"]
        for m in META["members"]:
            assert f"sub-{s:02d}/{m['name']}.npz" in META["files"]
            assert m["name"] in META["hyperparameters"][f"sub-{s:02d}"]
    for m in META["members"]:
        cand = np.load(MODEL / "candidates" / f"{m['name']}.npy")
        assert cand.shape[0] == 102 and np.isfinite(cand).all()
    assert all(c["live_vs_cached_max_abs"] < 1e-6 for c in META["checks"].values())
    for key in ("clip_b32", "dinov2_b", "vjepa2_l", "clap"):
        assert len(META["public_models"][key]["revision"]) == 40


def test_validate_rejects_bad_submissions() -> None:
    good = np.tile(predict.TEST_IDS, (102, 1))
    predict.validate(good)
    for bad in (good[:, :101], good.astype(np.float64), np.where(np.arange(102)[None] == 1, 1001, good)):
        with pytest.raises(AssertionError):
            predict.validate(bad)


def test_csls_and_ranking_on_a_toy_matrix() -> None:
    sim = np.array([[3.0, 1.0, 0.0], [2.0, 4.0, 1.0], [0.0, 1.0, 5.0]])
    out = predict.csls(sim, 1)  # k = 1: 2 * sim - row max - column max
    np.testing.assert_allclose(out, 2 * sim - sim.max(1, keepdims=True) - sim.max(0, keepdims=True))
    ranked = predict.TEST_IDS[np.argsort(-np.tile(np.arange(102.0), (102, 1)), axis=1, kind="stable")]
    assert ranked[0, 0] == 1102 and ranked[0, -1] == 1001


@needs_data
def test_dev_subjects_reproduce_the_reported_score() -> None:
    model = predict.Model(MODEL)
    hits1 = hits5 = total = 0
    for s in (1, 2, 3, 4):
        ranked = predict.predict_subject(s, model, DATA)
        pos = np.argmax(ranked == np.load(DATA / "tst_stim" / f"sub-{s:02d}_tst_stim.npy")[:, None], axis=1)
        hits1, hits5, total = hits1 + int((pos < 1).sum()), hits5 + int((pos < 5).sum()), total + len(pos)
    assert (hits1, hits5, total) == (138, 278, 408)  # 33.8% top-1, 68.1% top-5 over sub-01..04


@needs_data
@pytest.mark.skipif(not REFERENCE.is_dir(), reason="submissions/final_v4b_csls not in this checkout")
def test_heldout_subjects_match_the_committed_predictions(tmp_path: Path) -> None:
    model = predict.Model(MODEL)
    for s in (5, 6):
        got = predict.predict_subject(s, model, DATA)
        want = np.load(REFERENCE / f"sub-{s:02d}_ranked_candidate_list.npy")
        assert got.dtype == want.dtype and got.shape == want.shape == (102, 102)
        assert np.array_equal(got, want), f"sub-{s:02d}: {int((got != want).any(axis=1).sum())} rows differ"
