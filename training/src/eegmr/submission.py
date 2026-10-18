"""Writing and validating ranked candidate lists in the official submission format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eegmr.data import TEST_IDS


def rank_candidates(similarity: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> np.ndarray:
    """Turn a ``(n_recordings, n_candidates)`` similarity matrix into video IDs sorted best-first per row.

    Row order is preserved exactly: row i of the output belongs to row i of ``similarity``.
    """
    order = np.argsort(-similarity, axis=1, kind="stable")
    return candidate_ids[order].astype(np.int64)


def validate_ranked(ranked: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> None:
    """Raise if ``ranked`` is not a valid submission array."""
    n = len(candidate_ids)
    assert ranked.shape == (n, n), f"expected shape {(n, n)}, got {ranked.shape}"
    assert np.issubdtype(ranked.dtype, np.integer), f"expected integer dtype, got {ranked.dtype}"
    expected = np.sort(candidate_ids)
    bad = [i for i, row in enumerate(ranked) if not np.array_equal(np.sort(row), expected)]
    assert not bad, f"rows {bad[:5]}... are not permutations of {expected[0]}..{expected[-1]}"


def submission_path(subject: int, out_dir: Path) -> Path:
    return Path(out_dir) / f"sub-{subject:02d}_ranked_candidate_list.npy"


def write_submission(similarity: np.ndarray, subject: int, out_dir: Path) -> Path:
    """Rank, validate and save ``sub-XX_ranked_candidate_list.npy``; return its path."""
    ranked = rank_candidates(similarity)
    validate_ranked(ranked)
    path = submission_path(subject, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ranked)
    return path


def score_submission_file(path: Path, true_ids: np.ndarray) -> dict[str, float]:
    """Score a saved submission file exactly as the organizers would (reload from disk)."""
    ranked = np.load(path)
    validate_ranked(ranked)
    hit_pos = np.argmax(ranked == np.asarray(true_ids)[:, None], axis=1)  # 0-based position of truth
    return {"top1": float(np.mean(hit_pos < 1)), "top5": float(np.mean(hit_pos < 5)),
            "median_rank": float(np.median(hit_pos + 1))}
