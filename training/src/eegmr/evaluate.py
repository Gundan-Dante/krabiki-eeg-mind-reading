"""Retrieval metrics shared by all tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from eegmr.data import DEFAULT_ROOT, TEST_IDS, load_test_labels

RESULT_FIELDS = ["time", "experiment", "features", "frames", "subject", "top1", "top5", "median_rank",
                 "alpha", "n_pcs", "window"]  # columns of results/results.csv


def true_ranks(similarity: np.ndarray, true_ids: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> np.ndarray:
    """1-based rank of the true video in each row of ``similarity`` (rows: recordings, cols: candidates).

    Ties count against us: a candidate scoring equal to the true one is ranked ahead of it.
    """
    col = np.searchsorted(candidate_ids, true_ids)
    assert np.array_equal(candidate_ids[col], true_ids), "true_ids must be among candidate_ids"
    true_score = similarity[np.arange(len(true_ids)), col]
    return (similarity >= true_score[:, None]).sum(axis=1)  # counts the true video itself -> 1-based


def evaluate(similarity: np.ndarray, true_ids: np.ndarray, candidate_ids: np.ndarray = TEST_IDS) -> dict[str, float]:
    """Top-1 / top-5 accuracy and median rank of the true video."""
    ranks = true_ranks(similarity, true_ids, candidate_ids)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
    }


def trial_ranks(experiment: str, subjects: list[int], root: Path, sim_dir: Path) -> dict[int, np.ndarray]:
    return {s: true_ranks(np.load(sim_dir / experiment / f"sub-{s:02d}.npy"), load_test_labels(s, root))
            for s in subjects}


def compare_table(baseline: str, experiments: list[str], subjects: list[int], root: Path = DEFAULT_ROOT,
                  sim_dir: Path = Path("results/similarity")) -> str:
    """Markdown table: per-subject top-1 / top-5 / median rank, means, and the paired Wilcoxon vs ``baseline``."""
    base = trial_ranks(baseline, subjects, root, sim_dir)
    base_all = np.concatenate([base[s] for s in subjects])
    header = "| experiment | " + " | ".join(f"sub-{s:02d} top1/top5/med" for s in subjects) + \
             " | mean top-1 | mean top-5 | mean median rank | better/worse | Wilcoxon p |"
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for name in [baseline, *experiments]:
        ranks = base if name == baseline else trial_ranks(name, subjects, root, sim_dir)
        cells = [f"{np.mean(ranks[s] <= 1):.1%} / {np.mean(ranks[s] <= 5):.1%} / {np.median(ranks[s]):.1f}"
                 for s in subjects]
        top1 = np.mean([np.mean(ranks[s] <= 1) for s in subjects])
        top5 = np.mean([np.mean(ranks[s] <= 5) for s in subjects])
        med = np.mean([np.median(ranks[s]) for s in subjects])
        if name == baseline:
            cmp = "– | –"
        else:
            all_ranks = np.concatenate([ranks[s] for s in subjects])
            diff = base_all - all_ranks  # > 0: the experiment ranks the true video higher
            p = wilcoxon(base_all, all_ranks).pvalue if np.any(diff != 0) else float("nan")
            cmp = f"{int((diff > 0).sum())} / {int((diff < 0).sum())} | {p:.3g}"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {top1:.1%} | {top5:.1%} | {med:.1f} | {cmp} |")
    return "\n".join(lines)
