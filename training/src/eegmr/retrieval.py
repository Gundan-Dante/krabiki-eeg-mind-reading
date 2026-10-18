"""Post-processing of recording x candidate similarity matrices."""

from __future__ import annotations

import numpy as np


def csls(similarity: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling (Conneau et al., 2018) to penalize "hub" candidates.

    ``2 * sim - r_rec - r_cand``, where ``r_rec[i]`` is the mean of recording i's k highest similarities and
    ``r_cand[j]`` the mean of candidate j's k highest similarities. Uses no labels. Within a row only the
    ``r_cand`` term changes the ranking.
    """
    k_rec, k_cand = min(k, similarity.shape[1]), min(k, similarity.shape[0])
    r_rec = np.sort(similarity, axis=1)[:, -k_rec:].mean(axis=1)
    r_cand = np.sort(similarity, axis=0)[-k_cand:, :].mean(axis=0)
    return 2 * similarity - r_rec[:, None] - r_cand[None, :]
