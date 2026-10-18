"""Choose decoder variants (noise-whitened input, reliability-weighted cosine) by training-only nested CV.

For every subject: the encoding alpha (for the residual covariances) is chosen by encoding retrieval CV, then
``eegmr.decoding.nested_cv_scores`` gives a CV curve over alphas for each variant. Each variant takes its own best
alpha. For the dev subjects, every variant is also scored on the test set, with a paired Wilcoxon test on per-trial
ranks against the plain decoder (``raw``, identical to ``predict.py --head decoding``).

Example:
    python scripts/select_decoding.py --features clip_b32 --frames 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eegmr import gpu
from eegmr.data import DEFAULT_ROOT, SFREQ, TEST_IDS, decimate, load_test, load_test_labels, load_train
from eegmr.decoding import DecodingVariant, nested_cv_scores
from eegmr.evaluate import evaluate, true_ranks
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.linear import DEFAULT_ALPHAS, retrieval_cv_curve


def parse_variant(name: str) -> tuple[bool, float | None]:
    """``"whitened_p0.5"`` -> (True, 0.5); ``"raw"`` -> (False, None)."""
    base, _, power = name.partition("_p")
    return base == "whitened", float(power) if power else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="clip_b32")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--powers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--out", type=Path, default=Path("results/decoding_variants.csv"))
    args = p.parse_args()
    window, n_raw = slice(*args.window), int(round(args.max_seconds * SFREQ))

    rows, ranks = [], {}
    for subject in args.subjects:
        X, ids = load_train(subject, args.root)
        eeg = decimate(X[..., :n_raw])
        F_trn = load_features(args.features, args.frames, ids, args.feature_dir)
        enc_alpha = float(DEFAULT_ALPHAS[retrieval_cv_curve(F_trn, eeg, DEFAULT_ALPHAS, window).argmin()])
        curves = nested_cv_scores(F_trn, eeg, DEFAULT_ALPHAS, enc_alpha, tuple(args.powers), window)
        plain = gpu.decoding_cv_scores(eeg, F_trn, DEFAULT_ALPHAS, [window])[0]
        assert np.abs(curves["raw"] - plain).max() < 1e-6, "nested raw curve must equal the plain decoder's CV"

        true_ids = load_test_labels(subject, args.root)
        if true_ids is not None:
            measured = decimate(load_test(subject, args.root)[..., :n_raw])
            F_tst = load_features(args.features, args.frames, TEST_IDS, args.feature_dir)
        for name, curve in curves.items():
            whiten, power = parse_variant(name)
            alpha = float(DEFAULT_ALPHAS[curve.argmin()])
            row = {"subject": subject, "variant": name, "alpha": alpha, "enc_alpha": enc_alpha,
                   "cv_score": float(curve.min())}
            if true_ids is not None:
                model = DecodingVariant(alpha, whiten, power, enc_alpha, window).fit(F_trn, eeg)
                sim = model.similarity(measured, F_tst)
                row.update(evaluate(sim, true_ids))
                ranks.setdefault(name, []).append(true_ranks(sim, true_ids))
            rows.append(row)
        print(f"sub-{subject:02d} done (enc alpha {enc_alpha:g})", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby("variant").agg(cv_all=("cv_score", "mean"), top1=("top1", "mean"), top5=("top5", "mean"),
                                        median_rank=("median_rank", "mean"), alpha_med=("alpha", "median"))
    if ranks:
        base = np.concatenate(ranks["raw"])
        for name in summary.index:
            r = np.concatenate(ranks[name])
            summary.loc[name, "better"], summary.loc[name, "worse"] = np.sum(r < base), np.sum(r > base)
            summary.loc[name, "p"] = wilcoxon(base, r).pvalue if np.any(r != base) else 1.0
    with pd.option_context("display.width", 200):
        print(summary.sort_values("cv_all").round(4))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
