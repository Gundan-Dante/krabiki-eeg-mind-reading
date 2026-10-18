"""Sweep encoding-model settings, choosing among them by training-only retrieval CV.

For every (PCA threshold, matching window) config and every subject, alpha is picked by retrieval CV on the
training videos, and the CV score (mean percentile rank of the true video on held-out training folds; 0.5 = chance)
is recorded. CV scores exist for all six subjects, so configs can be ranked without touching any test label. The
dev test scores (sub-01..04) are reported next to them, only to check whether the CV ranking agrees with test.

Example:
    python scripts/sweep_linear.py --features clip_b32 --frames 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eegmr.data import DEFAULT_ROOT, SFREQ, TEST_IDS, decimate, load_test, load_test_labels, load_train
from eegmr.evaluate import evaluate
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.gpu import retrieval_cv_scores
from eegmr.linear import EncodingModel, correlation_similarity

FINE_ALPHAS = np.logspace(0, 9, 73)  # eighth-decade steps


def parse_pca(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--pooling", default="mean", help="mean | first:<s> | concat | segments:<k>")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--pca", type=parse_pca, nargs="+", default=[None, 0.99, 0.95, 0.9, 0.8])
    p.add_argument("--starts", type=int, nargs="+", default=[0, 5, 10], help="window starts (samples at 100 Hz)")
    p.add_argument("--ends", type=int, nargs="+", default=[40, 50, 60, 80], help="window ends (samples at 100 Hz)")
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--out", type=Path, default=None, help="default: results/sweep_<features>_f<frames>.csv")
    args = p.parse_args()
    out = args.out or Path(f"results/sweep_{args.features}_f{args.frames}"
                                f"{'' if args.pooling == 'mean' else '_' + args.pooling.replace(':', '')}.csv")

    n_raw = int(round(args.max_seconds * SFREQ))
    windows = [(s, e) for s in args.starts for e in args.ends if e > s]
    rows = []
    for subject in args.subjects:
        X, ids = load_train(subject, args.root)
        eeg_trn = decimate(X[..., :n_raw])
        measured = decimate(load_test(subject, args.root)[..., :n_raw])
        true_ids = load_test_labels(subject, args.root)
        F_trn = load_features(args.features, args.frames, ids, args.feature_dir, args.pooling)
        F_tst = load_features(args.features, args.frames, TEST_IDS, args.feature_dir, args.pooling)
        for pca in args.pca:
            curves = retrieval_cv_scores(F_trn, eeg_trn, FINE_ALPHAS, [slice(*w) for w in windows], pca_variance=pca)
            for (start, end), curve in zip(windows, curves):
                alpha = float(FINE_ALPHAS[curve.argmin()])
                row = {"subject": subject, "pca": "none" if pca is None else pca, "window": f"{start}:{end}",
                       "alpha": alpha, "cv_score": float(curve.min())}
                if true_ids is not None:  # dev only: final fit on all training videos, score on test
                    model = EncodingModel(pca_variance=pca, alphas=np.array([alpha])).fit(F_trn, eeg_trn)
                    sim = correlation_similarity(measured, model.predict(F_tst), slice(start, end))
                    row.update(dims=model.n_components, **evaluate(sim, true_ids))
                rows.append(row)
        print(f"sub-{subject:02d} done ({len(args.pca) * len(windows)} configs)", flush=True)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = (df.groupby(["pca", "window"])
                 .agg(cv_all6=("cv_score", "mean"), top1=("top1", "mean"), top5=("top5", "mean"),
                      median_rank=("median_rank", "mean"), alpha_med=("alpha", "median"))
                 .sort_values("cv_all6"))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.round(4))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
