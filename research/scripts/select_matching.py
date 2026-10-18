"""Training-only CV score of every matching variant (no test EEG, no test labels).

Per subject and per ``--erp-subtract`` setting: pick alpha by retrieval CV (as predict.py does), then score each
matcher with ``eegmr.matching.nested_cv_scores`` (matcher parameters estimated inside each fold's training part).
The result is the mean percentile rank of the true video on held-out folds (0 = best, 0.5 = chance), appended to a CSV.

Example:
    python scripts/select_matching.py --features clip_b32 --frames 8 --max-seconds 1.5 --subjects 1 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eegmr.data import DEFAULT_ROOT, load_train
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.linear import DEFAULT_ALPHAS, retrieval_cv_curve
from eegmr.matching import Matcher, nested_cv_scores
from predict import preprocess  # scripts/ is on sys.path when run as a script

VARIANTS = {
    "corr": dict(kind="corr"),
    "weighted-p0.5": dict(kind="weighted", power=0.5),
    "weighted-p1": dict(kind="weighted", power=1.0),
    "weighted-p2": dict(kind="weighted", power=2.0),
    "noise-pertime-corr": dict(kind="noise", noise_cov="per_time", noise_similarity="corr"),
    "noise-pertime-euclid": dict(kind="noise", noise_cov="per_time", noise_similarity="euclid"),
    "noise-pooled-corr": dict(kind="noise", noise_cov="pooled", noise_similarity="corr"),
    "noise-pooled-euclid": dict(kind="noise", noise_cov="pooled", noise_similarity="euclid"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--pca-variance", type=float, default=0.95)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--out", type=Path, default=Path("results/matching_cv.csv"))
    args = p.parse_args()
    window = slice(*args.window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()  # (subject, erp) pairs already in the CSV are skipped
    if args.out.exists():
        with args.out.open(newline="") as fh:
            done = {(int(r["subject"]), int(r["erp_subtract"])) for r in csv.DictReader(fh)}
    new_file = not args.out.exists() or not done
    with args.out.open("a" if done else "w", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["subject", "erp_subtract", "variant", "alpha", "cv_score"])
        for subject in args.subjects:
            if all((subject, int(erp)) in done for erp in (False, True)):
                continue
            X, ids = load_train(subject, args.root)
            eeg = preprocess(X, args.max_seconds)
            feats = load_features(args.features, args.frames, ids, args.feature_dir)
            for erp in (False, True):
                if (subject, int(erp)) in done:
                    continue
                curve = retrieval_cv_curve(feats, eeg, DEFAULT_ALPHAS, window, pca_variance=args.pca_variance, erp_subtract=erp)
                alpha = float(DEFAULT_ALPHAS[np.argmin(curve)])
                matchers = {name: Matcher(erp_subtract=erp, window=window, **kw) for name, kw in VARIANTS.items()}
                scores = nested_cv_scores(feats, eeg, alpha, matchers, args.pca_variance)
                print(f"sub-{subject:02d} erp={erp} alpha={alpha:g} curve_min={curve.min():.4f}  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()), flush=True)
                for name, score in scores.items():
                    writer.writerow([subject, int(erp), name, alpha, f"{score:.6f}"])
                fh.flush()


if __name__ == "__main__":
    main()
