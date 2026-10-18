"""Training-only nested CV of the encoder alpha under plain correlation and under noise-whitened Euclid matching.

For every subject and every other grid alpha, scores both matchers with ``eegmr.matching.nested_cv_scores`` (matcher
parameters estimated inside each fold). Mean percentile rank of the true video, 0.5 = chance, lower is better.

Example:
    python scripts/sweep_alpha_euclid.py --features clip_b32 --frames 8 --max-seconds 1.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eegmr.data import DEFAULT_ROOT, load_train
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.linear import DEFAULT_ALPHAS
from eegmr.matching import Matcher, nested_cv_scores
from predict import preprocess  # scripts/ is on sys.path when run as a script


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--out", type=Path, default=Path("results/alpha_euclid_cv.csv"))
    args = p.parse_args()
    window = slice(*args.window)
    matchers = {"corr": Matcher(window=window),
                "noise-pertime-euclid": Matcher(kind="noise", noise_cov="per_time", noise_similarity="euclid", window=window)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "alpha", "variant", "cv_score"])
        for subject in args.subjects:
            X, ids = load_train(subject, args.root)
            eeg = preprocess(X, args.max_seconds)
            feats = load_features(args.features, args.frames, ids, args.feature_dir)
            for alpha in DEFAULT_ALPHAS[::2]:  # half-decade steps
                scores = nested_cv_scores(feats, eeg, float(alpha), matchers)
                for name, score in scores.items():
                    writer.writerow([subject, alpha, name, f"{score:.6f}"])
                fh.flush()
                print(subject, f"{alpha:.3g}", {k: round(v, 4) for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
