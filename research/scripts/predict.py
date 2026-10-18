"""Train the linear encoding model per subject and write ranked candidate lists.

The same code path runs for dev (sub-01..04) and held-out (sub-05/06) subjects. For dev subjects the
written file is reloaded and scored against ``tst_stim`` and the result is appended to results.csv.

Example:
    python scripts/predict.py --features clip_b32 --frames 1 --subjects 1 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np

from eegmr.data import DEFAULT_ROOT, SFREQ, TEST_IDS, channel_stats, preprocess, zscore_channels, load_test, load_test_labels, load_train
from eegmr.evaluate import RESULT_FIELDS, evaluate
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.decoding import DecodingVariant, nested_cv_scores
from eegmr.linear import DEFAULT_ALPHAS, DecodingModel, EncodingModel, retrieval_cv_curve
from eegmr.matching import Matcher, oof_predictions
from eegmr.retrieval import csls
from eegmr.submission import score_submission_file, write_submission



def run_subject(subject: int, args: argparse.Namespace) -> dict[str, float] | None:
    X_trn, ids = load_train(subject, args.root)
    eeg_trn = preprocess(X_trn, args.max_seconds)
    measured = preprocess(load_test(subject, args.root), args.max_seconds)
    if args.zscore:  # per-channel statistics from this subject's training recordings only
        stats = channel_stats(eeg_trn)
        eeg_trn, measured = zscore_channels(eeg_trn, stats), zscore_channels(measured, stats)
    assert args.window[1] <= measured.shape[-1], f"window {args.window} exceeds {measured.shape[-1]} samples"
    F_trn = load_features(args.features, args.frames, ids, args.feature_dir, args.pooling)
    F_tst = load_features(args.features, args.frames, TEST_IDS, args.feature_dir, args.pooling)
    if args.head == "decoding":
        assert args.match == "corr" and not args.erp_subtract, "matching variants apply to the encoding head only"
        if args.dec_whiten:  # noise-whitened decoder input; alpha by nested train-CV on the whitened input
            enc_alpha = float(DEFAULT_ALPHAS[retrieval_cv_curve(F_trn, eeg_trn, DEFAULT_ALPHAS,
                                                                slice(*args.window)).argmin()])
            curve = nested_cv_scores(F_trn, eeg_trn, DEFAULT_ALPHAS, enc_alpha, powers=(),
                                     window=slice(*args.window), inputs=("whitened",))["whitened"]
            model = DecodingVariant(float(DEFAULT_ALPHAS[curve.argmin()]), True, None, enc_alpha,
                                    slice(*args.window), cv_curve=curve).fit(F_trn, eeg_trn)
        else:
            model = DecodingModel(window=slice(*args.window)).fit(F_trn, eeg_trn)
        similarity = model.similarity(measured, F_tst)
    else:
        model = EncodingModel(pca_variance=args.pca_variance, alpha_selection=args.alpha_selection,
                              whiten=args.whiten, erp_subtract=args.erp_subtract,
                              window=slice(*args.window)).fit(F_trn, eeg_trn)
        matcher = Matcher(kind=args.match, erp_subtract=args.erp_subtract, power=args.match_power,
                          noise_cov=args.noise_cov, noise_similarity=args.noise_similarity, window=slice(*args.window))
        oof = (oof_predictions(F_trn, eeg_trn, model.alpha, args.pca_variance, args.whiten)
               if args.match != "corr" else None)  # training trials only, alpha fixed to the chosen one
        similarity = matcher.fit(eeg_trn, oof).similarity(measured, model.predict(F_tst))
    if args.csls_k:
        similarity = csls(similarity, args.csls_k)
    sim_path = Path("results/similarity") / args.experiment / f"sub-{subject:02d}.npy"
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sim_path, similarity)

    path = write_submission(similarity, subject, args.out)
    cv = f" cv={model.cv_curve.min():.4f}" if model.cv_curve is not None else ""
    true_ids = load_test_labels(subject, args.root)
    if true_ids is None:
        print(f"sub-{subject:02d}: wrote {path} (no labels)  alpha={model.alpha:g} pcs={model.n_components}{cv}")
        return None
    scores = score_submission_file(path, true_ids)
    assert np.isclose(scores["top1"], evaluate(similarity, true_ids)["top1"]), "file and in-memory scores differ"
    print(f"sub-{subject:02d}: top1={scores['top1']:.3f} top5={scores['top5']:.3f} "
          f"median_rank={scores['median_rank']:.0f}  alpha={model.alpha:g} pcs={model.n_components}{cv}")
    return {**scores, "alpha": model.alpha, "n_pcs": model.n_components}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", choices=["encoding", "decoding"], default="encoding",
                   help="encoding: features -> EEG, correlation; decoding: EEG -> features, cosine (alpha by CV)")
    p.add_argument("--dec-whiten", action="store_true",
                   help="decoding: noise-whiten the EEG input per timepoint (encoding OOF residuals, Ledoit-Wolf)")
    p.add_argument("--features", required=True, help="feature cache name, e.g. alexnet or clip_b32")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--pooling", default="mean", help="mean | first:<s> | concat | segments:<k>")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60], help="sample range at 100 Hz for matching")
    p.add_argument("--max-seconds", type=float, default=None, help="discard EEG after this time (default: keep 3 s)")
    p.add_argument("--pca-variance", type=float, default=0.95)
    p.add_argument("--alpha-selection", choices=["gcv", "retrieval"], default="gcv",
                   help="gcv: RidgeCV on all EEG targets; retrieval: 5-fold CV of matching on training videos")
    p.add_argument("--whiten", action="store_true",
                   help="standardize PCA components (makes alpha a no-op; only to reproduce runs before 2026-10-18)")
    p.add_argument("--zscore", action="store_true", help="z-score each EEG channel with training statistics")
    p.add_argument("--erp-subtract", action="store_true",
                   help="subtract the training-mean response from measured and predicted EEG before matching")
    p.add_argument("--match", choices=["corr", "weighted", "noise"], default="corr",
                   help="corr: Pearson correlation; weighted: reliability-weighted; noise: noise-whitened")
    p.add_argument("--match-power", type=float, default=1.0, help="weighted: w = max(r, 0) ** power")
    p.add_argument("--noise-cov", choices=["per_time", "pooled"], default="per_time",
                   help="noise: Ledoit-Wolf covariance per timepoint, or one pooled over the window")
    p.add_argument("--noise-similarity", choices=["corr", "euclid"], default="corr",
                   help="noise: correlation or negative squared distance of the whitened EEG")
    p.add_argument("--csls-k", type=int, default=0, help="apply CSLS with this k (0 = off)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--out", type=Path, default=None, help="default: submissions/<experiment>")
    p.add_argument("--results", type=Path, default=Path("results/results.csv"))
    p.add_argument("--experiment", default=None, help="default: enc_<features>_f<frames>")
    args = p.parse_args()
    crop = f"_t{args.max_seconds:g}s" if args.max_seconds is not None else ""
    zscore = ("_whiten" if args.whiten else "") + ("_z" if args.zscore else "")
    csls_tag = f"_csls{args.csls_k}" if args.csls_k else ""
    pool = "" if args.pooling == "mean" else "_" + args.pooling.replace(":", "")
    head = ("decw" if args.dec_whiten else "dec") if args.head == "decoding" else "enc"
    selection = "" if args.head == "decoding" else f"_{args.alpha_selection}"
    match_tag = "_erp" if args.erp_subtract else ""
    if args.match == "weighted":
        match_tag += f"_weighted-p{args.match_power:g}"
    elif args.match == "noise":
        match_tag += f"_noise-{args.noise_cov.replace('_', '')}-{args.noise_similarity}"
    args.experiment = args.experiment or (f"{head}_{args.features}_f{args.frames}{pool}{crop}_w{args.window[0]}-{args.window[1]}"
                                          f"{selection}{zscore}{match_tag}{csls_tag}")
    args.out = args.out or Path("submissions") / args.experiment

    rows = []
    for subject in args.subjects:
        scores = run_subject(subject, args)
        if scores is not None:
            rows.append({"subject": subject, **scores})
    if not rows:
        return

    subjects = ",".join(f"{r['subject']:02d}" for r in rows)
    print(f"mean over sub-{subjects}: "
          f"top1={np.mean([r['top1'] for r in rows]):.3f} top5={np.mean([r['top5'] for r in rows]):.3f} "
          f"median_rank={np.mean([r['median_rank'] for r in rows]):.1f}")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.results.exists()
    with args.results.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        now = dt.datetime.now().isoformat(timespec="seconds")
        for r in rows:
            writer.writerow({"time": now, "experiment": args.experiment, "features": args.features,
                             "frames": args.frames, "window": f"{args.window[0]}:{args.window[1]}", **r})


if __name__ == "__main__":
    main()
