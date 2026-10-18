"""Linear contrastive heads (InfoNCE on frozen features): train-CV selection and dev confirmation.

Mode ``cv`` (default): for every subject, outer 5-fold train-CV of every config in the grid, with the ridge decoder
(M8a) on the same folds as the reference. Writes one row per (subject, config) and prints the configs ranked by mean
train-CV score over all subjects. No test labels are touched.

Mode ``dev``: for the named configs (or ``best`` = lowest mean train-CV in the CV file), train the final head per
subject on all training videos, for each seed. It reports dev top-1 / top-5 / median rank (mean ± std over seeds), a
paired Wilcoxon test on per-trial ranks vs the M8a decoder, and a row-z ensemble with M8a (weight ``--ens-weight``). It writes
similarity matrices and validated submission files for seed 0.

Examples:
    python scripts/train_contrastive.py --features clip_b32 --frames 8
    python scripts/train_contrastive.py --mode dev --configs best --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from eegmr import gpu
from eegmr.contrastive import SELECTED_CONFIGS, ContrastiveConfig, cv_scores, prepare, train_head
from eegmr.data import DEFAULT_ROOT, SFREQ, TEST_IDS, decimate, load_test, load_test_labels, load_train
from eegmr.decoding import DecodingVariant, nested_cv_scores, noise_whiteners
from eegmr.evaluate import evaluate, true_ranks
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.linear import DEFAULT_ALPHAS, retrieval_cv_curve
from eegmr.submission import write_submission


def grid(name: str = "main") -> list[ContrastiveConfig]:
    """``main``: (A) linear from scratch, (B) two towers, (C) ridge warm start (plain / anchored decay).
    ``selected``: the CV-best config of each variant on CLIP (for repeating on other features).
    ``strong_wd``: (A) with weight decay large enough to act within the few dozen steps before early stopping."""
    if name == "selected":  # CV-best config per variant on CLIP B-32 (results/contrastive_cv_clip_b32.csv)
        return list(SELECTED_CONFIGS.values())
    if name == "strong_wd":
        return [ContrastiveConfig("linear", lr=lr, weight_decay=wd) for lr in (1e-3, 3e-3) for wd in (0.1, 1.0, 10.0)]
    A = [ContrastiveConfig("linear", lr=lr, weight_decay=wd) for lr in (3e-4, 1e-3, 3e-3) for wd in (1e-4, 1e-3, 1e-2)]
    B = [ContrastiveConfig("two_tower", dim=d, lr=1e-3, weight_decay=wd) for d in (64, 128, 256) for wd in (1e-3, 1e-2)]
    C = [ContrastiveConfig("warm", lr=lr, weight_decay=wd, anchor=anchor)
         for lr in (1e-4, 3e-4) for wd in (1e-3, 1e-2) for anchor in (False, True)]
    return A + B + C


def subject_data(subject: int, args: argparse.Namespace) -> dict:
    X, ids = load_train(subject, args.root)
    n_raw = int(round(args.max_seconds * SFREQ))
    eeg = decimate(X[..., :n_raw])
    F = load_features(args.features, args.frames, ids, args.feature_dir)
    window = slice(*args.window)
    enc_alpha = float(DEFAULT_ALPHAS[retrieval_cv_curve(F, eeg, DEFAULT_ALPHAS, window).argmin()])
    curve = nested_cv_scores(F, eeg, DEFAULT_ALPHAS, enc_alpha, powers=(), window=window, inputs=("whitened",))
    dec_alpha = float(DEFAULT_ALPHAS[curve["whitened"].argmin()])
    return {"eeg": eeg, "F": F, "enc_alpha": enc_alpha, "dec_alpha": dec_alpha, "window": window}


def row_z(sim: np.ndarray) -> np.ndarray:
    return (sim - sim.mean(axis=1, keepdims=True)) / sim.std(axis=1, keepdims=True)


def run_cv(args: argparse.Namespace) -> None:
    configs = grid(args.grid)
    rows = []
    for subject in args.subjects:
        d = subject_data(subject, args)
        for r in cv_scores(d["F"], d["eeg"], configs, d["enc_alpha"], d["dec_alpha"], d["window"], device=args.device,
                           ensemble_weights=tuple(args.ensemble_weights)):
            rows.append({"subject": subject, "enc_alpha": d["enc_alpha"], "dec_alpha": d["dec_alpha"], **r})
        torch.cuda.empty_cache()
        print(f"sub-{subject:02d} done", flush=True)
    df = pd.DataFrame(rows)
    args.cv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.cv_file, index=False)
    summary = (df.groupby("config")[["cv_score", "exp_top1", "exp_top5", "mean_steps"]].mean()
                 .sort_values("cv_score"))
    beats = df.pivot(index="config", columns="subject", values="cv_score")
    summary["subjects_better_than_ridge"] = (beats.lt(beats.loc["ridge"], axis=1)).sum(axis=1)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary.round(4))
    print(f"wrote {args.cv_file}")


def run_dev(args: argparse.Namespace) -> None:
    configs = {c.name: c for g in ("main", "strong_wd") for c in grid(g)}
    names = args.configs
    if names == ["best"]:
        df = pd.read_csv(args.cv_file)
        names = [df[df.config != "ridge"].groupby("config").cv_score.mean().idxmin()]
        names = [n for n in names if not n.startswith("ens")] or names
    out_rows, ranks, ref_ranks, ens_ranks = [], {}, [], {}
    for subject in args.subjects:
        d = subject_data(subject, args)
        measured = decimate(load_test(subject, args.root)[..., : int(round(args.max_seconds * SFREQ))])
        F_tst = load_features(args.features, args.frames, TEST_IDS, args.feature_dir)
        true_ids = load_test_labels(subject, args.root)
        ref = DecodingVariant(d["dec_alpha"], True, None, d["enc_alpha"], d["window"]).fit(d["F"], d["eeg"])
        ref_sim = ref.similarity(measured, F_tst)
        W = noise_whiteners(d["F"], d["eeg"], d["enc_alpha"], d["window"], device=args.device)
        X_trn, X_tst, V_trn, V_tst = prepare(d["eeg"], measured, d["F"], F_tst, W, d["window"],
                                             args.device or gpu.default_device())
        if true_ids is not None:
            ref_ranks.append(true_ranks(ref_sim, true_ids))
        for name in names:
            for seed in args.seeds:
                head = train_head(X_trn, V_trn, configs[name], d["dec_alpha"], seed=seed)
                sim = head.similarity(X_tst, V_tst).cpu().numpy().astype(np.float64)
                exp = f"con_{args.features}_f{args.frames}_{name}_s{seed}"
                if seed == args.seeds[0]:
                    path = Path("results/similarity") / exp / f"sub-{subject:02d}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, sim)
                    write_submission(sim, subject, Path("submissions") / exp)
                row = {"subject": subject, "config": name, "seed": seed, "steps": head.steps,
                       "tau": float(head.log_tau.exp())}
                if true_ids is not None:
                    row.update(evaluate(sim, true_ids))
                    ranks.setdefault((name, seed), []).append(true_ranks(sim, true_ids))
                    ens_ranks.setdefault((name, seed), []).append(
                        true_ranks(args.ens_weight * row_z(ref_sim) + (1 - args.ens_weight) * row_z(sim), true_ids))
                out_rows.append(row)
        print(f"sub-{subject:02d} done", flush=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(args.dev_file, index=False)
    if not ranks:
        return
    base = np.concatenate(ref_ranks)
    print(f"M8a reference: top1={np.mean(base <= 1):.3f} top5={np.mean(base <= 5):.3f}")
    for name in names:
        per_seed = df[(df.config == name) & df.top1.notna()].groupby("seed")[["top1", "top5", "median_rank"]].mean()
        m, s = per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else per_seed.mean() * 0
        print(f"{name}: top1 {m.top1*100:.1f} ± {s.top1*100:.1f} | top5 {m.top5*100:.1f} ± {s.top5*100:.1f} | "
              f"medR {m.median_rank:.2f} ± {s.median_rank:.2f} (over seeds {list(per_seed.index)})")
        for seed in args.seeds:
            r, e = np.concatenate(ranks[(name, seed)]), np.concatenate(ens_ranks[(name, seed)])
            print(f"  seed {seed}: vs M8a better {np.sum(r < base)} / worse {np.sum(r > base)}, "
                  f"p={wilcoxon(base, r).pvalue:.3g} | ensemble (w_M8a={args.ens_weight:g}) top1={np.mean(e <= 1):.3f} "
                  f"top5={np.mean(e <= 5):.3f}, vs M8a better {np.sum(e < base)} / worse {np.sum(e > base)}, "
                  f"p={wilcoxon(base, e).pvalue:.3g}")
    print(f"wrote {args.dev_file}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "dev"], default="cv")
    p.add_argument("--features", default="clip_b32")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--grid", choices=["main", "selected", "strong_wd"], default="main", help="cv mode: which config grid")
    p.add_argument("--configs", nargs="+", default=["best"], help="dev mode: config names, or 'best' (by train-CV)")
    p.add_argument("--ensemble-weights", type=float, nargs="*", default=[],
                   help="cv mode: also score w * rowz(ridge) + (1 - w) * rowz(head) for these w")
    p.add_argument("--ens-weight", type=float, default=0.5, help="dev mode: weight of M8a in the ensemble diagnostic")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--cv-file", type=Path, default=None, help="default: results/contrastive_cv_<features>.csv")
    p.add_argument("--dev-file", type=Path, default=None, help="default: results/contrastive_dev_<features>.csv")
    args = p.parse_args()
    suffix = ("" if args.grid == "main" else f"_{args.grid}") + ("_ens" if args.ensemble_weights else "")
    args.cv_file = args.cv_file or Path(f"results/contrastive_cv_{args.features}{suffix}.csv")
    args.dev_file = args.dev_file or Path(f"results/contrastive_dev_{args.features}.csv")
    torch.set_num_threads(8)
    run_cv(args) if args.mode == "cv" else run_dev(args)


if __name__ == "__main__":
    main()
