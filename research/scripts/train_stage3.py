"""Stage 3 feasibility: nonlinear EEG encoders with InfoNCE, per subject (see ``eegmr.stage3``).

Modes:
    cv     train-CV (outer 5-fold, KFold seed 0) of a rung's config grid plus the ridge decoder (M8a) on the same
           folds, with optional blends with M8a; ranked by mean train-CV over subjects. No test labels.
    dev    final encoders per subject and seed: dev top-1 / top-5 / median rank, paired Wilcoxon vs M8a and vs the
           M12 two-tower head, and blends M8a + model and M8a + M12 two-tower + model.
    curve  learning curves: train-CV of the ridge decoder and the given configs on 50 / 75 / 100% of each outer
           training part.

Examples:
    python scripts/train_stage3.py --rung 1
    python scripts/train_stage3.py --mode dev --configs mlp512_do0.5_lr0.001_wd0.01_w5-60 --seeds 0 1 2
    python scripts/train_stage3.py --mode curve --configs linear_lr0.001_wd0.01_w5-60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from eegmr import gpu
from eegmr.contrastive import ContrastiveConfig, prepare, train_head
from eegmr.data import DEFAULT_ROOT, SFREQ, TEST_IDS, decimate, load_test, load_test_labels, load_train
from eegmr.decoding import DecodingVariant, nested_cv_scores, noise_whiteners
from eegmr.evaluate import evaluate, true_ranks
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.linear import DEFAULT_ALPHAS, retrieval_cv_curve
from eegmr.stage3 import Stage3Config, cv_scores, inputs, train_encoder

M12_TWO_TOWER = ContrastiveConfig("two_tower", dim=256, lr=1e-3, weight_decay=1e-2)


def rung(n: int) -> list[Stage3Config]:
    """Config grid per rung. 0: linear reference; 1: MLPs on the M12 input; 2: small conv nets on the time course;
    3: rung 2 with augmentation."""
    if n == 0:
        return [Stage3Config("linear", lr=1e-3, weight_decay=1e-2)]
    if n == 1:
        return ([Stage3Config("linear", lr=1e-3, weight_decay=1e-2)]
                + [Stage3Config("mlp", dropout=do, lr=lr) for do in (0.3, 0.5) for lr in (3e-4, 1e-3)]
                + [Stage3Config("resmlp", dropout=0.5, lr=lr) for lr in (3e-4, 1e-3)])
    conv = [Stage3Config(m, dropout=do, lr=lr, window=w) for m in ("eegnet", "shallow") for do in (0.25, 0.5)
            for lr in (1e-3, 3e-3) for w in ((0, 60), (0, 80))]
    if n == 2:
        return conv
    if n == 3:
        return [Stage3Config(c.model, dropout=c.dropout, lr=c.lr, window=c.window, augment=True) for c in conv]
    raise ValueError(n)


def window_controls() -> list[Stage3Config]:
    """Non-conv heads on the conv window (0-800 ms), to separate "conv architecture" from "longer window"."""
    return [Stage3Config("linear", lr=1e-3, weight_decay=1e-2, window=(0, 80)),
            Stage3Config("mlp", dropout=0.5, lr=1e-3, window=(0, 80))]


def all_configs() -> dict[str, Stage3Config]:
    return {c.name: c for c in [*window_controls(), *(c for n in range(4) for c in rung(n))]}


def subject_data(subject: int, args: argparse.Namespace) -> dict:
    X, ids = load_train(subject, args.root)
    n_raw = int(round(args.max_seconds * SFREQ))
    eeg = decimate(X[..., :n_raw])
    F = load_features(args.features, args.frames, ids, args.feature_dir)
    w = slice(5, 60)
    enc_alpha = float(DEFAULT_ALPHAS[retrieval_cv_curve(F, eeg, DEFAULT_ALPHAS, w).argmin()])
    curve = nested_cv_scores(F, eeg, DEFAULT_ALPHAS, enc_alpha, powers=(), window=w, inputs=("whitened",))
    return {"eeg": eeg, "F": F, "enc_alpha": enc_alpha, "dec_alpha": float(DEFAULT_ALPHAS[curve["whitened"].argmin()])}


def summarize(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    s = df.groupby(group)[["cv_score", "exp_top1", "exp_top5", "mean_steps"]].mean()
    if "ridge" in df.config.values:
        piv = df.pivot_table(index=group, columns="subject", values="cv_score")
        ridge = df[df.config == "ridge"].set_index("subject").cv_score
        if group == ["config"]:
            s["n_better_than_ridge"] = piv.lt(ridge, axis=1).sum(axis=1)
    return s.sort_values("cv_score")


def run_cv(args: argparse.Namespace, configs: list[Stage3Config], fractions: list[float], out: Path) -> None:
    rows = []
    for subject in args.subjects:
        d = subject_data(subject, args)
        for frac in fractions:
            for r in cv_scores(d["F"], d["eeg"], configs, d["enc_alpha"], d["dec_alpha"], device=args.device,
                               ensemble_weights=tuple(args.ensemble_weights), train_fraction=frac):
                rows.append({"subject": subject, **r})
        print(f"sub-{subject:02d} done", flush=True)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(summarize(df, ["config"] if len(fractions) == 1 else ["config", "train_fraction"]).round(4))
    print(f"wrote {out}")


def run_dev(args: argparse.Namespace) -> None:
    cfgs = all_configs()
    rows, ranks = [], {}
    for subject in args.subjects:
        d = subject_data(subject, args)
        n_raw = int(round(args.max_seconds * SFREQ))
        measured = decimate(load_test(subject, args.root)[..., :n_raw])
        F_tst = load_features(args.features, args.frames, TEST_IDS, args.feature_dir)
        true_ids = load_test_labels(subject, args.root)
        dev = args.device or gpu.default_device()
        ref = DecodingVariant(d["dec_alpha"], True, None, d["enc_alpha"], slice(5, 60)).fit(d["F"], d["eeg"])
        sims = {"M8a": ref.similarity(measured, F_tst)}
        W = noise_whiteners(d["F"], d["eeg"], d["enc_alpha"], slice(5, 60), device=dev)
        X_trn, X_tst, V_trn, V_tst = prepare(d["eeg"], measured, d["F"], F_tst, W, slice(5, 60), dev)
        tt = [train_head(X_trn, V_trn, M12_TWO_TOWER, d["dec_alpha"], seed=s).similarity(X_tst, V_tst).cpu().numpy()
              for s in args.seeds]
        for name in args.configs:
            cfg = cfgs[name]
            Wc = noise_whiteners(d["F"], d["eeg"], d["enc_alpha"], slice(*cfg.window), device=dev)
            Xc_trn, Xc_tst, Vc_trn, Vc_tst = inputs(d["eeg"], measured, d["F"], F_tst, Wc, slice(*cfg.window), dev)
            for i, seed in enumerate(args.seeds):
                model = train_encoder(Xc_trn, Vc_trn, cfg, seed=seed)
                sims[(name, seed)] = model.similarity(Xc_tst, Vc_tst).cpu().numpy().astype(np.float64)
                sims[("M12tt", seed)] = tt[i].astype(np.float64)
                row = {"subject": subject, "config": name, "seed": seed, "steps": model.steps}
                if true_ids is not None:
                    row.update(evaluate(sims[(name, seed)], true_ids))
                rows.append(row)
        if true_ids is not None:
            z = lambda s: (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)  # noqa: E731
            ranks.setdefault("M8a", []).append(true_ranks(sims["M8a"], true_ids))
            for key, s in sims.items():
                if key == "M8a":
                    continue
                ranks.setdefault(key, []).append(true_ranks(s, true_ids))
                if key[0] != "M12tt":
                    m12 = sims[("M12tt", key[1])]
                    ranks.setdefault(("ens2",) + key, []).append(
                        true_ranks(0.75 * z(sims["M8a"]) + 0.25 * z(s), true_ids))
                    ranks.setdefault(("ens3",) + key, []).append(
                        true_ranks(0.6 * z(sims["M8a"]) + 0.2 * z(m12) + 0.2 * z(s), true_ids))
                else:
                    ranks.setdefault(("ens2",) + key, []).append(
                        true_ranks(0.75 * z(sims["M8a"]) + 0.25 * z(s), true_ids))
        print(f"sub-{subject:02d} done", flush=True)
    pd.DataFrame(rows).to_csv(args.dev_file, index=False)
    base = np.concatenate(ranks["M8a"])

    def line(label: str, key) -> str:
        r = np.concatenate(ranks[key])
        p = wilcoxon(base, r).pvalue if np.any(r != base) else 1.0
        return (f"{label:58s} top1={np.mean(r <= 1)*100:5.1f} top5={np.mean(r <= 5)*100:5.1f} "
                f"medR={np.median(r):4.1f} | vs M8a {np.sum(r < base)}/{np.sum(r > base)} p={p:.2g}")

    print(line("M8a", "M8a"))
    for seed in args.seeds:
        print(line(f"M12 two-tower s{seed}", ("M12tt", seed)))
        print(line(f"  0.75 M8a + 0.25 M12tt s{seed}", ("ens2", "M12tt", seed)))
    for name in args.configs:
        for seed in args.seeds:
            print(line(f"{name} s{seed}", (name, seed)))
            print(line(f"  0.75 M8a + 0.25 model", ("ens2", name, seed)))
            print(line(f"  0.6 M8a + 0.2 M12tt + 0.2 model", ("ens3", name, seed)))
    print(f"wrote {args.dev_file}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "dev", "curve"], default="cv")
    p.add_argument("--rung", type=int, default=1, help="cv mode: which config grid (0-3)")
    p.add_argument("--configs", nargs="+", default=[], help="config names (dev / curve; cv: instead of --rung)")
    p.add_argument("--features", default="clip_b32")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--ensemble-weights", type=float, nargs="*", default=[0.75])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--dev-file", type=Path, default=None)
    p.add_argument("--tag", default="", help="cv mode with --configs: output name results/stage3_cv_<tag>_<features>.csv")
    args = p.parse_args()
    torch.set_num_threads(8)
    tag = args.features
    if args.mode == "cv" and args.configs:  # an explicit config list instead of a rung grid
        cfgs = all_configs()
        run_cv(args, [cfgs[n] for n in args.configs], [1.0], Path(f"results/stage3_cv_{args.tag or 'selected'}_{tag}.csv"))
    elif args.mode == "cv":
        run_cv(args, rung(args.rung), [1.0], Path(f"results/stage3_cv_rung{args.rung}_{tag}.csv"))
    elif args.mode == "curve":
        cfgs = all_configs()
        run_cv(args, [cfgs[n] for n in args.configs], args.fractions, Path(f"results/stage3_curve_{tag}.csv"))
    else:
        args.dev_file = args.dev_file or Path(f"results/stage3_dev_{tag}.csv")
        run_dev(args)


if __name__ == "__main__":
    main()
