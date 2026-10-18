"""THINGS-EEG2 Step 0: does a decoder trained on THINGS-EEG2 (image RSVP) transfer zero-shot to our video EEG?

Pipeline (all external data were converted once to plain float32 .npz by ``python -I src/eegmr/external.py convert ...``):

    embed   CLIP B-32 (open_clip ViT-B-32 / laion2b_s34b_b79k) image embeddings of the THINGS training and test images
    fit     whitened-decoder-style ridge, THINGS EEG (17 ch x 50-600 ms, 935 values) -> standardized CLIP vector;
            alpha chosen on the held-out THINGS test images (disjoint from training images)
    ours    apply it unchanged to OUR EEG (no fitting on our data) and rank videos by cosine to their CLIP vectors, scored on
            (a) our TRAINING recordings, candidates = the subject's training videos (never used for fitting here), expected
            top-1 / top-5 of 102 (hypergeometric) pooled over subjects, and (b) the dev test sets (sub-01..04), top-1/5/median.

Our test recordings are used only for the final scoring in (b); nothing is estimated from them.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

from eegmr import external

EXT = Path("/home/kparpie/ext_data/things_eeg2")
SHA256 = {  # as published by the OSF API (files in component y63gw)
    "training_images.zip": "dc9cc627141ba230641f729f84b80eb01f0a963d288c971eefcf76b4a7581f3a",
    "test_images.zip": "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a",
    "image_metadata.npy": "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c",
    "sub-01.zip": "f25d5ea30ba6811edd7ebd460d7b271edaf527b6920ab2d3d6e7950e9ee9639f",
    "sub-02.zip": "e1eba2d74009c6dd41becf53d84b8d47532103b41110a5855e7e286bcca6fe41",
}


def cmd_verify(args: argparse.Namespace) -> None:
    for name, expected in SHA256.items():
        path = EXT / name
        if path.exists():
            got = external.sha256(path)
            print(f"{name}: {path.stat().st_size} bytes, sha256 {'OK' if got == expected else 'MISMATCH ' + got}")


def _extract_images(zip_name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EXT / zip_name) as zf:
        for info in zf.infolist():
            assert not info.filename.startswith("/") and ".." not in Path(info.filename).parts, info.filename
        zf.extractall(out)


def cmd_embed(args: argparse.Namespace) -> None:
    import open_clip
    import torch
    from PIL import Image

    meta = np.load(EXT / "image_metadata.npz", allow_pickle=False)
    print("metadata arrays:", {k: meta[k].shape for k in meta.files})
    for zip_name, folder in (("training_images.zip", "train"), ("test_images.zip", "test")):
        if not (EXT / "images" / folder).exists():
            _extract_images(zip_name, EXT / "images" / folder)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(args.device)
    out = {}
    for split in ("train", "test"):
        files = meta[f"{split}_img_files"]
        paths = {p.name: p for p in (EXT / "images" / split).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        print(split, len(files), "images in metadata;", len(paths), "files on disk")
        embs = []
        with torch.no_grad():
            for i in range(0, len(files), 128):
                batch = torch.stack([preprocess(Image.open(paths[str(f)]).convert("RGB")) for f in files[i:i + 128]])
                embs.append(model.encode_image(batch.to(args.device)).float().cpu().numpy())
        out[split] = np.concatenate(embs)
        print(split, out[split].shape)
    np.savez(EXT / "clip_b32_things.npz", **out)


# ------------------------------------------------------------------------------------------------ fit / evaluate
def prep_x(eeg: np.ndarray, car: bool) -> np.ndarray:
    """(N, 17, T) -> (N, 17 * T); ``car`` = subtract the mean over channels at each time point (common-average reference)."""
    if car:
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
    return eeg.reshape(len(eeg), -1)


def zstats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = x.std(axis=0)
    return x.mean(axis=0), np.where(sd == 0, 1.0, sd)


def ridge_path(X: np.ndarray, Y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Ridge solutions ``(len(alphas), D, K)`` for standardized X (n, D), centred Y (n, K); alphas relative to n."""
    n = len(X)
    C = (X.T.astype(np.float64) @ X) / n
    B = (X.T.astype(np.float64) @ Y) / n
    lam, Q = np.linalg.eigh(C)
    QtB = Q.T @ B
    return np.stack([Q @ (QtB / (lam + a)[:, None]) for a in alphas])


def retrieval(pred: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """1-based rank of the true candidate (the diagonal) for cosine similarity ``pred (n, K)`` vs ``cand (n, K)``."""
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    c = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = p @ c.T
    return (s >= np.diag(s)[:, None]).sum(axis=1)


def expected_topk(ranks: np.ndarray, pool: int, k: int, n_candidates: int = 102) -> float:
    from scipy.stats import hypergeom

    return float(np.mean(hypergeom.cdf(k - 1, pool - 1, ranks - 1, n_candidates - 1)))


def load_things(subjects: list[int], car: bool) -> dict:
    """Aligned THINGS data for the chosen subjects: standardization statistics from the training trials."""
    clip = np.load(EXT / "clip_b32_things.npz", allow_pickle=False)
    trn_x, trn_y, tst = [], [], []
    for s in subjects:
        z = external.load_converted(EXT / "converted" / f"sub-{s:02d}.npz")
        a = external.things_to_ours(z["train"], z["ch_names"])  # (cond, rep, 17, 55)
        t = external.things_to_ours(z["test"], z["ch_names"])
        trn_x.append(prep_x(a.reshape(-1, *a.shape[-2:]), car))
        trn_y.append(np.repeat(clip["train"][None], a.shape[1], axis=0).transpose(1, 0, 2).reshape(-1, clip["train"].shape[1]))
        tst.append(prep_x(t.reshape(-1, *t.shape[-2:]), car).reshape(*t.shape[:2], -1))
    X = np.concatenate(trn_x)
    mu, sd = zstats(X)
    y_mu, y_sd = zstats(clip["train"])
    return {"X": (X - mu) / sd, "Y": (np.concatenate(trn_y) - y_mu) / y_sd, "tst": tst, "x_stats": (mu, sd),
            "y_stats": (y_mu, y_sd), "clip_test": (clip["test"] - y_mu) / y_sd}


def things_score(W: np.ndarray, d: dict) -> dict:
    """Held-out THINGS images (200): single-trial and 80-repetition-average retrieval among the 200 images."""
    mu, sd = d["x_stats"]
    single, avg = [], []
    for tst in d["tst"]:  # (200, 80, 935)
        x = (tst - mu) / sd
        for rep in range(0, x.shape[1], 8):  # a subset of repetitions is enough
            single.append(retrieval(x[:, rep] @ W, d["clip_test"]))
        avg.append(retrieval(x.mean(axis=1) @ W, d["clip_test"]))
    s, a = np.concatenate(single), np.concatenate(avg)
    return {"single_pct": float(np.mean((s - 1) / 199)), "single_top1": float(np.mean(s <= 1)), "single_top5": float(np.mean(s <= 5)),
            "avg_pct": float(np.mean((a - 1) / 199)), "avg_top1": float(np.mean(a <= 1)), "avg_top5": float(np.mean(a <= 5))}


ALPHAS = np.logspace(-3, 2, 11)  # relative to n (per-sample covariance units)


def cmd_fit(args: argparse.Namespace) -> None:
    out = {}
    for car in (False, True):
        d = load_things(args.subjects, car)
        paths = ridge_path(d["X"], d["Y"] - d["Y"].mean(axis=0), ALPHAS)
        print(f"\ncar={car}: X {d['X'].shape}, held-out THINGS images (200 way):")
        best, best_score = 0, 9.0
        for i, a in enumerate(ALPHAS):
            sc = things_score(paths[i], d)
            print(f"  alpha/n={a:8.3g}  single-trial pct {sc['single_pct']:.4f} top1 {sc['single_top1']:.3f} top5 {sc['single_top5']:.3f}"
                  f" | 80-rep avg pct {sc['avg_pct']:.4f} top1 {sc['avg_top1']:.3f} top5 {sc['avg_top5']:.3f}")
            if sc["single_pct"] < best_score:
                best, best_score = i, sc["single_pct"]
        print(f"  chosen alpha/n = {ALPHAS[best]:.3g} (best single-trial percentile rank on the held-out THINGS images)")
        out[f"W_car{int(car)}"] = paths.astype(np.float32)
        out[f"best_car{int(car)}"] = np.array(best)
        out[f"x_mu_car{int(car)}"], out[f"x_sd_car{int(car)}"] = d["x_stats"]
        out[f"y_mu_car{int(car)}"], out[f"y_sd_car{int(car)}"] = d["y_stats"]
    np.savez(EXT / f"ridge_{'_'.join(map(str, args.subjects))}.npz", alphas=ALPHAS, **out)


def zero_shot_similarity(eeg_trn: np.ndarray, eeg_new: np.ndarray, feats_trn: np.ndarray, feats_new: np.ndarray,
                         window: slice = slice(5, 60), car: bool = True, subjects: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Cosine similarity ``(n_eeg_new, n_videos_new)`` from the THINGS-trained ridge, applied without fitting on our data.

    Our EEG window is common-average-referenced (if ``car``) and standardized with the statistics of ``eeg_trn`` (training
    recordings only); candidate video features are standardized with ``feats_trn`` statistics."""
    r = np.load(EXT / f"ridge_{'_'.join(map(str, subjects))}.npz", allow_pickle=False)
    W = r[f"W_car{int(car)}"][int(r[f"best_car{int(car)}"])]
    x_trn, x_new = (prep_x(e[..., window], car) for e in (eeg_trn, eeg_new))
    mu, sd = zstats(x_trn)
    f_mu, f_sd = zstats(feats_trn)
    pred = ((x_new - mu) / sd) @ W
    cand = (feats_new - f_mu) / f_sd
    return (pred / np.linalg.norm(pred, axis=1, keepdims=True)) @ (cand / np.linalg.norm(cand, axis=1, keepdims=True)).T


def cmd_ours(args: argparse.Namespace) -> None:
    from eegmr.data import DEFAULT_ROOT, TEST_IDS, load_test, load_test_labels, load_train
    from eegmr.decoding import decoder_input, noise_whiteners
    from eegmr.evaluate import evaluate, true_ranks
    from eegmr.features import load_features
    from eegmr.linear import DEFAULT_ALPHAS, retrieval_cv_curve
    from predict import preprocess

    r = np.load(EXT / f"ridge_{'_'.join(map(str, args.subjects))}.npz", allow_pickle=False)
    window = slice(5, 60)
    rows = []
    for s in args.eval_subjects:
        X, ids = load_train(s, DEFAULT_ROOT)
        eeg = preprocess(X, 1.5)
        feats = load_features("clip_b32", 8, ids, Path("features"))
        f_mu, f_sd = zstats(feats)
        measured = preprocess(load_test(s, DEFAULT_ROOT), 1.5)
        F_tst = load_features("clip_b32", 8, TEST_IDS, Path("features"))
        enc_alpha = float(DEFAULT_ALPHAS[retrieval_cv_curve(feats, eeg, DEFAULT_ALPHAS, window).argmin()])
        wh = noise_whiteners(feats, eeg, enc_alpha, window)
        for car in (False, True):
            for whiten in (False, True):
                def xin(e):
                    x = decoder_input(e - e.mean(axis=1, keepdims=True) if car else e, window, wh if whiten else None).cpu().numpy()
                    return x
                x_trn, x_tst = xin(eeg), xin(measured)
                mu, sd = zstats(x_trn)
                for ai in sorted({int(r[f"best_car{int(car)}"]) + o for o in (-2, 0, 2)}):
                    W = r[f"W_car{int(car)}"][ai]
                    cand_train = (feats - f_mu) / f_sd
                    pred_train = ((x_trn - mu) / sd) @ W
                    rk = retrieval(pred_train, cand_train)
                    n = len(rk)
                    row = {"subject": s, "car": car, "whiten": whiten, "alpha_rel": ALPHAS[ai], "n_train": n,
                           "pct": float(np.mean((rk - 1) / (n - 1))), "e1": expected_topk(rk, n, 1), "e5": expected_topk(rk, n, 5)}
                    true_ids = load_test_labels(s, DEFAULT_ROOT)
                    if true_ids is not None:
                        pred_t = ((x_tst - mu) / sd) @ W
                        c = (F_tst - f_mu) / f_sd
                        p = pred_t / np.linalg.norm(pred_t, axis=1, keepdims=True)
                        sim = p @ (c / np.linalg.norm(c, axis=1, keepdims=True)).T
                        sc = evaluate(sim, true_ids)
                        row |= {"dev_top1": sc["top1"], "dev_top5": sc["top5"], "dev_median": sc["median_rank"]}
                    rows.append(row)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(Path("results") / f"things_step0_ours_{'_'.join(map(str, args.subjects))}.csv", index=False)
    g = df.groupby(["car", "whiten", "alpha_rel"])
    summary = g.agg(pct=("pct", "mean"), e1=("e1", "mean"), e5=("e5", "mean"), dev_top1=("dev_top1", "mean"),
                    dev_top5=("dev_top5", "mean"), dev_median=("dev_median", "mean"))
    print("\nmean over subjects (train-EEG pool metrics over all subjects; dev over sub-01..04). Chance: pct 0.5, e1 0.0098, e5 0.049")
    print(summary.round(4).to_string())


def cmd_prior(args: argparse.Namespace) -> None:
    """Training-only 5-fold CV of a per-subject ridge shrunk toward the THINGS solution (W0 scaled by ``scale``;
    scale 0 = the ordinary ridge). Same folds and score as every other train-CV (seed-0 KFold, expected top-k of 102)."""
    from sklearn.model_selection import KFold

    from eegmr.data import DEFAULT_ROOT, load_train
    from eegmr.features import load_features
    from predict import preprocess

    r = np.load(EXT / f"ridge_{'_'.join(map(str, args.subjects))}.npz", allow_pickle=False)
    W0 = r["W_car1"][int(r["best_car1"])].astype(np.float64)
    scales, rel_alphas = (0.0, 0.5, 1.0, 2.0), np.logspace(-1, 3, 9)
    rows = []
    for s in args.eval_subjects:
        X, ids = load_train(s, DEFAULT_ROOT)
        eeg = preprocess(X, 1.5)[..., 5:60]
        F = load_features("clip_b32", 8, ids, Path("features"))
        x_all = prep_x(eeg, True)
        acc = {(sc, a): [] for sc in scales for a in rel_alphas}
        for trn, val in KFold(5, shuffle=True, random_state=0).split(F):
            mu, sd = zstats(x_all[trn])
            fm, fs = zstats(F[trn])
            Xt, Xv = (x_all[trn] - mu) / sd, (x_all[val] - mu) / sd
            Yt, Yv = (F[trn] - fm) / fs, (F[val] - fm) / fs
            n = len(Xt)
            C, B = Xt.T.astype(np.float64) @ Xt / n, Xt.T.astype(np.float64) @ (Yt - Yt.mean(0)) / n
            lam, Q = np.linalg.eigh(C)
            for sc in scales:
                R = B - C @ (sc * W0)
                QtR = Q.T @ R
                for a in rel_alphas:
                    W = sc * W0 + Q @ (QtR / (lam + a)[:, None])
                    rk = retrieval(Xv @ W, Yv)
                    acc[(sc, a)].append((rk, len(val)))
        for (sc, a), parts in acc.items():
            rk = np.concatenate([p for p, _ in parts])
            pct = float(np.mean(np.concatenate([(p - 1) / (m - 1) for p, m in parts])))
            e1 = float(np.average([expected_topk(p, m, 1) for p, m in parts], weights=[len(p) for p, _ in parts]))
            e5 = float(np.average([expected_topk(p, m, 5) for p, m in parts], weights=[len(p) for p, _ in parts]))
            rows.append({"subject": s, "scale": sc, "alpha_rel": a, "pct": pct, "e1": e1, "e5": e5})
        print("done sub", s, flush=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(Path("results") / "things_prior_cv.csv", index=False)
    best = df.loc[df.groupby(["subject", "scale"]).pct.idxmin()]  # alpha picked per subject and scale by the training CV
    print("\nbest alpha per subject (CV percentile rank), mean over subjects:")
    print(best.groupby("scale")[["pct", "e1", "e5"]].mean().round(4).to_string())
    print("\nper subject pct (scale 0 vs 1):")
    print(best.pivot(index="subject", columns="scale", values="pct").round(4).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["verify", "embed", "fit", "ours", "prior"])
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2], help="THINGS-EEG2 subjects used for fitting")
    p.add_argument("--eval-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="our subjects to score")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    {"verify": cmd_verify, "embed": cmd_embed, "fit": cmd_fit, "ours": cmd_ours, "prior": cmd_prior}[args.cmd](args)


if __name__ == "__main__":
    main()
