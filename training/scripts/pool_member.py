"""Subject-pooled whitened decoder (``pdecw:<features>[:frames]``): build its similarity caches for scripts/ensemble.py.

For target subject t: its own training trials (weight 1) plus the other five subjects' training trials (weight lam). Each
subject's EEG window is noise-whitened with ITS OWN whiteners (Ledoit-Wolf on its encoding OOF residuals, as decw) and
standardized with its own statistics. Targets are the video features standardized with the target fold's training-video
statistics. Weighted ridge with a weighted-mean intercept (rows scaled by sqrt(w)), cosine against the standardized
candidates. Zero-shot CV: in every outer fold the other subjects' recordings of t's held-out videos are dropped.

Selection (training-only, as ``ensemble.member_alphas`` / ``decoding.nested_cv_scores``): per subject, (lam, alpha) on the
grid LAMS x DEFAULT_ALPHAS by mean percentile rank over the 5 folds (target whiteners re-estimated per fold, no
centering). lam = 0 is decw itself: its fold / test similarities are produced by ``DecodingVariant`` (exact), and the
pooled implementation is asserted to match it at lam = 0. Caches: ``<cache>/sub-XX_pdecw-<features>_f<n>_t1.5_w5-60_ctr_{cv,test}.npz``
(``alphas`` = [alpha, enc_alpha], ``f0..f4`` / ``sim``, plus ``lam``), i.e. ensemble.py's --center-test format.

Example:
    PYTHONPATH=src:scripts python scripts/pool_member.py --features dinov2_b_patch_all --frames 8 \\
        --decw-cache /home/kparpie/Krabiki-matching/results/similarity/_cache --cache results/similarity/_cache_pool
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from eegmr.data import TEST_IDS, load_test, load_train
from eegmr.decoding import DecodingVariant, decoder_input, noise_whiteners
from eegmr.features import load_features
from eegmr.linear import DEFAULT_ALPHAS
from ensemble import center_like, preprocess

LAMS = (0.0, 0.05, 0.1, 0.2, 0.5)
WINDOW, MAX_SECONDS = slice(5, 60), 1.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # set by --device


def t64(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=DEV)


def std_stats(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sd = X.std(0, unbiased=False)
    return X.mean(0), torch.where(sd == 0, torch.ones_like(sd), sd)


def pooled_predict(Xc: torch.Tensor, Yc: torch.Tensor, w: torch.Tensor, Xnew: torch.Tensor, alphas) -> torch.Tensor:
    """Weighted ridge with weighted-mean intercept for every alpha: (n_alpha, n_new, D)."""
    xm = (w[:, None] * Xc).sum(0) / w.sum()
    ym = (w[:, None] * Yc).sum(0) / w.sum()
    sw = w.sqrt()[:, None]
    U, S, Vh = torch.linalg.svd((Xc - xm) * sw, full_matrices=False)
    UtY = U.T @ ((Yc - ym) * sw)
    shrink = S / (S**2 + t64(alphas)[:, None])
    return torch.einsum("vr,ar,rd->avd", (Xnew - xm) @ Vh.T, shrink, UtY) + ym


def cosine(P: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(P, dim=-1) @ torch.nn.functional.normalize(C, dim=-1).T


def pct_rank(sim: torch.Tensor) -> torch.Tensor:  # (..., n, n) -> sum over rows
    ranks = (sim >= sim.diagonal(dim1=-2, dim2=-1)[..., None]).sum(-1)
    return ((ranks - 1) / (sim.shape[-1] - 1)).sum(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--decw-cache", type=Path, required=True, help="ensemble cache with the decw member (alphas)")
    p.add_argument("--cache", type=Path, required=True, help="output cache directory")
    p.add_argument("--root", type=Path, default=Path("eeg-mind-reading-challenge"))
    p.add_argument("--feature-dir", type=Path, default=Path("features"))
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="target subjects to build")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--skip-existing", action="store_true", help="skip targets whose cv + test caches already exist")
    a = p.parse_args()
    global DEV
    DEV = a.device or DEV
    a.cache.mkdir(parents=True, exist_ok=True)
    key = lambda s, kind, head: a.decw_cache / f"sub-{s:02d}_{head}-{a.features}_f{a.frames}_t1.5_w5-60_ctr_{kind}.npz"  # noqa: E731
    subj = {}
    for s in range(1, 7):
        X, ids = load_train(s, a.root)
        eeg = preprocess(X, MAX_SECONDS)
        F = load_features(a.features, a.frames, ids, a.feature_dir)
        with np.load(key(s, "cv", "decw")) as z:
            dec_alpha, enc_alpha = (float(x) for x in z["alphas"])
        Wh = noise_whiteners(F, eeg, enc_alpha, WINDOW, device=DEV)  # full-training whiteners (used as "other" subject)
        Xs = decoder_input(eeg, WINDOW, Wh, DEV)
        mu, sd = std_stats(Xs)
        subj[s] = dict(ids=ids, eeg=eeg, F=F, Ft=t64(F), X=(Xs - mu) / sd, dec_alpha=dec_alpha, enc_alpha=enc_alpha)
    report = {}
    for t, st in subj.items():
        out = a.cache / f"sub-{t:02d}_pdecw-{a.features}_f{a.frames}_t1.5_w5-60_ctr"
        if t not in a.subjects or (a.skip_existing and Path(f"{out}_cv.npz").exists() and Path(f"{out}_test.npz").exists()):
            continue
        folds = list(KFold(5, shuffle=True, random_state=0).split(st["ids"]))
        score = torch.zeros(len(LAMS), len(DEFAULT_ALPHAS), dtype=torch.float64, device=DEV)
        fold_parts = []
        for trn, val in folds:
            held = set(st["ids"][val].tolist())
            Wf = noise_whiteners(st["F"][trn], st["eeg"][trn], st["enc_alpha"], WINDOW, device=DEV)
            Xt = decoder_input(st["eeg"][trn], WINDOW, Wf, DEV)
            mu, sd = std_stats(Xt)
            fm, fs = std_stats(st["Ft"][trn])
            parts = dict(trn=trn, val=val, Wf=Wf, mu=mu, sd=sd, fm=fm, fs=fs, Xt=(Xt - mu) / sd, Yt=(st["Ft"][trn] - fm) / fs,
                         C=(st["Ft"][val] - fm) / fs)
            others = [(o, np.array([v not in held for v in subj[o]["ids"]])) for o in subj if o != t]
            parts["Xo"] = torch.cat([subj[o]["X"][torch.as_tensor(m)] for o, m in others])
            parts["Yo"] = torch.cat([(subj[o]["Ft"][torch.as_tensor(m)] - fm) / fs for o, m in others])
            Xv = (decoder_input(st["eeg"][val], WINDOW, Wf, DEV) - mu) / sd  # selection: no centering (as member_alphas)
            for li, lam in enumerate(LAMS):
                Xc = parts["Xt"] if lam == 0 else torch.cat([parts["Xt"], parts["Xo"]])
                Yc = parts["Yt"] if lam == 0 else torch.cat([parts["Yt"], parts["Yo"]])
                w = torch.ones(len(Xc), dtype=torch.float64, device=DEV)
                w[len(trn):] = lam
                score[li] += pct_rank(cosine(pooled_predict(Xc, Yc, w, Xv, DEFAULT_ALPHAS), parts["C"]))
            fold_parts.append(parts)
        score = (score / len(st["ids"])).cpu().numpy()
        li, ai = np.unravel_index(score.argmin(), score.shape)
        lam, alpha = LAMS[li], float(DEFAULT_ALPHAS[ai])
        a0 = float(DEFAULT_ALPHAS[score[0].argmin()])
        if t == 1:  # lam = 0 of the pooled code is decw: same alpha choice and (numerically) the same predictions
            assert np.isclose(a0, st["dec_alpha"]), f"lam=0 alpha {a0} != decw alpha {st['dec_alpha']}"
            with np.load(key(1, "cv", "decw")) as z:  # the decw member's cached, centered fold-0 matrix
                ref = z["f0"]
            parts = fold_parts[0]
            w = torch.ones(len(parts["Xt"]), dtype=torch.float64, device=DEV)
            e_val = center_like(st["eeg"][parts["val"]], st["eeg"][parts["trn"]])
            Xv = (decoder_input(e_val, WINDOW, parts["Wf"], DEV) - parts["mu"]) / parts["sd"]
            mine = cosine(pooled_predict(parts["Xt"], parts["Yt"], w, Xv, [st["dec_alpha"]])[0], parts["C"]).cpu().numpy()
            diff = float(np.abs(mine - ref).max())
            same_rank = float((np.argsort(-mine, 1) == np.argsort(-ref, 1)).all(1).mean())
            assert diff < 1e-4, f"pooled(lam=0) vs cached decw fold 0: max abs diff {diff}"
            report["lam0_vs_decw"] = {"max_abs_diff": diff, "rank_identical_rows": same_rank}
            print(f"{a.features} lam=0 check (sub-01 fold 0 vs cached decw): max abs diff {diff:.2e}, "
                  f"rank-identical rows {same_rank:.3f}", flush=True)
        # cache fold similarities (center-test on the held-out fold, as ensemble.py --center-test)
        fsims = []
        for parts in fold_parts:
            trn, val = parts["trn"], parts["val"]
            e_val = center_like(st["eeg"][val], st["eeg"][trn])
            if lam == 0:
                dec = DecodingVariant(st["dec_alpha"], whiten_input=True, enc_alpha=st["enc_alpha"], window=WINDOW)
                fsims.append(dec.fit(st["F"][trn], st["eeg"][trn]).similarity(e_val, st["F"][val]))
                if t == 1 and len(fsims) == 1:
                    w = torch.ones(len(parts["Xt"]), dtype=torch.float64, device=DEV)
                    Xv = (decoder_input(e_val, WINDOW, parts["Wf"], DEV) - parts["mu"]) / parts["sd"]
                    mine = cosine(pooled_predict(parts["Xt"], parts["Yt"], w, Xv, [st["dec_alpha"]])[0], parts["C"]).cpu().numpy()
                    diff = float(np.abs(mine - fsims[-1]).max())
                    assert diff < 1e-4, f"pooled(lam=0) vs decw max abs diff {diff}"
                    report["lam0_vs_decw_max_abs_diff"] = diff
                continue
            Xc, Yc = torch.cat([parts["Xt"], parts["Xo"]]), torch.cat([parts["Yt"], parts["Yo"]])
            w = torch.ones(len(Xc), dtype=torch.float64, device=DEV)
            w[len(trn):] = lam
            Xv = (decoder_input(e_val, WINDOW, parts["Wf"], DEV) - parts["mu"]) / parts["sd"]
            fsims.append(cosine(pooled_predict(Xc, Yc, w, Xv, [alpha])[0], parts["C"]).cpu().numpy().astype(np.float64))
        # test similarity: target's full training + all of the others' training trials
        measured = center_like(preprocess(load_test(t, a.root), MAX_SECONDS), st["eeg"])
        F_tst = load_features(a.features, a.frames, TEST_IDS, a.feature_dir)
        if lam == 0:
            dec = DecodingVariant(st["dec_alpha"], whiten_input=True, enc_alpha=st["enc_alpha"], window=WINDOW)
            tsim = dec.fit(st["F"], st["eeg"]).similarity(measured, F_tst)
            alpha = st["dec_alpha"]
        else:
            Wt = noise_whiteners(st["F"], st["eeg"], st["enc_alpha"], WINDOW, device=DEV)
            Xt = decoder_input(st["eeg"], WINDOW, Wt, DEV)
            mu, sd = std_stats(Xt)
            fm, fs = std_stats(st["Ft"])
            Xc = torch.cat([(Xt - mu) / sd] + [subj[o]["X"] for o in subj if o != t])
            Yc = torch.cat([(st["Ft"] - fm) / fs] + [(subj[o]["Ft"] - fm) / fs for o in subj if o != t])
            w = torch.ones(len(Xc), dtype=torch.float64, device=DEV)
            w[len(Xt):] = lam
            Xn = (decoder_input(measured, WINDOW, Wt, DEV) - mu) / sd
            tsim = cosine(pooled_predict(Xc, Yc, w, Xn, [alpha])[0], (t64(F_tst) - fm) / fs).cpu().numpy().astype(np.float64)
        np.savez(f"{out}_cv.npz", alphas=np.array([alpha, st["enc_alpha"]]), lam=np.float64(lam),
                 **{f"f{f}": s_ for f, s_ in enumerate(fsims)})
        np.savez(f"{out}_test.npz", sim=tsim)
        report[f"sub-{t:02d}"] = {"lam": lam, "alpha": alpha, "cv_pct_rank_pooled": float(score.min()),
                                  "cv_pct_rank_decw": float(score[0].min()), "cv_by_lam": {str(l): float(score[i].min()) for i, l in enumerate(LAMS)}}
        print(f"{a.features} sub-{t:02d}: lam={lam} alpha={alpha:g}  CV pct rank {score.min():.4f} (decw {score[0].min():.4f})  "
              + " ".join(f"{l}:{score[i].min():.4f}" for i, l in enumerate(LAMS)), flush=True)
    tag = "" if a.subjects == [1, 2, 3, 4, 5, 6] else "_sub" + "-".join(map(str, a.subjects))
    (a.cache / f"pdecw-{a.features}_report{tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
