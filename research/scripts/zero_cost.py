"""Zero-complexity ideas on the whitened decoder (decw): test-time normalization, time windows, jitter, CSLS.

One run per (feature set, subject, spec). For every spec the script stores, per subject, the 5-fold training-only CV
similarity matrices (same folds as every other CV in the package: KFold(5, shuffle, seed 0); the noise whiteners are
re-estimated inside each outer fold) and the test similarity matrix, both for ALL alphas, so the alpha can be chosen by
CV afterwards. Nothing about the test labels is used before the report step.

    python scripts/zero_cost.py run --features clip_b32 --frames 8 --specs base cand_own eeg_own both_own jit2
    python scripts/zero_cost.py report --features clip_b32 --frames 8 --specs base cand_own ...
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hypergeom, wilcoxon
from sklearn.model_selection import KFold

from eegmr import gpu
from eegmr.data import DEFAULT_ROOT, TEST_IDS, decimate, load_test, load_test_labels, load_train
from eegmr.decoding import noise_whiteners
from eegmr.evaluate import true_ranks
from eegmr.features import load_features
from eegmr.linear import DEFAULT_ALPHAS

OUT = Path("results/zero_cost")
N_T = 150  # samples at 100 Hz after the 1.5 s crop


@dataclass(frozen=True)
class Spec:
    window: tuple[int, int] = (5, 60)
    cand_norm: str = "train"  # candidates standardized with "train" stats; "center": train std, own mean; "own": own mean/std
    eeg_norm: str = "none"  # decoder inputs of the recordings being scored: "center": own mean replaced by the training
    # mean (train scale kept); "own": own mean and std (mapped onto the training scale)
    shifts: tuple[int, ...] = (0,)  # training copies shifted by these many samples (jitter augmentation)


SPECS = {
    "base": Spec(),
    "cand_own": Spec(cand_norm="own"),
    "eeg_own": Spec(eeg_norm="own"),
    "both_own": Spec(cand_norm="own", eeg_norm="own"),
    "eeg_center": Spec(eeg_norm="center"),
    "cand_center": Spec(cand_norm="center"),
    "both_center": Spec(cand_norm="center", eeg_norm="center"),
    "jit1": Spec(shifts=(-1, 0, 1)),
    "jit2": Spec(shifts=(-2, -1, 0, 1, 2)),
    "win_150_350": Spec(window=(15, 35)),
    "win_600_1200": Spec(window=(60, 120)),
}


def own_standardize(x: torch.Tensor, ref_mean: torch.Tensor | None = None, ref_std: torch.Tensor | None = None) -> torch.Tensor:
    """z-score the columns of ``x`` with its own mean/std; optionally map back to the scale of a reference (mean, std)."""
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    z = (x - x.mean(dim=0)) / std
    return z if ref_mean is None else z * ref_std + ref_mean


class Context:
    """Everything one subject needs, computed once: data, per-fold noise whiteners, whitened EEG (all 150 samples)."""

    def __init__(self, subject: int, features: str, frames: int, device: str = "cuda") -> None:
        self.subject, self.device = subject, device
        X, ids = load_train(subject)
        eeg = decimate(X[..., :750])
        self.eeg_np = eeg
        self.test_np = decimate(load_test(subject)[..., :750])
        self.F_np = load_features(features, frames, ids)
        self.Ft_np = load_features(features, frames, TEST_IDS)
        self.true_ids = load_test_labels(subject)
        self.alphas = DEFAULT_ALPHAS
        w = slice(5, 60)
        self.enc_alpha = float(self.alphas[gpu.retrieval_cv_scores(self.F_np, eeg, self.alphas, [w])[0].argmin()])
        t = lambda a: gpu.as_tensor(a, device, torch.float64)  # noqa: E731
        self.F, self.Ft = t(self.F_np), t(self.Ft_np)
        self.folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(eeg))))
        whiten = lambda W, e: torch.einsum("tcd,ndt->nct", W, t(e))  # noqa: E731
        self.fold_data = []
        for trn, val in self.folds:
            W = noise_whiteners(self.F_np[trn], eeg[trn], self.enc_alpha, slice(0, N_T), device=device)
            self.fold_data.append((trn, val, whiten(W, eeg[trn]), whiten(W, eeg[val])))
        W = noise_whiteners(self.F_np, eeg, self.enc_alpha, slice(0, N_T), device=device)
        self.full = (self.F, whiten(W, eeg), whiten(W, self.test_np))
        self._bases: dict = {}

    def _basis(self, key, Xtrn) -> gpu.RidgeBasis:
        if key not in self._bases:
            self._bases[key] = gpu.ridge_basis(Xtrn, pca_variance=None, device=self.device)
        return self._bases[key]

    def _sims(self, key, spec: Spec, F_trn, Xw_trn, Xw_new, F_new) -> torch.Tensor:
        """Cosine similarity (alphas, n_new, n_new) for the recordings Xw_new (whitened, all samples) vs candidates F_new."""
        a, b = spec.window
        Xtrn = torch.cat([Xw_trn[..., a + s:b + s].flatten(1) for s in spec.shifts])
        Ytrn = F_trn.repeat(len(spec.shifts), 1)
        basis = self._basis((key, spec.window, spec.shifts), Xtrn)
        Xnew = Xw_new[..., a:b].flatten(1)
        if spec.eeg_norm == "own":
            Xnew = own_standardize(Xnew, basis.mean, basis.std)
        elif spec.eeg_norm == "center":
            Xnew = Xnew - Xnew.mean(dim=0) + basis.mean
        pred = gpu.ridge_predict(basis, gpu.standardize(Ytrn, Ytrn), Xnew, self.alphas)
        cand = gpu.standardize(F_new, F_trn)
        if spec.cand_norm == "own":
            cand = own_standardize(F_new)
        elif spec.cand_norm == "center":
            cand = cand - cand.mean(dim=0)
        return gpu.cosine_similarity(pred, cand.float())

    def run(self, spec: Spec) -> dict:
        cv = []
        for k, (trn, val, Xw_trn, Xw_val) in enumerate(self.fold_data):
            cv.append(self._sims(("fold", k), spec, self.F[trn], Xw_trn, Xw_val, self.F[val]).cpu())
        F, Xw_trn, Xw_test = self.full
        test = self._sims(("full",), spec, F, Xw_trn, Xw_test, self.Ft).cpu()
        return {"cv": cv, "test": test}


# ------------------------------------------------------------------------------------------------ metrics
def ranks(sim: torch.Tensor) -> torch.Tensor:
    """1-based rank of the diagonal entry in each row; ties count against. ``sim``: ``(..., n, n)``."""
    return (sim >= sim.diagonal(dim1=-2, dim2=-1)[..., None]).sum(-1)


def csls_t(sim: torch.Tensor, k: int) -> torch.Tensor:
    """Torch CSLS (same formula as ``eegmr.retrieval.csls``) on ``(..., n, m)``."""
    r_rec = sim.topk(min(k, sim.shape[-1]), dim=-1).values.mean(-1, keepdim=True)
    r_cand = sim.topk(min(k, sim.shape[-2]), dim=-2).values.mean(-2, keepdim=True)
    return 2 * sim - r_rec - r_cand


def row_z(sim: torch.Tensor) -> torch.Tensor:
    return (sim - sim.mean(-1, keepdim=True)) / sim.std(-1, keepdim=True)


def cv_scores(fold_sims: list[torch.Tensor]) -> dict:
    """Mean percentile rank, expected top-1 / top-5 of a 102-candidate test (hypergeometric), from per-fold (n, n) sims."""
    pct, e1, e5, n = 0.0, 0.0, 0.0, 0
    for s in fold_sims:
        r = ranks(s).numpy()
        nv = s.shape[-1]
        pct += ((r - 1) / (nv - 1)).sum()
        e1 += hypergeom.cdf(0, nv - 1, r - 1, 101).sum()
        e5 += hypergeom.cdf(4, nv - 1, r - 1, 101).sum()
        n += len(r)
    return {"pct": pct / n, "e1": e1 / n, "e5": e5 / n}


def curve(cv: list[torch.Tensor]) -> np.ndarray:
    """CV percentile-rank curve over alphas (lower is better)."""
    tot, n = 0.0, 0
    for s in cv:
        tot = tot + ((ranks(s) - 1) / (s.shape[-1] - 1)).sum(-1).double()
        n += s.shape[-2]
    return (tot / n).numpy()


def pick(res: dict) -> dict:
    """Choose the alpha by CV; return per-fold sims and the test sim at it, plus the CV scores."""
    c = curve(res["cv"])
    i = int(c.argmin())
    return {"i": i, "curve": c, "cv": [s[i] for s in res["cv"]], "test": res["test"][i]}


def file_for(features: str, frames: int, spec: str, subject: int) -> Path:
    return OUT / f"{features}_f{frames}_{spec}_sub{subject:02d}.pkl"


def run_cmd(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.subjects:
        ctx = Context(s, args.features, args.frames)
        for name in args.specs:
            res = ctx.run(SPECS[name])
            with open(file_for(args.features, args.frames, name, s), "wb") as f:
                pickle.dump(res, f)
        print(f"sub-{s:02d} done ({args.features}, enc_alpha={ctx.enc_alpha:g})", flush=True)


# ------------------------------------------------------------------------------------------------ report
def load(features: str, frames: int, spec: str, subject: int) -> dict:
    with open(file_for(features, frames, spec, subject), "rb") as f:
        return pickle.load(f)


def member(features: str, frames: int, spec: str, subject: int, csls: int = 0) -> dict:
    p = pick(load(features, frames, spec, subject))
    if csls:
        p["cv"] = [csls_t(s, csls) for s in p["cv"]]
        p["test"] = csls_t(p["test"], csls)
    return p


def blend(parts: list[dict], csls: int = 0) -> dict:
    cv = [sum(row_z(p["cv"][k]) for p in parts) for k in range(5)]
    test = sum(row_z(p["test"]) for p in parts)
    if csls:
        cv, test = [csls_t(s, csls) for s in cv], csls_t(test, csls)
    return {"cv": cv, "test": test}


def summarize(per_subject: dict[int, dict], base_ranks: dict[int, np.ndarray] | None = None) -> dict:
    """CV (mean over the 6 subjects) and dev (subjects 1-4) numbers for one variant."""
    cvs = {s: cv_scores(p["cv"]) for s, p in per_subject.items()}
    out = {"cv_pct": np.mean([c["pct"] for c in cvs.values()]), "cv_e1": np.mean([c["e1"] for c in cvs.values()]),
           "cv_e5": np.mean([c["e5"] for c in cvs.values()]), "cv_by_sub": {s: c["pct"] for s, c in cvs.items()}}
    dev = [s for s in per_subject if s <= 4]
    rk = {s: true_ranks(per_subject[s]["test"].numpy().astype(np.float64), load_test_labels(s)) for s in dev}
    allr = np.concatenate([rk[s] for s in dev])
    out.update(top1=(allr <= 1).mean() * 100, top5=(allr <= 5).mean() * 100,
               med=np.mean([np.median(rk[s]) for s in dev]), ranks=rk, allr=allr)
    if base_ranks is not None:
        b = np.concatenate([base_ranks[s] for s in dev])
        out.update(better=int((allr < b).sum()), worse=int((allr > b).sum()),
                   p=float(wilcoxon(allr, b).pvalue) if not np.array_equal(allr, b) else 1.0)
    return out


def fmt(name: str, o: dict, base: dict | None) -> str:
    d = ""
    if base is not None:
        d = (f" | dCV {o['cv_pct'] - base['cv_pct']:+.4f} dE1 {o['cv_e1'] - base['cv_e1']:+.4f} "
             f"dE5 {o['cv_e5'] - base['cv_e5']:+.4f} | dev better/worse {o.get('better', 0)}/{o.get('worse', 0)} p={o.get('p', 1):.3g}")
    return (f"{name:28s} CV pct {o['cv_pct']:.4f} E1 {o['cv_e1']*100:5.2f} E5 {o['cv_e5']*100:5.2f} | dev "
            f"{o['top1']:5.1f} / {o['top5']:5.1f} / {o['med']:5.2f}{d}")


def report_cmd(args) -> None:
    """Each entry of ``--specs`` is ``<spec>`` or ``<spec>@csls<k>``; the first entry is the baseline."""
    base = None
    for entry in args.specs:
        name, _, tail = entry.partition("@csls")
        per = {s: member(args.features, args.frames, name, s, int(tail) if tail else 0) for s in args.subjects}
        o = summarize(per, None if base is None else base["ranks"])
        if base is None:
            base = o
        print(fmt(entry, o, None if o is base else base), flush=True)


def blend_cmd(args) -> None:
    """Equal-weight row-z blends. ``--members`` / ``--ref`` entries are ``features:frames:spec``; the paired test is
    against ``--ref`` (default: the plain blend of the same members). ``--csls`` adds a CSLS row for each k."""
    def build(members: list[str], k: int) -> dict[int, dict]:
        out = {}
        for s in args.subjects:
            parts = []
            for m in members:
                feat, frames, spec = m.split(":")
                parts.append(member(feat, int(frames), spec, s))
            out[s] = blend(parts, k)
        return out

    ref = summarize(build(args.ref or args.members, 0))
    print(fmt("ref " + ",".join(args.ref or args.members), ref, None), flush=True)
    for k in [0] + args.csls:
        if k == 0 and not args.ref:
            continue
        print(fmt(f"blend csls{k}", summarize(build(args.members, k), ref["ranks"]), ref), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["run", "report", "blend"])
    p.add_argument("--features", default="clip_b32")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--specs", nargs="+", default=["base"])
    p.add_argument("--members", nargs="+", default=[])
    p.add_argument("--ref", nargs="+", default=[])
    p.add_argument("--csls", type=int, nargs="*", default=[])
    args = p.parse_args()
    {"run": run_cmd, "report": report_cmd, "blend": blend_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
