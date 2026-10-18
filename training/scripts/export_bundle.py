"""Export the fitted parameters of a CV-weighted ensemble (``scripts/ensemble.py``) into the numpy-only inference model
used by ``inference/predict.py`` (default output: ``inference/model/``).

Runs the pipeline code in ``--code-dir`` (the ``training/`` folder, default: the one this script lives in) to refit every
member per subject with the alphas (and lambdas) stored in the ensemble's similarity cache (or re-chosen by the same
training-only CV if a cache entry is missing). Every member is checked against the live pipeline object, and against
the cached test similarity when present. Only training recordings enter the exported parameters; the test EEG is used
for those checks only, and the test videos' features are exported as the candidate set.

Members use ``ensemble.py``'s syntax: ``decw:FEATURES[:FRAMES]`` / ``dec:...`` (ridge decoders, any feature cache, e.g.
CLIP / DINOv2 / V-JEPA / AlexNet maps / hand-crafted descriptors / CLAP / low-level AV), ``pdecw:...`` (subject-pooled
ridge decoders, built by ``scripts/pool_member.py``) and ``con:FEATURES:CONFIG[:FRAMES]`` (contrastive heads, averaged
over ``--con-seeds``). Weights: ``--weights`` (one global vector, integers summing to 10 or fractions) or
``--subject-weights`` (JSON ``{"1": [...], ...}``) for per-subject weights.

Output layout (every file < ``--max-file-mb``, so a plain git clone works without LFS)::

    <out>/meta.json                       members, weights, options, alphas / lambdas, public models, file hashes, checks
    <out>/candidates/<member>.npy         (102, D) float32 test-video features, row j = video 1001 + j
    <out>/sub-XX/train_eeg_mean.npy       mean preprocessed training EEG (for test-time centering)
    <out>/sub-XX/<member>.npz             that member's parameters for that subject (schema: see inference/model/README.md)

Example (the final v4b model, 'final_v4b_csls'; run from the repository root, features/ and the data linked there)::

    python training/scripts/export_bundle.py --cache results/similarity/_cache --out inference/model \\
        --member pdecw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 \\
        --member pdecw:alexnet_c5p6:8 --member decw:lowlevel_all:1 --member decw:clap_all:1 --member pdecw:lowav_all:1 \\
        --weights 2 1 1 2 1 1 2 --center-test --csls 5 --csls-stage member --source "v4b_ctr_m5 (final_v4b_csls)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np

MODEL_FORMAT = 2

# Public pretrained models behind the stimulus features. Revisions are the Hugging Face snapshots the cached features
# were computed with (checked against the local hub cache); the installed package versions are recorded in meta.json.
PUBLIC_MODELS = {
    "clip_b32": {"used_by": ["clip_b32", "clip_b32_*"], "library": "open_clip_torch", "name": "ViT-B-32",
                 "pretrained": "laion2b_s34b_b79k", "hf_repo": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                 "revision": "1a25a446712ba5ee05982a381eed697ef9b435cf", "output": "image embedding (encode_image), 512-d"},
    "dinov2_b": {"used_by": ["dinov2_b_*"], "library": "timm", "name": "vit_base_patch14_dinov2.lvd142m",
                 "hf_repo": "timm/vit_base_patch14_dinov2.lvd142m", "revision": "4685c99dabffe5affac90bd99dbffd25801ae58d",
                 "output": "mean of the final normed patch tokens (224 px), 768-d"},
    "vjepa2_l": {"used_by": ["vjepa2_l_*"], "library": "transformers", "name": "facebook/vjepa2-vitl-fpc64-256",
                 "hf_repo": "facebook/vjepa2-vitl-fpc64-256", "revision": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
                 "output": "mean encoder token embedding of a 16-frame clip, 1024-d"},
    "clap": {"used_by": ["clap_*"], "library": "transformers", "name": "laion/clap-htsat-unfused",
             "hf_repo": "laion/clap-htsat-unfused", "revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
             "output": "audio embedding (get_audio_features), 512-d"},
    "alexnet": {"used_by": ["alexnet_*"], "library": "torchvision", "name": "alexnet",
                "weights": "AlexNet_Weights.IMAGENET1K_V1 (alexnet-owt-7be5be79.pth)",
                "output": "conv5 (features[10], pre-ReLU) average-pooled to 6 x 6, mean of 8 frames"},
}
PUBLIC_MODEL_PACKAGES = ("open_clip_torch", "timm", "transformers", "torchvision", "torch", "av", "librosa", "opencv-python-headless")
OWN_FEATURES = ("lowlevel_*", "lowav_*", "lowvis_*", "lowaud_*")  # computed by our own code, no external model


def import_audited(code_dir: Path):
    """Import ``ensemble`` from ``code_dir/scripts`` with ``code_dir/src`` first on the path (the eegmr package)."""
    code_dir = code_dir.resolve()
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path[:0] = [str(code_dir / "src"), str(code_dir / "scripts")]
    import ensemble  # noqa: E402

    assert Path(ensemble.__file__).resolve().parent == (code_dir / "scripts").resolve()
    return ensemble


def bundle_name(m) -> str:
    return f"{m.name.replace(':', '-')}_f{m.frames}"


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Uncompressed ``.npz`` with fixed zip timestamps, so equal arrays give byte-identical files (stable hashes)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(value), allow_pickle=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(code_dir: Path) -> dict:
    def version(pkg: str) -> str | None:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(code_dir), *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {"code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--porcelain", "--", ".")),
            "packages": {p: version(p) for p in PUBLIC_MODEL_PACKAGES}, "python": sys.version.split()[0]}


def public_models_of(features: str) -> list[str]:
    return [k for k, v in PUBLIC_MODELS.items()
            if any(features == pat or (pat.endswith("*") and features.startswith(pat[:-1])) for pat in v["used_by"])]


def main() -> None:
    default_code = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-dir", type=Path, default=default_code, help="the training/ folder (src/ and scripts/); default: this one")
    p.add_argument("--cache", type=Path, default=None, help="the ensemble's --sim-cache directory (alphas, lambdas, checks)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--member", action="append", required=True)
    p.add_argument("--weights", type=float, nargs="+", default=None)
    p.add_argument("--subject-weights", type=Path, default=None, help='JSON {"1": [w, ...], ...}')
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--center-test", action="store_true")
    p.add_argument("--csls", type=int, default=0)
    p.add_argument("--csls-stage", choices=["member", "blend"], default="member")
    p.add_argument("--con-seeds", type=int, default=3)
    p.add_argument("--con-seed-start", type=int, default=0)
    p.add_argument("--max-file-mb", type=float, default=50.0, help="fail if a written file is larger (plain-git size limit)")
    p.add_argument("--source", default="", help="provenance note (experiment name)")
    p.add_argument("--root", type=Path, default=Path("eeg-mind-reading-challenge"))
    p.add_argument("--feature-dir", type=Path, default=Path("features"))
    a = p.parse_args()
    root, feature_dir, out_dir = a.root.resolve(), a.feature_dir.resolve(), a.out
    ens = import_audited(a.code_dir)
    import torch

    from eegmr import gpu
    from eegmr.contrastive import prepare, train_head
    from eegmr.data import TEST_IDS, load_test, load_train
    from eegmr.decoding import DecodingVariant, decoder_input, noise_whiteners
    from eegmr.features import load_features

    members = [ens.parse_member(spec, a.frames) for spec in a.member]
    for m in members:
        assert m.head in ("decw", "dec", "con", "pdecw"), f"export supports decw / dec / con / pdecw members, not {m.head!r}"
    seeds = list(range(a.con_seed_start, a.con_seed_start + a.con_seeds))
    if a.subject_weights:
        weights = {int(k): [float(x) for x in v] for k, v in json.loads(a.subject_weights.read_text()).items()}
    else:
        weights = {s: [float(x) for x in a.weights] for s in a.subjects}
    for s, w in weights.items():
        assert len(w) == len(members), f"sub-{s}: {len(w)} weights for {len(members)} members"
        total = sum(w)
        weights[s] = [x / total for x in w]  # integers summing to 10 or fractions: both normalized to sum 1
    window = slice(*a.window)
    cache_args = argparse.Namespace(sim_cache=a.cache, max_seconds=a.max_seconds, window=a.window,
                                    con_seeds=range(seeds[0], seeds[-1] + 1), center_test=a.center_test)
    written: list[Path] = []

    def put_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(path, arrays)
        written.append(path)

    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    for m in members:
        path = out_dir / "candidates" / f"{bundle_name(m)}.npy"
        np.save(path, load_features(m.features, m.frames, TEST_IDS, feature_dir).astype(np.float32))
        written.append(path)
    report: dict[str, dict] = {}
    hyper: dict[str, dict] = {}
    pooled = PooledExport(ens, a, cache_args, members, root, feature_dir) if any(m.head == "pdecw" for m in members) else None
    for s in a.subjects:
        X, ids = load_train(s, root)
        eeg = ens.preprocess(X, a.max_seconds)
        measured = ens.preprocess(load_test(s, root), a.max_seconds)
        if a.center_test:
            measured = ens.center_like(measured, eeg)
        sub_dir = out_dir / f"sub-{s:02d}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        np.save(sub_dir / "train_eeg_mean.npy", eeg.mean(axis=0).astype(np.float32))
        written.append(sub_dir / "train_eeg_mean.npy")
        hyper[f"sub-{s:02d}"] = {}
        for m in members:
            n = bundle_name(m)
            out: dict[str, np.ndarray] = {}
            F_trn = load_features(m.features, m.frames, ids, feature_dir)
            F_tst = load_features(m.features, m.frames, TEST_IDS, feature_dir)
            cv_path = ens.cache_path(cache_args, s, m, "cv") if a.cache else None
            if cv_path is not None and cv_path.exists():
                with np.load(cv_path) as z:
                    dec_alpha, enc_alpha = (None if np.isnan(x) else float(x) for x in z["alphas"])
            else:
                dec_alpha, enc_alpha = ens.member_alphas(m, F_trn, eeg, window)
            test_path = ens.cache_path(cache_args, s, m, "test") if a.cache else None
            cached = np.load(test_path)["sim"] if test_path is not None and test_path.exists() else None
            f_mean, f_std = F_trn.astype(np.float64).mean(0), F_trn.astype(np.float64).std(0)
            out["f_mean"], out["f_std"] = f_mean.astype(np.float32), np.where(f_std == 0, 1.0, f_std).astype(np.float32)
            lam = None
            if m.head == "pdecw":
                W, live, params, (dec_alpha, enc_alpha, lam) = pooled.export(s, m, measured, F_tst)
                out |= params
            elif m.head in ("decw", "dec"):
                dec = DecodingVariant(dec_alpha, whiten_input=m.head == "decw", enc_alpha=enc_alpha, window=window)
                dec.fit(F_trn, eeg)
                live = dec.similarity(measured, F_tst)
                b = dec.basis_
                Y = gpu.standardize(dec.features_trn_, dec.features_trn_)
                y_mean = Y.mean(dim=0)
                B = (b.V * (b.S / (b.S**2 + dec_alpha))) @ (b.U.T @ (Y - y_mean))
                W = dec.whiteners_
                out |= {"x_mean": b.mean.cpu().numpy().astype(np.float32), "x_std": b.std.cpu().numpy().astype(np.float32),
                        "B": B.cpu().numpy().astype(np.float32), "y_mean": y_mean.cpu().numpy().astype(np.float32)}
            else:
                dev = gpu.default_device()
                cfg = ens.contrastive_configs()[m.config]
                W = noise_whiteners(F_trn, eeg, enc_alpha, window, device=dev)
                X_trn, X_tst, V_trn, V_tst = prepare(eeg, measured, F_trn, F_tst, W, window, dev)
                Xr = decoder_input(eeg, window, W, dev)
                x_std = Xr.std(dim=0, unbiased=False)
                out |= {"x_mean": Xr.mean(dim=0).cpu().numpy().astype(np.float32),
                        "x_std": torch.where(x_std == 0, torch.ones_like(x_std), x_std).cpu().numpy().astype(np.float32)}
                z = 0
                for seed in seeds:
                    h = train_head(X_trn, V_trn, cfg, dec_alpha, seed)
                    z = z + ens.row_z(h.similarity(X_tst, V_tst).double().cpu().numpy())
                    out |= {f"seed{seed}_W_e": h.W_e.cpu().numpy().astype(np.float32),
                            f"seed{seed}_W_v": h.W_v.cpu().numpy().astype(np.float32),
                            f"seed{seed}_tau": np.float32(h.log_tau.exp().item())}
                live = z / len(seeds)
            if W is not None:
                out["whiteners"] = W.cpu().numpy().astype(np.float32)
            put_npz(sub_dir / f"{n}.npz", out)
            hyper[f"sub-{s:02d}"][n] = {"dec_alpha": dec_alpha, "enc_alpha": enc_alpha, **({"lam": lam} if lam is not None else {})}
            report[f"sub-{s:02d}/{n}"] = {"live_vs_cached_max_abs": None if cached is None else float(np.abs(live - cached).max())}
        print(f"sub-{s:02d} exported", flush=True)

    too_big = [(str(f), round(f.stat().st_size / 2**20, 1)) for f in written if f.stat().st_size > a.max_file_mb * 2**20]
    assert not too_big, f"files over {a.max_file_mb} MB: {too_big}"
    meta = {"format_version": MODEL_FORMAT, "source": a.source,
            "members": [{"name": bundle_name(m), "spec": spec, "head": m.head, "features": m.features, "frames": m.frames,
                         "public_models": public_models_of(m.features) or ["none (computed by our own code)"]}
                        for m, spec in zip(members, a.member)],
            "weights": {str(s): w for s, w in weights.items()}, "window": a.window, "max_seconds": a.max_seconds,
            "sfreq": 500, "decimate_factor": 5, "csls_k": a.csls, "csls_stage": a.csls_stage, "con_seeds": seeds,
            "center_test": a.center_test, "candidate_ids": [1001, 1102],
            "hyperparameters": hyper, "public_models": PUBLIC_MODELS, "own_features": list(OWN_FEATURES),
            "provenance": provenance(a.code_dir.resolve()),
            "files": {str(f.relative_to(out_dir)): {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in sorted(written)},
            "checks": report}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    worst = max((v["live_vs_cached_max_abs"] or 0.0) for v in report.values())
    missing = sum(v["live_vs_cached_max_abs"] is None for v in report.values())
    biggest = max(written, key=lambda f: f.stat().st_size)
    print(f"export checks: max |live - cached| = {worst:.3g} over {len(report) - missing} member fits ({missing} without cache)")
    print(f"{len(written)} files, {sum(f.stat().st_size for f in written) / 2**20:.0f} MB, largest {biggest.relative_to(out_dir)} "
          f"({biggest.stat().st_size / 2**20:.1f} MB)")


class PooledExport:
    """``pdecw`` members (scripts/pool_member.py): the subject-pooled whitened decoder of a target subject is still one
    linear map. At the cached (lam, alpha), the weighted ridge on [target rows (w=1); other subjects' rows (w=lam)] gives
    ``pred = X_std @ B + b`` with ``B = V diag(S / (S^2 + alpha)) U^T (Y - y_w)`` and ``b = y_w - x_w @ B``; ``X_std`` is
    the TARGET subject's whitened window standardized with its own training statistics (other subjects contribute only
    through B). So inference is identical to decw: export B, b (as y_mean), the target's whiteners and x_mean / x_std."""

    def __init__(self, ens, a, cache_args, members, root, feature_dir):
        import pool_member as pm  # scripts/pool_member.py of --code-dir
        from eegmr.data import load_train
        from eegmr.decoding import decoder_input, noise_whiteners
        from eegmr.features import load_features

        self.ens, self.pm, self.args, self.root, self.fdir = ens, pm, cache_args, root, feature_dir
        self.decoder_input, self.noise_whiteners, self.load_features = decoder_input, noise_whiteners, load_features
        self.data = {}
        for s in range(1, 7):
            X, ids = load_train(s, root)
            self.data[s] = dict(ids=ids, eeg=ens.preprocess(X, a.max_seconds))
        self.per_feature: dict[tuple[str, int], dict] = {}

    def _subjects(self, m) -> dict:
        """Per subject: features, decw alphas, full-training whiteners and standardized whitened window."""
        k = (m.features, m.frames)
        if k not in self.per_feature:
            pm, out = self.pm, {}
            for s, d in self.data.items():
                F = self.load_features(m.features, m.frames, d["ids"], self.fdir)
                decw = self.ens.Member("decw", m.features, m.frames)
                with np.load(self.ens.cache_path(self.args, s, decw, "cv")) as z:
                    dec_alpha, enc_alpha = (float(x) for x in z["alphas"])
                Wh = self.noise_whiteners(F, d["eeg"], enc_alpha, slice(*self.args.window), device=pm.DEV)
                Xs = self.decoder_input(d["eeg"], slice(*self.args.window), Wh, pm.DEV)
                mu, sd = pm.std_stats(Xs)
                out[s] = dict(F=F, Ft=pm.t64(F), Wh=Wh, mu=mu, sd=sd, X=(Xs - mu) / sd, enc_alpha=enc_alpha)
            self.per_feature[k] = out
        return self.per_feature[k]

    def export(self, t: int, m, measured: np.ndarray, F_tst: np.ndarray):
        import torch

        pm, subj = self.pm, self._subjects(m)
        with np.load(self.ens.cache_path(self.args, t, m, "cv")) as z:
            alpha, enc_alpha = (float(x) for x in z["alphas"])
            lam = float(z["lam"])
        st = subj[t]
        fm, fs = pm.std_stats(st["Ft"])
        others = [o for o in subj if o != t] if lam > 0 else []
        Xc = torch.cat([st["X"]] + [subj[o]["X"] for o in others])
        Yc = torch.cat([(st["Ft"] - fm) / fs] + [(subj[o]["Ft"] - fm) / fs for o in others])
        w = torch.ones(len(Xc), dtype=torch.float64, device=pm.DEV)
        w[len(st["X"]):] = lam
        xm, ym = (w[:, None] * Xc).sum(0) / w.sum(), (w[:, None] * Yc).sum(0) / w.sum()
        sw = w.sqrt()[:, None]
        U, S, Vh = torch.linalg.svd((Xc - xm) * sw, full_matrices=False)
        B = (Vh.T * (S / (S**2 + alpha))) @ (U.T @ ((Yc - ym) * sw))
        b = ym - xm @ B
        Xn = (self.decoder_input(measured, slice(*self.args.window), st["Wh"], pm.DEV) - st["mu"]) / st["sd"]
        live = pm.cosine(Xn @ B + b, (pm.t64(F_tst) - fm) / fs).cpu().numpy().astype(np.float64)
        params = {"x_mean": st["mu"].cpu().numpy().astype(np.float32), "x_std": st["sd"].cpu().numpy().astype(np.float32),
                  "B": B.cpu().numpy().astype(np.float32), "y_mean": b.cpu().numpy().astype(np.float32)}
        return st["Wh"], live, params, (alpha, enc_alpha, lam)


if __name__ == "__main__":
    main()
