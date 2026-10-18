"""Predict the ranked candidate lists for the challenge test recordings. numpy + scipy only, CPU, a few seconds.

    python inference/predict.py --data-root /path/to/eeg-mind-reading-challenge --out predictions

For every subject (default: 5 6) this writes ``<out>/sub-0X_ranked_candidate_list.npy``: int64 (102, 102), row i = the
i-th recording of ``<data-root>/tst/sub-0X_tst.npy`` (original order), values = video IDs 1001..1102 from most to least
likely. With ``--subjects 1 2 3 4`` it also scores the lists against ``tst_stim/`` (development labels, scoring only).

The fitted parameters are read from ``inference/model/`` (written by ``training/scripts/export_bundle.py``); every file is
checked against the SHA-256 listed in ``model/meta.json`` before use (``--no-verify`` to skip). Pipeline per subject,
identical to ``training/scripts/ensemble.py`` (members, weights and options are in ``model/meta.json``):

    test EEG (102, 17, 1500) @ 500 Hz
      -> crop to 0-1.5 s -> FIR zero-phase decimation x5 (100 Hz)       [eegmr.data.decimate]
      -> center-test (if set): replace the mean over the 102 recordings by the training mean   [ensemble.center_like]
      -> per member: window 5:60 -> per-timepoint whitening (Sigma_t^-1/2) -> flatten (channel-major) -> standardize
           decw / dec / pdecw: ridge  pred = X @ B + y_mean  -> cosine vs the test videos' features (training-video stats)
           con               : per seed  cos(X @ W_e, V @ W_v), row-z, mean over the seeds
      -> CSLS (if set) per member [stage "member"] -> row-z -> weighted sum (weights from meta.json)
         -> CSLS on the blend [stage "blend"]
      -> rank: candidates 1001..1102 sorted by descending score (stable ties), rows in the original recording order
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import decimate

HERE = Path(__file__).resolve().parent
TEST_IDS = np.arange(1001, 1103)
N_TEST, N_CHANNELS, N_SAMPLES = 102, 17, 1500
MODEL_FORMAT = 2


def preprocess(X: np.ndarray, max_seconds: float, sfreq: int, factor: int) -> np.ndarray:
    """Crop the raw EEG to ``max_seconds`` (before filtering), then decimate (as ``eegmr.data.decimate``)."""
    X = X[..., : int(round(max_seconds * sfreq))]
    return decimate(X, factor, ftype="fir", zero_phase=True, axis=-1).astype(np.float32)


def row_z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=1, keepdims=True)) / a.std(axis=1, keepdims=True)


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def csls(sim: np.ndarray, k: int) -> np.ndarray:
    """Cross-domain similarity local scaling, as ``eegmr.retrieval.csls``."""
    k_rec, k_cand = min(k, sim.shape[1]), min(k, sim.shape[0])
    r_rec = np.sort(sim, axis=1)[:, -k_rec:].mean(axis=1)
    r_cand = np.sort(sim, axis=0)[-k_cand:, :].mean(axis=0)
    return 2 * sim - r_rec[:, None] - r_cand[None, :]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Model:
    """The exported model directory: ``meta.json`` plus per-subject parameter files and the candidate features."""

    def __init__(self, model_dir: Path, verify: bool = True) -> None:
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if self.meta.get("format_version") != MODEL_FORMAT:
            raise SystemExit(f"{self.dir}: model format {self.meta.get('format_version')!r}, this script reads {MODEL_FORMAT}")
        self.verify = verify
        self.window = slice(*self.meta["window"])

    def path(self, rel: str) -> Path:
        path = self.dir / rel
        if self.verify:
            want = self.meta["files"][rel]["sha256"]
            got = sha256(path)
            if got != want:
                raise SystemExit(f"{path}: SHA-256 mismatch (expected {want[:12]}..., got {got[:12]}...); "
                                 "re-clone the repository or pass --no-verify")
        return path

    def candidates(self, member: dict) -> np.ndarray:
        return np.load(self.path(f"candidates/{member['name']}.npy"))

    def params(self, subject: int, member: dict) -> dict[str, np.ndarray]:
        with np.load(self.path(f"sub-{subject:02d}/{member['name']}.npz")) as f:
            return {k: f[k] for k in f.files}

    def train_eeg_mean(self, subject: int) -> np.ndarray:
        return np.load(self.path(f"sub-{subject:02d}/train_eeg_mean.npy"))


def member_input(eeg: np.ndarray, w: dict, window: slice) -> np.ndarray:
    """Whitened (per timepoint) and standardized EEG window, flattened channel-major: ``(n, C * T_w)``."""
    x = eeg[..., window].astype(np.float64)
    if "whiteners" in w:  # decw / con / pdecw; the raw decoder (dec) has none
        x = np.einsum("tcd,ndt->nct", w["whiteners"].astype(np.float64), x)
    x = x.reshape(len(x), -1)
    return (x - w["x_mean"]) / w["x_std"]


def member_similarity(eeg: np.ndarray, w: dict, member: dict, candidates: np.ndarray, window: slice,
                      seeds: list[int]) -> np.ndarray:
    X = member_input(eeg, w, window)
    cand = (candidates.astype(np.float64) - w["f_mean"]) / w["f_std"]
    if member["head"] in ("decw", "dec", "pdecw"):  # pdecw = subject-pooled decoder, also one linear map
        pred = X @ w["B"].astype(np.float64) + w["y_mean"]
        return normalize(pred) @ normalize(cand).T
    z = 0.0
    for seed in seeds:
        e = normalize(X @ w[f"seed{seed}_W_e"].astype(np.float64))
        v = normalize(cand @ w[f"seed{seed}_W_v"].astype(np.float64))
        z = z + row_z(e @ v.T)
    return z / len(seeds)


def rank_candidates(score: np.ndarray) -> np.ndarray:
    ranked = TEST_IDS[np.argsort(-score, axis=1, kind="stable")].astype(np.int64)
    validate(ranked)
    return ranked


def validate(ranked: np.ndarray) -> None:
    """Raise unless ``ranked`` is a valid submission array: (102, 102) int, every row a permutation of 1001..1102."""
    assert ranked.shape == (N_TEST, N_TEST), f"expected shape {(N_TEST, N_TEST)}, got {ranked.shape}"
    assert np.issubdtype(ranked.dtype, np.integer), f"expected an integer dtype, got {ranked.dtype}"
    bad = [i for i, row in enumerate(ranked) if not np.array_equal(np.sort(row), TEST_IDS)]
    assert not bad, f"rows {bad[:5]}... are not permutations of 1001..1102"


def predict_subject(subject: int, model: Model, data_root: Path) -> np.ndarray:
    meta = model.meta
    raw = np.load(data_root / "tst" / f"sub-{subject:02d}_tst.npy").astype(np.float32)
    if raw.shape != (N_TEST, N_CHANNELS, N_SAMPLES):
        raise SystemExit(f"sub-{subject:02d}_tst.npy has shape {raw.shape}, expected {(N_TEST, N_CHANNELS, N_SAMPLES)}")
    eeg = preprocess(raw, meta["max_seconds"], meta["sfreq"], meta["decimate_factor"])
    if meta["center_test"]:
        eeg = (eeg - eeg.mean(axis=0) + model.train_eeg_mean(subject)).astype(np.float32)
    weights = meta["weights"][str(subject)]
    k, stage = meta.get("csls_k", 0), meta.get("csls_stage", "member")
    total = 0.0
    for member, weight in zip(meta["members"], weights):
        if weight == 0:
            continue
        sim = member_similarity(eeg, model.params(subject, member), member, model.candidates(member), model.window,
                                meta["con_seeds"])
        if k and stage == "member":
            sim = csls(sim, k)
        total = total + weight * row_z(sim)
    if k and stage == "blend":
        total = csls(total, k)
    return rank_candidates(total)


def score(ranked: np.ndarray, true_ids: np.ndarray) -> dict[str, float]:
    pos = np.argmax(ranked == np.asarray(true_ids, dtype=int)[:, None], axis=1)  # 0-based position of the true video
    return {"top1": float(np.mean(pos < 1)), "top5": float(np.mean(pos < 5)), "median_rank": float(np.median(pos + 1))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="challenge data folder (contains tst/sub-XX_tst.npy)")
    p.add_argument("--out", type=Path, required=True, help="output folder for sub-0X_ranked_candidate_list.npy")
    p.add_argument("--model", type=Path, default=HERE / "model", help="model folder (default: inference/model)")
    p.add_argument("--subjects", type=int, nargs="+", default=[5, 6])
    p.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check of the model files")
    a = p.parse_args()
    model = Model(a.model, verify=not a.no_verify)
    a.out.mkdir(parents=True, exist_ok=True)
    for s in a.subjects:
        t = time.time()
        ranked = predict_subject(s, model, a.data_root)
        path = a.out / f"sub-{s:02d}_ranked_candidate_list.npy"
        np.save(path, ranked)
        validate(np.load(path))
        msg = f"sub-{s:02d}: wrote {path} ({time.time() - t:.1f} s)"
        labels = a.data_root / "tst_stim" / f"sub-{s:02d}_tst_stim.npy"
        if labels.exists():  # development subjects only: score the written file (labels are used for scoring only)
            r = score(np.load(path), np.load(labels))
            msg += f"  top1={r['top1']:.3f} top5={r['top5']:.3f} median_rank={r['median_rank']:.1f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
