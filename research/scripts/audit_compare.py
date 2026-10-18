"""Evaluate audit controls for any ensemble (CPU only). Subcommands:

    determinism  compare every ``*_test.npz`` (and optionally ``*_cv.npz``) member similarity present in both caches for the
                 given subjects, and optionally two submission directories (ranked lists)
    permutation  check that permuting the test recordings (``perms.json`` from ``audit_controls.py --permute``) permuted the
                 output rows identically
    rescore      re-rank saved blended similarity matrices (``results/similarity/<exp>/sub-XX.npy``) and compare them with
                 the written submission files; dev scores for subjects with labels
    score        dev top-1 / top-5 / median rank of a submission directory (e.g. the label-shuffle control's output)

Examples:
    python scripts/audit_compare.py determinism --new-cache A/cache --ref-cache B/cache --subjects 1 5 \\
        --new-sub A/submissions/x_cv --ref-sub B/submissions/final_csls
    python scripts/audit_compare.py permutation --perms A/perms.json --base-sub A/submissions/r_cv --perm-sub A/submissions/p_cv
    python scripts/audit_compare.py rescore --sim-dir B/results/similarity/fin_cv --sub-dir B/submissions/fin_cv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eegmr.data import DEFAULT_ROOT, TEST_IDS, load_test_labels


def ranks_of_truth(ranked: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.argmax(ranked == y[:, None], axis=1) + 1


def sub_file(d: Path, s: int) -> Path:
    return d / f"sub-{s:02d}_ranked_candidate_list.npy"


def compare_ranked(new: np.ndarray, ref: np.ndarray) -> dict:
    return {"identical": bool(np.array_equal(new, ref)), "rows_identical": int((new == ref).all(1).sum()),
            "entries_differing": int((new != ref).sum()), "top1_agree": float(np.mean(new[:, 0] == ref[:, 0]))}


def dev_scores(d: Path, subjects: list[int], root: Path) -> dict:
    out, t1, t5 = {}, [], []
    for s in subjects:
        y = load_test_labels(s, root)
        if y is None or not sub_file(d, s).exists():
            continue
        r = ranks_of_truth(np.load(sub_file(d, s)), y)
        out[f"sub-{s:02d}"] = {"top1": float(np.mean(r <= 1)), "top5": float(np.mean(r <= 5)), "median": float(np.median(r))}
        t1.append(np.mean(r <= 1)); t5.append(np.mean(r <= 5))
    if t1:
        out["mean"] = {"top1": float(np.mean(t1)), "top5": float(np.mean(t5))}
    return out


def determinism(a: argparse.Namespace) -> dict:
    rep = {}
    kinds = ["test"] + (["cv"] if a.include_cv else [])
    for s in a.subjects:
        d = {}
        for kind in kinds:
            for f in sorted(a.new_cache.glob(f"sub-{s:02d}_*_{kind}.npz")):
                r = a.ref_cache / f.name
                if not r.exists():
                    continue
                with np.load(f) as x, np.load(r) as y:
                    keys = [k for k in x.files if k in y.files]
                    same = all(np.array_equal(x[k], y[k]) for k in keys)
                    diff = max(float(np.nanmax(np.abs(x[k] - y[k]))) for k in keys)
                d[f.name] = {"bit_identical": same, "max_abs_diff": diff}
        if a.new_sub and a.ref_sub and sub_file(a.new_sub, s).exists() and sub_file(a.ref_sub, s).exists():
            d["ranked_lists"] = compare_ranked(np.load(sub_file(a.new_sub, s)), np.load(sub_file(a.ref_sub, s)))
        rep[f"sub-{s:02d}"] = d
    return rep


def permutation(a: argparse.Namespace) -> dict:
    perms = {int(k): np.array(v) for k, v in json.loads(a.perms.read_text()).items()}
    return {f"sub-{s:02d}": {"output_equals_permuted_input": bool(np.array_equal(
                np.load(sub_file(a.perm_sub, s)), np.load(sub_file(a.base_sub, s))[p])),
            **compare_ranked(np.load(sub_file(a.perm_sub, s)), np.load(sub_file(a.base_sub, s))[p])}
            for s, p in perms.items()}


def rescore(a: argparse.Namespace) -> dict:
    rep = {}
    for s in a.subjects:
        sim_f = a.sim_dir / f"sub-{s:02d}.npy"
        if not sim_f.exists() or not sub_file(a.sub_dir, s).exists():
            continue
        ranked = TEST_IDS[np.argsort(-np.load(sim_f), axis=1, kind="stable")]
        rep[f"sub-{s:02d}"] = compare_ranked(ranked, np.load(sub_file(a.sub_dir, s)))
    rep["dev"] = dev_scores(a.sub_dir, a.subjects, a.root)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("determinism")
    d.add_argument("--new-cache", type=Path, required=True)
    d.add_argument("--ref-cache", type=Path, required=True)
    d.add_argument("--subjects", type=int, nargs="+", default=[1, 5])
    d.add_argument("--include-cv", action="store_true")
    d.add_argument("--new-sub", type=Path)
    d.add_argument("--ref-sub", type=Path)
    q = sp.add_parser("permutation")
    q.add_argument("--perms", type=Path, required=True)
    q.add_argument("--base-sub", type=Path, required=True)
    q.add_argument("--perm-sub", type=Path, required=True)
    r = sp.add_parser("rescore")
    r.add_argument("--sim-dir", type=Path, required=True)
    r.add_argument("--sub-dir", type=Path, required=True)
    r.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    c = sp.add_parser("score")
    c.add_argument("--sub-dir", type=Path, required=True)
    c.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    a = p.parse_args()
    rep = {"determinism": determinism, "permutation": permutation, "rescore": rescore,
           "score": lambda a: dev_scores(a.sub_dir, a.subjects, a.root)}[a.cmd](a)
    text = json.dumps(rep, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
