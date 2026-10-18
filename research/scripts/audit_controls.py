"""Leakage / robustness controls for the headline ensemble (code audit). Patches data loading from outside; the audited
code (``--code-dir``, a checkout of the audited commit) is imported unchanged.

Modes:
    shuffle    permute each subject's training video IDs against its training EEG (pairs broken), then run the full
               ensemble pipeline (CV + test) and score dev: should be at chance
    shuffle_pool  the same permutation for ``scripts/pool_member.py`` (all six subjects' pairs broken, since pooled
               members use every subject): builds shuffled ``pdecw`` caches; pass that cache dir to a ``shuffle`` run
    rerun      recompute the test similarities of ``--subjects-recompute`` from scratch (CV and other subjects' test
               similarities come from ``--cache``), optionally with the test recordings of those subjects permuted
               (``--permute``); the permutation is saved so the output rows can be compared
    fp16       re-extract V-JEPA 2 features of the 102 test videos in fp32 and fp16, compare with the cached features
               (cosine) and rescore the decw V-JEPA member on dev with the fp32 test features (top-1 agreement, ranks)

All other arguments after ``--`` go to ``ensemble.py`` (members, flags, --experiment, ...).

Example (run from the audited checkout):
    python /path/to/audit_controls.py --code-dir . --mode shuffle -- --member decw:clip_b32 ... --experiment audit_shuffle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def import_audited(code_dir: Path):
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(here)]
    sys.path[:0] = [str(code_dir / "src"), str(code_dir / "scripts")]
    import ensemble  # noqa: E402  (the audited scripts/ensemble.py)

    assert Path(ensemble.__file__).resolve().parent == (code_dir / "scripts").resolve(), ensemble.__file__
    return ensemble


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["shuffle", "shuffle_pool", "rerun", "fp16"], required=True)
    p.add_argument("--subjects-recompute", type=int, nargs="*", default=[])
    p.add_argument("--permute", action="store_true", help="rerun: permute the test recordings of the recomputed subjects")
    p.add_argument("--perm-out", type=Path, default=Path("audit_out/perms.json"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--feature-dir-out", type=Path, default=Path("audit_out/features_fp"))
    args, rest = p.parse_known_args()
    rest = [a for a in rest if a != "--"]
    code_dir = args.code_dir.resolve()
    ens = import_audited(code_dir)
    import numpy as np

    rng = np.random.default_rng(args.seed)
    if args.mode == "shuffle_pool":
        import pool_member

        original_pool = pool_member.load_train

        def shuffled_pool(subject: int, root):
            X, ids = original_pool(subject, root)
            return X, ids[np.random.default_rng(args.seed + subject).permutation(len(ids))]

        pool_member.load_train = shuffled_pool
        sys.argv = ["pool_member.py", *rest]
        pool_member.main()
        return
    if args.mode == "shuffle":
        original = ens.load_train

        def shuffled_train(subject: int, root):
            X, ids = original(subject, root)
            return X, ids[np.random.default_rng(args.seed + subject).permutation(len(ids))]

        ens.load_train = shuffled_train
    elif args.mode == "rerun" and args.permute:
        perms = {s: rng.permutation(102).tolist() for s in args.subjects_recompute}
        args.perm_out.parent.mkdir(parents=True, exist_ok=True)
        args.perm_out.write_text(json.dumps(perms))
        original_test = ens.load_test

        def permuted_test(subject: int, root):
            X = original_test(subject, root)
            return X[np.array(perms[subject])] if subject in perms else X

        ens.load_test = permuted_test
        original_labels = ens.load_test_labels

        def permuted_labels(subject: int, root):  # keep the reported dev scores consistent with the permuted rows
            y = original_labels(subject, root)
            return y[np.array(perms[subject])] if (y is not None and subject in perms) else y

        ens.load_test_labels = permuted_labels
    if args.mode == "fp16":
        run_fp16(args, rest)
        return
    sys.argv = ["ensemble.py", *rest]
    ens.main()


def run_fp16(args: argparse.Namespace, rest: list[str]) -> None:
    """V-JEPA 2 test-video features in fp32 vs fp16, and the effect on the decw V-JEPA member's dev rankings."""
    import numpy as np
    import torch

    from eegmr.data import DEFAULT_ROOT, TEST_IDS, load_test, load_test_labels, load_train
    from eegmr.decoding import DecodingVariant
    from eegmr.evaluate import true_ranks
    from eegmr.features import extract_clip_features, load_features
    from predict import preprocess

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--alphas-json", type=Path, required=True, help="{subject: [dec_alpha, enc_alpha]} for decw:vjepa2_l_all")
    a = p.parse_args(rest)
    cached = load_features("vjepa2_l_all", 16, TEST_IDS, a.feature_dir)
    out = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        d = args.feature_dir_out / name
        extract_clip_features("vjepa2_l", 16, a.root, d, ids=TEST_IDS, dtype=dtype)
        out[name] = load_features("vjepa2_l_all", 16, TEST_IDS, d)

    def cos(x, y):
        return (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)

    report = {f"cos_cached_vs_{k}": {"min": float(cos(cached, v).min()), "median": float(np.median(cos(cached, v)))}
              for k, v in out.items()}
    report["cos_fp16_vs_fp32"] = {"min": float(cos(out["fp16"], out["fp32"]).min()),
                                  "median": float(np.median(cos(out["fp16"], out["fp32"])))}
    alphas = {int(k): v for k, v in json.loads(a.alphas_json.read_text()).items()}
    for s in range(1, 5):
        X, ids = load_train(s, a.root)
        eeg = preprocess(X, 1.5)
        measured = preprocess(load_test(s, a.root), 1.5)
        F_trn = load_features("vjepa2_l_all", 16, ids, a.feature_dir)
        dec = DecodingVariant(alphas[s][0], True, None, alphas[s][1], slice(5, 60)).fit(F_trn, eeg)
        y = load_test_labels(s, a.root)
        sims = {k: dec.similarity(measured, v) for k, v in (("cached", cached), *out.items())}
        top1 = {k: s_.argmax(1) for k, s_ in sims.items()}
        ranks = {k: true_ranks(s_, y) for k, s_ in sims.items()}
        report[f"sub-{s:02d}"] = {
            "top1_agree_cached_vs_fp32": float(np.mean(top1["cached"] == top1["fp32"])),
            "top1_agree_fp16_vs_fp32": float(np.mean(top1["fp16"] == top1["fp32"])),
            "dev_top1": {k: float(np.mean(r <= 1)) for k, r in ranks.items()},
            "dev_top5": {k: float(np.mean(r <= 5)) for k, r in ranks.items()},
            "max_abs_rank_change_cached_vs_fp32": int(np.abs(ranks["cached"] - ranks["fp32"]).max()),
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
