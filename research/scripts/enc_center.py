"""Diagnostic: test-time centering for the ENCODING head (CLIP B-32 f8, 1.5 s crop, noise-whitened Euclid / corr matching).

Variants, all label-free: (m) the measured test EEG's own mean over the 102 recordings replaces the training ERP;
(mp) measured and predicted EEG are each centered on their own mean over the 102 recordings / candidates.
Dev only (sub-01..04): there is no CV analogue for the train/test shift. Paired Wilcoxon vs the uncentred matcher.
"""

import numpy as np
from scipy.stats import wilcoxon

from eegmr.data import TEST_IDS, decimate, load_test, load_test_labels, load_train
from eegmr.evaluate import true_ranks
from eegmr.features import load_features
from eegmr.linear import EncodingModel
from eegmr.matching import Matcher, oof_predictions


def main() -> None:
    rows: dict[str, dict[int, np.ndarray]] = {}
    for s in (1, 2, 3, 4):
        X, ids = load_train(s)
        eeg = decimate(X[..., :750])
        test = decimate(load_test(s)[..., :750])
        F, Ft = load_features("clip_b32", 8, ids), load_features("clip_b32", 8, TEST_IDS)
        model = EncodingModel(alpha_selection="retrieval").fit(F, eeg)
        pred = model.predict(Ft)
        oof = oof_predictions(F, eeg, model.alpha)
        erp = eeg.mean(0)
        variants = {"measured": (test - test.mean(0) + erp, pred), "meas+pred": (test - test.mean(0), pred - pred.mean(0))}
        for kind, kw in (("corr", {}), ("noise_euclid", {"kind": "noise", "noise_similarity": "euclid"})):
            matcher = Matcher(**({"kind": "corr"} if kind == "corr" else kw)).fit(eeg, oof if kind != "corr" else None)
            rows.setdefault(f"{kind}/base", {})[s] = true_ranks(matcher.similarity(test, pred), load_test_labels(s))
            for name, (m, p) in variants.items():
                rows.setdefault(f"{kind}/{name}", {})[s] = true_ranks(matcher.similarity(m, p), load_test_labels(s))
        print("sub", s, "done", flush=True)
    for kind in ("corr", "noise_euclid"):
        base = np.concatenate([rows[f"{kind}/base"][s] for s in (1, 2, 3, 4)])
        for name in ("base", "measured", "meas+pred"):
            r = rows[f"{kind}/{name}"]
            a = np.concatenate([r[s] for s in (1, 2, 3, 4)])
            p = wilcoxon(a, base).pvalue if not np.array_equal(a, base) else 1.0
            print(f"{kind:13s} {name:10s} top1 {np.mean(a <= 1) * 100:5.1f} top5 {np.mean(a <= 5) * 100:5.1f} "
                  f"med {np.mean([np.median(r[s]) for s in (1, 2, 3, 4)]):5.2f} | better {int((a < base).sum())} worse {int((a > base).sum())} p={p:.3g}")


if __name__ == "__main__":
    main()
