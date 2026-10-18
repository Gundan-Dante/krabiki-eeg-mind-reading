"""Summary table of the final ensemble grid (experiments ``fin{4,5}_{plain,ctr}_{off,b5,b10,m5,m10}``).

Reads the logs written by ``scripts/ensemble.py`` (CV-chosen weights, leave-one-fold-out table, strategy) and the saved test
similarity matrices (dev numbers and a paired Wilcoxon on per-trial ranks vs a reference experiment). Dev numbers are
reported but never used to choose a variant: the choice is by LOFO of the chosen strategy.

Example:
    python scripts/summarize_final.py <log dir> --reference ens_vj_b_cv > results/ens_final_summary.md
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from compare_matching import trial_ranks
from eegmr.data import DEFAULT_ROOT

STRATEGY_KEY = {"best single": "best single", "equal": "equal", "global": "global", "per-subject": "per-subject"}


def parse_log(path: Path) -> dict:
    text = path.read_text()
    loo = ast.literal_eval(re.search(r"leave-one-fold-out mean of .*?: (\{.*?\})", text).group(1))
    strategy = re.search(r"CV-chosen strategy: (.*)", text).group(1).strip()
    weights = re.search(r"CV-chosen variant for sub-05/06: \S+\s+weights \[(.*?)\]\s+\((.*)\)", text)
    first = weights.group(2).split(", ")[0].split(": ")[1]  # sub-01's vector (identical for all subjects unless per-subject)
    members = weights.group(1).split(" / ")
    per_subject = {p.split(": ")[0]: p.split(": ")[1] for p in weights.group(2).split(", ")}
    return {"loo": loo, "strategy": strategy, "score": loo[strategy], "weights": first, "members": members,
            "per_subject": per_subject, "name": re.search(r"CV-chosen variant for sub-05/06: (\S+)", text).group(1)}


def dev_stats(exp: str, ref: str | None, subjects=(1, 2, 3, 4)) -> dict:
    sim_dir = Path("results/similarity")
    r = trial_ranks(exp, list(subjects), DEFAULT_ROOT, sim_dir)
    allr = np.concatenate([r[s] for s in subjects])
    out = {"top1": np.mean([np.mean(r[s] <= 1) for s in subjects]), "top5": np.mean([np.mean(r[s] <= 5) for s in subjects]),
           "med": np.mean([np.median(r[s]) for s in subjects]), "per_subject": [np.mean(r[s] <= 1) for s in subjects]}
    if ref:
        b = np.concatenate([v for v in trial_ranks(ref, list(subjects), DEFAULT_ROOT, sim_dir).values()])
        d = b - allr
        out |= {"better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                "p": float(wilcoxon(b, allr).pvalue) if np.any(d != 0) else float("nan")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--reference", default="ens_vj_b_cv")
    args = ap.parse_args()
    rows = []
    for log in sorted(args.log_dir.glob("fin*_*.log")):
        try:
            info = parse_log(log)
        except (AttributeError, KeyError):
            continue  # run not finished
        exp = info["name"]
        try:
            info["dev"] = dev_stats(exp, args.reference)
        except FileNotFoundError:
            continue
        info["tag"] = log.stem
        rows.append(info)
    print("| run | members | CSLS | center | CV-chosen weights | LOFO (best single / equal / global / per-subject) | strategy | LOFO chosen | dev top-1 / top-5 / median | vs reference (better / worse, p) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["score"]):
        tag = r["tag"].split("_")
        n, ctr, csls = tag[0][3:], tag[1] == "ctr", tag[2]
        csls_txt = {"off": "off", "b5": "k=5 after blend", "b10": "k=10 after blend", "m5": "k=5 per member", "m10": "k=10 per member"}[csls]
        d, l = r["dev"], r["loo"]
        print(f"| {r['tag']} | {n} | {csls_txt} | {'yes' if ctr else 'no'} | {r['weights']} ({r['strategy']}) | "
              f"{l['best single']:.5f} / {l['equal']:.5f} / {l['global']:.5f} / {l['per-subject']:.5f} | {r['strategy']} | **{r['score']:.5f}** | "
              f"{d['top1']:.1%} / {d['top5']:.1%} / {d['med']:.1f} | {d['better']} / {d['worse']}, {d['p']:.2g} |")


if __name__ == "__main__":
    main()
