"""Per-subject and mean scores of logged experiments, with a paired Wilcoxon test on per-trial ranks vs a baseline.

Reads the similarity matrices that predict.py saves under results/similarity/<experiment>/ and scores them against
the dev subjects' test labels (validation only).

Example:
    python scripts/compare_matching.py --baseline match_base match_erp match_weighted --subjects 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegmr.data import DEFAULT_ROOT
from eegmr.evaluate import compare_table, trial_ranks  # noqa: F401  (re-exported for research scripts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True)
    p.add_argument("experiments", nargs="+", help="experiments to compare with the baseline")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--sim-dir", type=Path, default=Path("results/similarity"))
    args = p.parse_args()
    print(compare_table(args.baseline, args.experiments, args.subjects, args.root, args.sim_dir))


if __name__ == "__main__":
    main()
