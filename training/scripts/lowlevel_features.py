"""Build the low-level stimulus features (see ``eegmr.lowlevel`` for the exact definitions).

    python scripts/lowlevel_features.py alexnet-pool --pool 3 6 9        # from the cached conv2 maps (features/alexnet_f8.npz)
    python scripts/lowlevel_features.py alexnet-layers --layers c1 c3 c4 c5 --pool 6   # GPU, torchvision AlexNet IMAGENET1K_V1
    python scripts/lowlevel_features.py descriptors                      # colour / frequency / edge / motion, CPU, 8 workers
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegmr.data import DEFAULT_ROOT
from eegmr.features import DEFAULT_FEATURE_DIR
from eegmr.lowlevel import ALEX_TAPS, extract_alexnet_layers, extract_descriptors, pool_cached_alexnet_conv2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("alexnet-pool")
    a.add_argument("--pool", type=int, nargs="+", default=[6])
    b = sub.add_parser("alexnet-layers")
    b.add_argument("--layers", nargs="+", choices=sorted(ALEX_TAPS), default=["c1", "c3", "c4", "c5"])
    b.add_argument("--pool", type=int, default=6)
    sub.add_parser("descriptors")
    for s in (a, b, sub.choices["descriptors"]):
        s.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        s.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    args = p.parse_args()
    if args.cmd == "alexnet-pool":
        for k in args.pool:
            print("wrote", pool_cached_alexnet_conv2(k, 8, args.feature_dir))
    elif args.cmd == "alexnet-layers":
        print("wrote", *extract_alexnet_layers(args.layers, args.pool, 8, args.root, args.feature_dir))
    else:
        print("wrote", *extract_descriptors(args.root, args.feature_dir))


if __name__ == "__main__":
    main()
