"""Cache frozen stimulus features for all 1,102 videos.

Example:
    python scripts/extract_features.py --model clip_b32 --frames 1 8
    python scripts/extract_features.py --model dinov2_b --all-frames --frames 8   # GPU, every frame, then f8 subset
    python scripts/extract_features.py --model vjepa2_l --frames 16               # clip-level model, one pass per video
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegmr.data import DEFAULT_ROOT
from eegmr.features import (CLIP_ENCODERS, DEFAULT_FEATURE_DIR, EXTRACTORS, GPU_ENCODERS, derive_frame_subset, extract_all_frames,
                            extract_clip_features, extract_features, extract_frame_subset, split_dinov2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(set(EXTRACTORS) | set(GPU_ENCODERS) | set(CLIP_ENCODERS)))
    p.add_argument("--frames", type=int, nargs="+", default=[8], help="frames per video (one cache file each)")
    p.add_argument("--all-frames", action="store_true",
                   help="embed every frame on the GPU once, then derive the --frames subsets from that cache")
    p.add_argument("--subset-only", action="store_true",
                   help="GPU encoders: embed only the --frames sampled frames of each video instead of every frame")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = p.parse_args()

    kwargs = {"device": args.device} if args.device else {}
    if args.model in CLIP_ENCODERS:
        for n in args.frames:
            print("wrote", extract_clip_features(args.model, n, args.root, args.feature_dir, **kwargs))
        return
    if args.subset_only:
        for n in args.frames:
            print("wrote", extract_frame_subset(args.model, n, args.root, args.feature_dir, **kwargs))
        return
    if args.all_frames:
        print("wrote", extract_all_frames(args.model, args.root, args.feature_dir, **kwargs))
        for n in args.frames:
            print("wrote", derive_frame_subset(args.model, n, args.feature_dir))
            if args.model.startswith("dinov2"):
                print("wrote", *split_dinov2(n, args.feature_dir, args.model))
        return
    for n in args.frames:
        print("wrote", extract_features(args.model, n, args.root, args.feature_dir, **kwargs))


if __name__ == "__main__":
    main()
