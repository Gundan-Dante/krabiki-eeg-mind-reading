"""Cache audio embeddings for all 1,102 clips (track-audio).

Example:
    PYTHONPATH=src python scripts/extract_audio.py --model clap      # -> features/clap_all_f1.npz
    PYTHONPATH=src python scripts/extract_audio.py --model lowlevel  # -> features/low{vis,aud,av}_all_f1.npz (CPU)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eegmr.audio import AUDIO_ENCODERS, extract_audio_features, extract_lowlevel_av
from eegmr.data import DEFAULT_ROOT
from eegmr.features import DEFAULT_FEATURE_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="clap", choices=sorted(AUDIO_ENCODERS) + ["lowlevel"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = p.parse_args()

    if args.model == "lowlevel":
        for out in extract_lowlevel_av(args.root, args.feature_dir):
            with np.load(out) as f:
                print("wrote", out, "mean", f["mean"].shape, "frames", f["frames"].shape)
        return
    kwargs = {"device": args.device} if args.device else {}
    out = extract_audio_features(args.model, args.root, args.feature_dir, **kwargs)
    with np.load(out) as f:
        no_audio = f["no_audio"]
        print("wrote", out, "mean", f["mean"].shape, "frames", f["frames"].shape)
    print(f"{len(no_audio)} clips without usable audio ({(no_audio > 1000).sum()} of them test videos):", no_audio.tolist())


if __name__ == "__main__":
    main()
