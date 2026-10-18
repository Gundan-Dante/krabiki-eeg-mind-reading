"""Timing and precision check for the V-JEPA 2 extractor on a few clips (no cache is written).

    PYTHONPATH=src python scripts/time_vjepa2.py --clips 10 --frames 16 64
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from eegmr.data import ALL_IDS, DEFAULT_ROOT, video_path
from eegmr.features import CLIP_ENCODERS, read_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=10)
    p.add_argument("--frames", type=int, nargs="+", default=[16, 64])
    p.add_argument("--no-crop", action="store_true")
    args = p.parse_args()
    ids = ALL_IDS[np.linspace(0, len(ALL_IDS) - 1, args.clips).astype(int)]  # fixed, spread over train and test
    encoders = {}
    for dtype in (torch.float16, torch.float32):
        encoders[dtype] = CLIP_ENCODERS["vjepa2_l"]("cuda", dtype, not args.no_crop)
    for n in args.frames:
        clips = [torch.from_numpy(np.stack(read_frames(video_path(int(i), Path(DEFAULT_ROOT)), n))).cuda()
                 .permute(0, 3, 1, 2).float() / 255.0 for i in ids]
        out = {}
        for dtype, enc in encoders.items():
            enc(clips[0])  # warm-up
            torch.cuda.synchronize(), torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out[dtype] = torch.stack([enc(c) for c in clips]).cpu()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / len(clips)
            print(f"frames={n:3d} {str(dtype)[6:]:8s} {dt:6.3f} s/clip  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB  "
                  f"-> 1102 clips: {dt * 1102 / 60:5.1f} min", flush=True)
        a, b = out[torch.float16].mean(1), out[torch.float32].mean(1)  # (clips, D) mean over segments
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        print(f"frames={n:3d} fp16-vs-fp32 cosine: min {cos.min():.6f} mean {cos.mean():.6f}", flush=True)


if __name__ == "__main__":
    main()
