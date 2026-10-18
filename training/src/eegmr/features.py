"""Frame sampling and frozen pretrained stimulus features, cached to ``features/<name>_f<n>.npz``.

Pretrained models (all public, released well before Oct 16, 2026):
    alexnet   torchvision AlexNet, weights IMAGENET1K_V1, layer ``features.3`` (conv2, pre-ReLU)
    clip_b32  open_clip ViT-B-32 / laion2b_s34b_b79k, image embedding (``encode_image``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from eegmr.data import ALL_IDS, DEFAULT_ROOT, video_path

DEFAULT_FEATURE_DIR = Path("features")
MAX_PER_FRAME_DIM = 4096  # per-frame features are cached only when they are this small

Extractor = Callable[[list[np.ndarray]], np.ndarray]


def read_frames(path: Path, n_frames: int) -> list[np.ndarray]:
    """Return ``n_frames`` RGB frames at the centres of ``n_frames`` equal segments of the video.

    ``n_frames=1`` gives the middle frame.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    idx = ((np.arange(n_frames) + 0.5) * len(frames) / n_frames).astype(int)
    return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx]


def _alexnet(device: str) -> Extractor:
    from torchvision.models import AlexNet_Weights, alexnet

    net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features[:4].eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(frames: list[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in frames])
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
        return net((x - mean) / std).flatten(1).cpu().numpy()

    return run


# Pinned public checkpoints (exact files and Hugging Face revisions used for the cached features).
CLIP_B32_REPO, CLIP_B32_REVISION = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "1a25a446712ba5ee05982a381eed697ef9b435cf"
CLIP_B32_FILE = "open_clip_model.safetensors"  # open_clip ViT-B-32 / laion2b_s34b_b79k
DINOV2_B_REPO, DINOV2_B_REVISION = "timm/vit_base_patch14_dinov2.lvd142m", "4685c99dabffe5affac90bd99dbffd25801ae58d"
DINOV2_B_FILE = "model.safetensors"  # timm vit_base_patch14_dinov2.lvd142m (DINOv2 ViT-B/14)


def _pinned_file(repo: str, filename: str, revision: str) -> str:
    """Local path of ``filename`` at an exact Hugging Face ``revision`` (downloaded once into the HF cache)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, revision=revision)


def _clip_b32_model():
    """open_clip ViT-B-32 / laion2b_s34b_b79k loaded from the pinned checkpoint; returns ``(model, preprocess)``."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=_pinned_file(CLIP_B32_REPO, CLIP_B32_FILE, CLIP_B32_REVISION),
        image_mean=open_clip.OPENAI_DATASET_MEAN, image_std=open_clip.OPENAI_DATASET_STD,
        image_interpolation="bicubic", image_resize_mode="shortest")  # the laion2b_s34b_b79k preprocessing config
    return model, preprocess


def _clip_b32(device: str) -> Extractor:
    from PIL import Image

    model, preprocess = _clip_b32_model()
    model = model.eval().to(device)

    @torch.no_grad()
    def run(frames: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
        return model.encode_image(x).float().cpu().numpy()

    return run


EXTRACTORS: dict[str, Callable[[str], Extractor]] = {"alexnet": _alexnet, "clip_b32": _clip_b32}


def feature_path(name: str, n_frames: int, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    return Path(feature_dir) / f"{name}_f{n_frames}.npz"


def extract_features(
    name: str,
    n_frames: int,
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    """Compute features for all 1,102 videos and cache ``ids``, ``mean`` and (if small) ``frames``."""
    extractor = EXTRACTORS[name](device)
    means, per_frame = [], []
    for vid in tqdm(ALL_IDS, desc=f"{name} f{n_frames}"):
        feats = extractor(read_frames(video_path(int(vid), root), n_frames))
        means.append(feats.mean(axis=0))
        if feats.shape[1] <= MAX_PER_FRAME_DIM:
            per_frame.append(feats.astype(np.float16))
    arrays = {"ids": ALL_IDS, "mean": np.stack(means).astype(np.float32)}
    if per_frame:
        arrays["frames"] = np.stack(per_frame)
    out = feature_path(name, n_frames, feature_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


# ---------------------------------------------------------------------------------------------------------------
# All-frames extraction: decode each video once (DataLoader workers), preprocess + embed every frame on the GPU.
# Any n-frame variant is then an exact index subset (same indices as read_frames), so it needs no re-extraction.
# Cache: features/<name>_all.npz with ids, n_video_frames (1102,), frames (1102, max_frames, D) float16 (zero-padded).
# Derived caches are written as features/<name>_all_f<n>.npz, so load_features(f"{name}_all", n, ...) works.
# ---------------------------------------------------------------------------------------------------------------

GpuEncoder = Callable[[torch.Tensor], torch.Tensor]  # (B, 3, H, W) float in [0, 1] on device -> (B, D)


def _clip_b32_gpu(device: str) -> GpuEncoder:
    """open_clip ViT-B-32 / laion2b_s34b_b79k with its preprocessing (bicubic resize to 224, CLIP mean/std) on GPU.

    The videos are square, so open_clip's resize + centre crop reduces to a plain resize.
    """
    import open_clip

    model = _clip_b32_model()[0].eval().to(device)
    mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        return model.encode_image((x.clamp(0, 1) - mean) / std).float()

    return run


def _dinov2_b_gpu(device: str) -> GpuEncoder:
    """timm ``vit_base_patch14_dinov2.lvd142m`` (DINOv2 ViT-B/14) at 224 px (16 x 16 patches), ImageNet mean/std.

    Returns ``[CLS token, mean of patch tokens]`` from the final normed layer, i.e. 768 + 768 = 1536 dims per frame.
    Use :func:`split_dinov2` to get the two halves as separate feature caches.
    """
    import timm

    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224,
                              pretrained_cfg_overlay=dict(file=_pinned_file(DINOV2_B_REPO, DINOV2_B_FILE, DINOV2_B_REVISION))).eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    n_prefix = model.num_prefix_tokens

    @torch.no_grad()
    def run(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        tokens = model.forward_features((x.clamp(0, 1) - mean) / std)  # (B, prefix + 256, 768)
        return torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=1).float()

    return run


CLIP_L14_PRETRAINED = "laion2b_s32b_b82k"  # HF laion/CLIP-ViT-L-14-laion2B-s32B-b82K, revision 1627032197142fbe2a7cfec626f4ced3ae60d07a (2022-09-14, MIT)


def _clip_l14_gpu(device: str) -> GpuEncoder:
    """open_clip ViT-L-14 / laion2b_s32b_b82k image embedding (768-d), fp16 on GPU, preprocessing from the model's own
    config (bicubic resize to 224, its mean/std). The videos are square, so resize + centre crop is a plain resize."""
    import open_clip

    model = open_clip.create_model_and_transforms("ViT-L-14", pretrained=CLIP_L14_PRETRAINED)[0].eval().to(device).half()
    cfg = model.visual.preprocess_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        return model.encode_image(((x.clamp(0, 1) - mean) / std).half()).float()

    return run


def _clip_b32_mid_gpu(device: str, block: int = 8) -> GpuEncoder:
    """CLIP B-32 (laion2b_s34b_b79k) residual-stream CLS token after ``block`` of its 12 transformer blocks (768-d)."""
    import open_clip

    model = _clip_b32_model()[0].eval().to(device)
    mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)
    grabbed: dict[str, torch.Tensor] = {}
    model.visual.transformer.resblocks[block - 1].register_forward_hook(lambda m, i, o: grabbed.__setitem__("x", o))

    @torch.no_grad()
    def run(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        model.encode_image((x.clamp(0, 1) - mean) / std)
        h = grabbed["x"]
        return (h[:, 0] if model.visual.transformer.batch_first else h[0]).float()

    return run


GPU_ENCODERS: dict[str, Callable[[str], GpuEncoder]] = {"clip_b32": _clip_b32_gpu, "dinov2_b": _dinov2_b_gpu,
                                                       "clip_l14": _clip_l14_gpu, "clip_b32_mid8": _clip_b32_mid_gpu}
# Other residual-stream depths of the same CLIP B-32 tower (CLS token after block 2 / 4 / 6 / 10).
GPU_ENCODERS.update({f"clip_b32_mid{b}": (lambda device, b=b: _clip_b32_mid_gpu(device, block=b)) for b in (2, 4, 6, 10)})


def split_dinov2(n_frames: int, feature_dir: Path = DEFAULT_FEATURE_DIR, name: str = "dinov2_b") -> list[Path]:
    """Split ``<name>_all_f<n>.npz`` ([CLS, patch mean]) into ``<name>_cls_all_f<n>`` and ``<name>_patch_all_f<n>``."""
    with np.load(feature_path(f"{name}_all", n_frames, feature_dir)) as f:
        ids, mean, frames = f["ids"], f["mean"], f["frames"]
    half = mean.shape[1] // 2
    outs = []
    for part, cols in (("cls", slice(None, half)), ("patch", slice(half, None))):
        out = feature_path(f"{name}_{part}_all", n_frames, feature_dir)
        np.savez(out, ids=ids, mean=mean[:, cols], frames=frames[..., cols])
        outs.append(out)
    return outs


class _VideoFrames(torch.utils.data.Dataset):
    """All RGB frames of one video as uint8 ``(F, H, W, 3)``."""

    def __init__(self, ids: np.ndarray, root: Path) -> None:
        self.ids, self.root = ids, root

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> np.ndarray:
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(str(video_path(int(self.ids[i]), self.root)))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"could not decode video {self.ids[i]}")
        return np.stack(frames)


def all_frames_path(name: str, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    return Path(feature_dir) / f"{name}_all.npz"


def extract_all_frames(
    name: str,
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
) -> Path:
    """Embed every frame of all 1,102 videos and cache them (see the section comment above)."""
    encode = GPU_ENCODERS[name](device)
    loader = torch.utils.data.DataLoader(_VideoFrames(ALL_IDS, root), batch_size=None, num_workers=num_workers)
    per_video = []
    for frames in tqdm(loader, desc=f"{name} all frames"):
        x = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
        per_video.append(encode(x).cpu().numpy().astype(np.float16))
    counts = np.array([len(v) for v in per_video])
    out_frames = np.zeros((len(per_video), counts.max(), per_video[0].shape[1]), dtype=np.float16)
    for i, v in enumerate(per_video):
        out_frames[i, : len(v)] = v
    out = all_frames_path(name, feature_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ALL_IDS, n_video_frames=counts, frames=out_frames)
    return out


class _VideoSubset(_VideoFrames):
    """Only the ``n_frames`` frames :func:`read_frames` would pick, uint8 ``(n, H, W, 3)``."""

    def __init__(self, ids: np.ndarray, root: Path, n_frames: int) -> None:
        super().__init__(ids, root)
        self.n_frames = n_frames

    def __getitem__(self, i: int) -> np.ndarray:
        frames = super().__getitem__(i)
        return frames[((np.arange(self.n_frames) + 0.5) * len(frames) / self.n_frames).astype(int)]


def extract_frame_subset(
    name: str,
    n_frames: int,
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 3,
) -> Path:
    """GPU-embed only the ``n_frames`` frames of each video (``read_frames`` indices) and write ``<name>_all_f<n>.npz``
    directly (the same file :func:`derive_frame_subset` would give, without embedding every frame)."""
    encode = GPU_ENCODERS[name](device)
    loader = torch.utils.data.DataLoader(_VideoSubset(ALL_IDS, root, n_frames), batch_size=None, num_workers=num_workers)
    per_video = []
    for frames in tqdm(loader, desc=f"{name} f{n_frames}"):
        per_video.append(encode(frames.to(device).permute(0, 3, 1, 2).float() / 255.0).cpu().numpy())
    picked = np.stack(per_video).astype(np.float16)
    out = feature_path(f"{name}_all", n_frames, feature_dir)
    np.savez(out, ids=ALL_IDS, mean=np.stack(per_video).mean(axis=1).astype(np.float32), frames=picked)
    return out


def derive_frame_subset(name: str, n_frames: int, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    """Write ``<name>_all_f<n>.npz`` (ids, mean, frames) from the all-frames cache, with read_frames' indices."""
    with np.load(all_frames_path(name, feature_dir)) as f:
        ids, counts, frames = f["ids"], f["n_video_frames"], f["frames"]
    picked = np.stack([frames[i, ((np.arange(n_frames) + 0.5) * c / n_frames).astype(int)]
                       for i, c in enumerate(counts)])
    out = feature_path(f"{name}_all", n_frames, feature_dir)
    np.savez(out, ids=ids, mean=picked.astype(np.float32).mean(axis=1), frames=picked)
    return out


VIDEO_SECONDS = 3.0


def frame_times(n_frames: int, duration: float = VIDEO_SECONDS) -> np.ndarray:
    """Centre time (s) of each sampled frame, matching :func:`read_frames`."""
    return (np.arange(n_frames) + 0.5) * duration / n_frames


def pool_frames(frames: np.ndarray, pooling: str) -> np.ndarray:
    """Pool per-frame features ``(n, F, D)`` into one vector per video.

    ``first:<s>``      mean of the frames centred before ``s`` seconds (onset-weighted)
    ``concat``         all frames side by side, ``(n, F*D)``
    ``segments:<k>``   mean within k equal consecutive segments, concatenated, ``(n, k*D)``
    """
    kind, _, arg = pooling.partition(":")
    if kind == "first":
        keep = frame_times(frames.shape[1]) < float(arg)
        assert keep.any(), f"no frame centred before {arg} s"
        return frames[:, keep].mean(axis=1)
    if kind == "concat":
        return frames.reshape(len(frames), -1)
    if kind == "segments":
        return np.concatenate([seg.mean(axis=1) for seg in np.array_split(frames, int(arg), axis=1)], axis=1)
    raise ValueError(f"unknown pooling {pooling!r}")


def load_features(name: str, n_frames: int, ids: np.ndarray, feature_dir: Path = DEFAULT_FEATURE_DIR,
                  pooling: str = "mean") -> np.ndarray:
    """Pooled features ``(len(ids), D')`` for the requested video IDs (``pooling="mean"``: the cached frame mean)."""
    with np.load(feature_path(name, n_frames, feature_dir)) as f:
        all_ids = f["ids"]
        values = f["mean"] if pooling == "mean" else pool_frames(f["frames"].astype(np.float32), pooling)
    rows = np.searchsorted(all_ids, ids)
    assert np.array_equal(all_ids[rows], ids), "requested IDs missing from feature cache"
    return values[rows]


# ---------------------------------------------------------------------------------------------------------------
# Clip-level video encoder (V-JEPA 2): one forward pass embeds a whole clip of n frames, so there are no per-frame
# features. Cache: features/<name>_all_f<n>.npz with ids, mean (1102, D) and frames (1102, S, D), where "frames" holds
# the mean token embedding of S consecutive temporal segments (for segment pooling). n in the file name is the number
# of frames fed to the model, so load_features("vjepa2_l_all", 16, ids) works. The frames follow read_frames' indices.
# ---------------------------------------------------------------------------------------------------------------

VJEPA2_REPO = "facebook/vjepa2-vitl-fpc64-256"  # MIT licence; pretrained on VideoMix22M (SSv2, Kinetics, HowTo100M, YT-Temporal-1B, ImageNet)
VJEPA2_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
VJEPA2_SEGMENTS = 4


def _vjepa2_l(device: str, dtype: torch.dtype = torch.float16, crop: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """V-JEPA 2 ViT-L/16 (fpc64, 256 px, tubelet 2) via transformers. Input: one clip ``(T, 3, H, W)`` in [0, 1].

    Preprocessing as in the model's video_preprocessor_config: bilinear resize to 292 px, centre crop 256, ImageNet
    mean/std (``crop=False`` resizes straight to 256 and keeps the whole frame instead). Output: ``(S, 1024)``, the mean of
    the encoder's final token embeddings (``get_vision_features``) within S temporal segments; the mean over segments is
    the mean over all tokens. Tokens are ordered time-major (tubelet grid T/2 x 16 x 16).
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(VJEPA2_REPO, revision=VJEPA2_REVISION, dtype=dtype).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(clip: torch.Tensor) -> torch.Tensor:
        side = 292 if crop else 256
        x = torch.nn.functional.interpolate(clip, size=(side, side), mode="bilinear", antialias=True, align_corners=False)
        if crop:
            o = (side - 256) // 2
            x = x[..., o:o + 256, o:o + 256]
        x = ((x.clamp(0, 1) - mean) / std).unsqueeze(0).to(dtype)  # (1, T, 3, 256, 256)
        tokens = model.get_vision_features(pixel_values_videos=x).float()  # (1, T/2 * 256, 1024)
        t = clip.shape[0] // 2
        tokens = tokens.view(t, -1, tokens.shape[-1])  # (T/2, 256, D)
        return torch.stack([s.mean(dim=(0, 1)) for s in tokens.tensor_split(min(VJEPA2_SEGMENTS, t), dim=0)])

    return run


CLIP_ENCODERS: dict[str, Callable[..., Callable[[torch.Tensor], torch.Tensor]]] = {"vjepa2_l": _vjepa2_l}


class _VideoClip(torch.utils.data.Dataset):
    """``n_frames`` RGB frames of a video, uint8 ``(n, H, W, 3)``, at the same indices as :func:`read_frames`."""

    def __init__(self, ids: np.ndarray, root: Path, n_frames: int) -> None:
        self.ids, self.root, self.n_frames = ids, root, n_frames

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> np.ndarray:
        cv2.setNumThreads(1)
        return np.stack(read_frames(video_path(int(self.ids[i]), self.root), self.n_frames))


def extract_clip_features(
    name: str,
    n_frames: int,
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ids: np.ndarray = ALL_IDS,
    dtype: torch.dtype = torch.float16,
    num_workers: int = 3,
    crop: bool = True,
) -> Path:
    """Embed each video as one clip of ``n_frames`` frames and cache ``<name>_all_f<n>.npz`` (see section comment)."""
    encode = CLIP_ENCODERS[name](device, dtype, crop)
    loader = torch.utils.data.DataLoader(_VideoClip(ids, root, n_frames), batch_size=None, num_workers=num_workers)
    segs = []
    for clip in tqdm(loader, desc=f"{name} f{n_frames}", total=len(ids)):
        segs.append(encode(clip.to(device).permute(0, 3, 1, 2).float() / 255.0).cpu().numpy())
    segs = np.stack(segs)
    out = feature_path(f"{name}_all", n_frames, feature_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ids, mean=segs.mean(axis=1).astype(np.float32), frames=segs.astype(np.float16))
    return out
