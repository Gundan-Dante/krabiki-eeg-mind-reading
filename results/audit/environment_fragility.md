# Environment and fragility (item 4)

## Package pins (7297a59 `requirements.txt` vs imports actually used in `src/` + `scripts/`)
Every third-party import is pinned, and the pins equal the installed versions: numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.1, pandas 3.0.6, torch 2.11.0+cu128, torchvision 0.26.0+cu128, open_clip_torch 3.3.0, timm 1.0.30,
transformers 5.18.0, braindecode 1.8.1, opencv-python-headless 5.0.0.93, pillow 13.0.0, tqdm 4.70.1. Python 3.11.17
(uv-managed); CUDA 12.8 wheels via `--extra-index-url https://download.pytorch.org/whl/cu128`. **PASS.**
Caveat: requirements.txt is a full `pip freeze` (also unused packages such as mne / aiohttp); harmless.

## Pretrained weights
| Model | How it's loaded | Pinned? | Locally cached snapshot |
|---|---|---|---|
| open_clip ViT-B-32 laion2b_s34b_b79k | open_clip tag | **no revision** (tag only) | HF laion/CLIP-ViT-B-32-laion2B-s34B-b79K `1a25a446712ba5ee05982a381eed697ef9b435cf` |
| DINOv2 ViT-B/14 `vit_base_patch14_dinov2.lvd142m` | `timm.create_model(pretrained=True)` | **no revision** | HF timm/... `4685c99dabffe5affac90bd99dbffd25801ae58d` |
| V-JEPA 2 ViT-L fpc64-256 | transformers `from_pretrained(revision=...)` | **yes** `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | same |
| AlexNet IMAGENET1K_V1 | torchvision weights enum | yes (versioned file `alexnet-owt-7be5be79.pth`) | same |
**CONCERN (minor):** pin the CLIP and DINOv2 revisions, or ship the feature caches, for exact reproduction.

## Numeric precision
- Every all-frames cache **stores** per-frame embeddings as float16 (`features.py:258`). Means are averaged from those
  fp16 values, while `clip_b32_f8` (the PIL path) keeps float32 means.
- **Compute precision:** V-JEPA 2 runs in **fp16** (`_vjepa2_l` default dtype). CLIP B-32 (both paths), CLIP block-8,
  DINOv2 and AlexNet run in fp32. CLIP L-14 (not in the headline) runs in fp16.
- GPU float32 ridge / CV vs the numpy/sklearn reference: parity tests pass (`tests/test_gpu.py`); near-ties can flip a few
  ranks on CPU vs GPU, so record the device.
- **fp16 vs fp32 V-JEPA re-extraction: NOT RUN** (time limit). `scripts/audit_controls.py --mode fp16` implements it
  (fp32 and fp16 re-extraction of the 102 test videos, cosine to the cache, dev top-1 agreement of the decw V-JEPA member).
