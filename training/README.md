# Training pipeline

Retrains the final model (v4b) from the raw challenge data: feature extraction with frozen public models, the seven
ensemble members per subject, the cross-validated weight search, the ranked candidate lists, and the export of the fitted
parameters to `inference/model/`.

## Setup

Python 3.11, one CUDA GPU (8 GB is enough; the pipeline also runs on CPU, much slower). The pinned environment uses
the CUDA 12.8 torch wheels:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt        # includes the --extra-index-url for the cu128 torch builds
# macOS / no CUDA: pip install -r training/requirements-cpu.txt   (same pins without CUDA; runs on CPU, slower)
```

Put the challenge data at `<repo>/eeg-mind-reading-challenge/` (or set `DATA=...`). Public model weights are downloaded
once from Hugging Face / torchvision at the pinned revisions (see the root README).

## Run

```bash
bash training/run_pipeline.sh                  # all stages
bash training/run_pipeline.sh ensemble verify  # or selected stages
```

| Stage | What it does | Output |
|---|---|---|
| `features` | Frame and audio features of all 1,102 videos: CLIP B-32 (8 frames), DINOv2-B patch tokens (every frame, then 8), V-JEPA 2-L (16 frames), AlexNet conv5 pooled 6 × 6, hand-crafted descriptors, CLAP audio, low-level audio-visual features per 100 ms | `features/*.npz` |
| `decw` | Per-subject alphas of the plain (unpooled) whitened decoders of the three pooled feature sets, chosen by training-only CV | `work/similarity_cache/` |
| `pool` | Subject-pooled decoders (`scripts/pool_member.py`): own training trials + the other subjects' training trials at weight λ. λ and alpha are chosen per subject by zero-shot CV, where the other subjects' recordings of held-out videos are dropped | `work/similarity_cache/` |
| `ensemble` | Remaining members, 5-fold training CV of all seven, exhaustive weight search on the simplex (step 0.1), strategy by leave-one-fold-out, then the test similarities, CSLS k=5 per member, the weighted blend, and the ranked lists | `work/submissions/v4b_ctr_m5_cv/` |
| `export` | Fitted parameters → `inference/model/` (numpy-only predictor) | `inference/model/` |
| `verify` | Rebuilt sub-05 / sub-06 lists vs the committed `submissions/final_v4b_csls/` | printed |

Environment variables: `DATA`, `FEATURES` (default `<repo>/features`), `WORK` (default `<repo>/work`), `PY` (Python
interpreter), `SUBJECTS`.

## Code map

| Where | What |
|---|---|
| `src/eegmr/data.py` | loading, crop + FIR decimation (`preprocess`) |
| `src/eegmr/features.py`, `lowlevel.py`, `audio.py` | frozen feature extractors (pinned revisions) and caches |
| `src/eegmr/decoding.py`, `matching.py`, `gpu.py`, `linear.py` | whitened ridge decoder, Ledoit-Wolf noise whitening, torch ridge / retrieval CV |
| `src/eegmr/contrastive.py` | InfoNCE two-tower head |
| `src/eegmr/retrieval.py`, `evaluate.py`, `submission.py` | CSLS; metrics; the validated submission writer |
| `scripts/ensemble.py` | members, training CV, weight search, test phase |
| `scripts/pool_member.py` | subject-pooled decoders |
| `scripts/extract_features.py`, `lowlevel_features.py`, `extract_audio.py` | feature CLIs |
| `scripts/export_bundle.py` | export to `inference/model/` |
| `tests/` | parity tests (torch ridge / CV vs sklearn, contrastive, low-level features): `PYTHONPATH=training/src:training/scripts pytest training/tests` |

The experiment heads that are not in the final model (`enc`, `enc_noise`, `zs`, `s3`) and the scripts behind the
earlier experiments live in `research/scripts/`. The `zs` and `s3` heads of `ensemble.py` need that folder on
`PYTHONPATH`.
