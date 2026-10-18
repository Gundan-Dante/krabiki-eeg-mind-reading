# Krabiki: EEG mind reading (team submission)

For each test EEG recording (17 posterior channels, 500 Hz, 3 s, recorded while a person watched a 3 s video with sound)
we rank the 102 test videos (IDs 1001–1102) from most to least likely. The method is zero-shot retrieval: linear decoders
map the EEG into the feature spaces of frozen public models (image, video, audio, low-level), and an ensemble weighted
by cross-validation on training data ranks the candidates.

| | |
|---|---|
| **Predictions** | [`submissions/final_v4b_csls/`](submissions/final_v4b_csls/): `sub-05_ranked_candidate_list.npy`, `sub-06_ranked_candidate_list.npy` (int64, 102 × 102, rows in the original test order, values = video IDs 1001–1102) |
| **Dev score** (test sets of sub-01..04, labels used only for scoring) | **33.8% top-1 / 68.1% top-5**, median rank 2.6 (chance 1.0% / 4.9% / 51.5). Per subject: 35.3/71.6, 36.3/68.6, 26.5/65.7, 37.3/66.7 |
| **Reproduce the predictions** (CPU, numpy + scipy, seconds) | [`inference/`](inference/): `python inference/predict.py --data-root <eeg-mind-reading-challenge> --out out/` |
| **Retrain everything** (one GPU, Python 3.11) | [`training/`](training/): `bash training/run_pipeline.sh` |

## Repository layout

| Folder | What it contains |
|---|---|
| `submissions/final_v4b_csls/` | The submitted sub-05 / sub-06 ranked candidate lists. |
| `inference/` | Standalone predictor (numpy + scipy only, no GPU, no deep-learning libraries) and **our trained weights** (`inference/model/`): per subject and member the noise whiteners, standardization statistics, ridge maps / contrastive projections, plus the test-video features. It reproduces the submitted files bit-for-bit. See [`inference/README.md`](inference/README.md). |
| `training/` | The full training pipeline: package `eegmr` (`training/src/eegmr`), the scripts on the final path (`training/scripts`), tests, pinned `requirements.txt`, and `run_pipeline.sh` (features → decoders → subject pooling → CV-weighted ensemble → ranked lists → export to `inference/model`). See [`training/README.md`](training/README.md). |
| `docs/` | How the method was developed and checked: [`IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) (every step from 2% to 34%), [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) (pipeline, members, selection protocol). |
| `research/` | Experiment and audit scripts, and earlier submission sets, that are not needed for the final model. Kept for transparency. |
| `results/` | Logs of the experiments (`results.csv`, cross-validation tables). |

The data folder `eeg-mind-reading-challenge/` and the feature caches `features/` are not in the repository (videos and
frames must not be committed). Put the data folder at the repository root, or pass its path explicitly.

The rest of this document describes only the final model (v4b). How it was developed step by step is in
[`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md); the pipeline in more detail is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 1. Summary of the final model (v4b)

The test videos never appear in training, so we treat the task as **zero-shot retrieval**. For every subject we learn
linear maps from the EEG into seven frozen descriptions of a video: image, video, audio, low-level visual and low-level
audio-visual. Each map scores the 102 candidate videos by cosine similarity, and a weighted sum of the seven score
matrices ranks them. Every parameter is fitted on the subject's training recordings, plus other subjects' training
recordings for three members. Every choice (regularization, pooling weight, member weights) was made by
cross-validation on training videos.

| Dev result (test sets of sub-01..04, labels used only for scoring) | top-1 | top-5 | median rank |
|---|---|---|---|
| sub-01 / sub-02 / sub-03 / sub-04 | 35.3 / 36.3 / 26.5 / 37.3 % | 71.6 / 68.6 / 65.7 / 66.7 % | 2.0 / 3.0 / 3.0 / 2.5 |
| **Mean** | **33.8 %** | **68.1 %** | **2.6** |
| Chance | 1.0 % | 4.9 % | 51.5 |

The organizers' example recipe (AlexNet encoding model), re-implemented, gives 2.0 % / 8.6 % on the same dev sets.

## 2. Pipeline

```
EEG recording (17 ch x 1500 samples, 500 Hz)
  -> crop 0-1.5 s -> FIR anti-alias decimation x5 (zero phase) -> 17 x 150 at 100 Hz
  -> test-set centering (unsupervised)
  -> window 50-600 ms (17 x 55) -> per-timepoint noise whitening -> flatten (935) -> standardize
  -> 7 members: EEG -> feature space of one frozen model -> cosine vs the 102 candidate videos -> 102 x 102 scores
  -> per member: CSLS (k = 5) -> row z-score
  -> weighted sum (0.2 / 0.1 / 0.1 / 0.2 / 0.1 / 0.1 / 0.2) -> rank the 102 video IDs per recording
```

### 2.1 EEG preprocessing
- **Crop and decimate to 100 Hz; window 50–600 ms.** The recording is cropped to 0–1.5 s before filtering (no later
  samples leak in) and decimated with a zero-phase FIR filter. The decoders read 17 channels × 55 time points = 935
  values. Information peaks at 150–250 ms, and cross-validation preferred this window over wider or later ones for every
  subject.
- **Test-set centering.** The mean of a subject's 102 test recordings is replaced by its training mean. The test
  recordings are cleaner than single training trials (likely repeat-averaged) and their average response is shifted;
  centering removes that offset. It uses no labels (unsupervised normalization, rule 5).
- **Noise whitening.** EEG noise is shared by neighbouring electrodes. From training data only, we take the residuals
  of an out-of-fold encoding model (video features → EEG) and estimate, for each time point, a 17 × 17 Ledoit-Wolf noise
  covariance Σₜ. Multiplying each time point by Σₜ^(−1/2) makes that noise uncorrelated, so the stimulus-related signal
  stands out (+6 points top-5 for a single decoder).

### 2.2 Video descriptions (frozen public models, nothing fine-tuned)

| # | Feature set | Model and exact version | Descriptor |
|---|---|---|---|
| 1 | DINOv2 patch tokens | DINOv2 ViT-B/14, timm `vit_base_patch14_dinov2.lvd142m`, HF revision `4685c99dabffe5affac90bd99dbffd25801ae58d` | mean of the 256 patch tokens at 224 px, averaged over 8 evenly spaced frames (768-d) |
| 2 | V-JEPA 2 | `facebook/vjepa2-vitl-fpc64-256`, HF revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | one clip-level pass on 16 frames, mean of the final-layer tokens (1024-d) |
| 3 | CLIP image embedding | open_clip ViT-B-32 / `laion2b_s34b_b79k`, HF `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` revision `1a25a446712ba5ee05982a381eed697ef9b435cf` | image embedding averaged over 8 frames (512-d) |
| 4 | AlexNet conv5 | torchvision AlexNet `IMAGENET1K_V1` (`alexnet-owt-7be5be79.pth`) | pre-ReLU conv5 maps average-pooled to 6 × 6, averaged over 8 frames (9,216-d) |
| 5 | Hand-crafted descriptors | none (our code) | colour (Lab, hue), spatial-frequency and orientation power, Sobel edges, frame-difference and Farnebäck optical-flow motion (255-d) |
| 6 | CLAP audio | LAION-CLAP `laion/clap-htsat-unfused`, HF revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` | embedding of the clip's soundtrack (512-d). Clips without usable audio get the mean training embedding. |
| 7 | Low-level audio-visual time series | none (our code) | per 100 ms bin over 3 s: luminance, contrast, edges, motion, loudness (RMS), onset strength, 8 log-mel bands (30 × 14 = 420-d) |

Every feature is computed per video, from that video alone. Before use, features are standardized with the statistics
of the subject's training videos, for both the training targets and the 102 candidates.

### 2.3 The seven members

| Member | Feature set | Algorithm | Weight |
|---|---|---|---|
| A | 1 DINOv2 | subject-pooled whitened ridge decoder | 0.2 |
| B | 2 V-JEPA 2 | whitened ridge decoder | 0.1 |
| C | 3 CLIP | contrastive two-tower head (InfoNCE), 3 seeds | 0.1 |
| D | 4 AlexNet conv5 | subject-pooled whitened ridge decoder | 0.2 |
| E | 5 descriptors | whitened ridge decoder | 0.1 |
| F | 6 CLAP | whitened ridge decoder | 0.1 |
| G | 7 low-level AV | subject-pooled whitened ridge decoder | 0.2 |

- **Whitened ridge decoder (B, E, F, and the base of A, D, G).** Ridge regression maps the whitened EEG to the video's
  standardized feature vector, and each candidate is scored by the cosine between the predicted and its real vector.
  Decoding into a feature space beats the organizers' direction (predicting EEG from video): the comparison happens in a
  clean space instead of in noisy EEG (14.0 % vs 5.4 % top-1 with the same CLIP features). The ridge penalty is chosen per
  subject and member by retrieval cross-validation on training videos.
- **Subject pooling (A, D, G).** Each subject has only 850–972 noisy training trials. The pooled decoder also learns
  from the other five subjects' **training** recordings at a small weight λ, each whitened with its own noise model.
  λ is chosen per subject (0.05–0.1) by cross-validation in which the other subjects' recordings of held-out videos are
  removed, so it stays zero-shot. Full pooling (λ = 1) hurts because brains differ; a small share helps every subject,
  the weakest most.
- **Contrastive head (C).** Two linear towers map the EEG and the CLIP vector into a shared 256-d space and are trained
  with a symmetric InfoNCE loss (learnable temperature) to rank the true video first, with early stopping on held-out
  training videos. Three seeds are averaged. Weak alone, it makes different errors from the ridge decoders, which is
  why it earns a place in the ensemble.

### 2.4 Combining the members
1. **CSLS hub correction** (per member, k = 5): `2·S − r_rec − r_cand`, where `r_cand[j]` is the mean of candidate j's 5
   highest scores across the recordings. It lowers videos that score high for every recording. It uses no labels and
   was approved by the organizers.
2. **Row z-score** of each member's 102 × 102 matrix, then the **weighted sum** with the global weights above.
3. **Ranking**: candidates sorted by descending score (stable). Each row holds the IDs 1001–1102 and keeps the original
   order of the test recordings. The writer asserts shape, integer type and that every row is a permutation.

### 2.5 How the weights were chosen
- **CV data.** 5-fold cross-validation over each subject's training videos (`KFold(5, shuffle=True, random_state=0)`).
  It produces out-of-fold score matrices for every member, with whiteners, centering and CSLS applied inside each fold,
  as at test time.
- **Criterion.** The mean of expected top-1 and expected top-5 accuracy among 102 candidates (hypergeometric).
- **Search.** Exhaustive over all 8,008 weight vectors on the simplex in steps of 0.1.
- **Strategy.** Leave-one-fold-out compared best single member (0.189), equal weights (0.269), one global vector
  (**0.287**) and per-subject vectors (0.285). The global vector was chosen and is used for all six subjects.
- **Dev labels** (sub-01..04) were used only to confirm results (paired Wilcoxon on per-recording ranks), never to fit
  or select parameters.

## 3. Expected performance on sub-05 / sub-06
Without labels we can only estimate. Training cross-validation scaled by the test/CV ratio measured on sub-01..04
gives about **26 % top-1 / 60 % top-5 for sub-05** and **18 % / 49 % for sub-06**. These are probably optimistic: both
are the two weakest subjects by cross-validation, and sub-06 has the fewest training recordings (850).

## 4. Rules and validity
- **Environment and models.** Python 3.11. All external models are public, released before 2026-10-16, and loaded at
  the pinned revisions listed above. Versions of all packages are in `training/requirements.txt`.
- **No test data in training.** No test recording of any subject is used to fit anything. Test recordings enter only
  through centering on their own mean (unsupervised normalization) and CSLS on the unlabeled score matrix. Dev labels
  are read only after the ranked lists are written, and are never used as templates for sub-05 / sub-06.
- **Controls.**
  - With training EEG ↔ video pairs shuffled, the pipeline falls to chance.
  - Permuting the test recordings permutes the output rows exactly.
  - Reruns are bit-identical.
  - Rebuilding all seven members of sub-05 from scratch reproduces the submitted `sub-05` file exactly.

## 5. Reproducing
- **Predictions from our weights** (CPU, numpy + scipy, about 2 s): `python inference/predict.py --data-root
  <eeg-mind-reading-challenge> --out out/`. It reproduces `submissions/final_v4b_csls` byte-for-byte. Our trained
  parameters are in `inference/model/` (whiteners, standardization statistics, ridge maps, contrastive projections and
  candidate features per subject and member, with SHA-256 hashes in `meta.json`).
- **Full retraining** (one CUDA GPU): `bash training/run_pipeline.sh` runs feature extraction → decoders → subject
  pooling → ensemble → ranked lists → export. See [`training/README.md`](training/README.md).
