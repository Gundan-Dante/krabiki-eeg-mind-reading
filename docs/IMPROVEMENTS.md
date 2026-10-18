# How we got from 2% to 33%: every major improvement, explained

**Task:** while a person watches a 3-second video, we record 17 EEG channels at the back of the head. For each recording we
rank the 102 candidate test videos by how likely they caused it. Scored on top-1 and top-5 for subjects 5 and 6 (chance:
1.0% and 4.9%).

**How to read the numbers.** Dev = the test sets of subjects 1–4, for which labels are available (4 × 102 recordings). Every
design choice was made on **training data only** (cross-validation over each subject's training videos). Dev was only used
to confirm, with a paired significance test per recording. The pipeline is described
in `ARCHITECTURE.md`, the final model in `../README.md`.

| # | Step | Dev top-1 | Dev top-5 |
|---|---|---|---|
| 0 | Organizers' recipe, re-implemented (AlexNet, encoding) | 2.0% | 8.6% |
| 1 | Better image features (CLIP) | 4.2% | 16.9% |
| 2 | Fixing a hidden bug that disabled regularization | 5.4% | 21.1% |
| 3 | Flipping the direction: decoding instead of encoding | 14.0% | 36.0% |
| 4 | Removing correlated EEG noise ("noise whitening") | 14.7% | 42.2% |
| 5 | Several different feature models in one ensemble | 20.8% | 48.0% |
| 6 | Correcting "hub" videos (CSLS) + test-set centering | 24.0% | 57.1% |
| 7 | Low-level vision and sound | 32.4% | 66.7% |
| 8 | Learning a little from the other subjects (pooling) | **33.8%** | **68.1%** |

---

## 0. Starting point: the organizers' recipe (M0), 2.0% / 8.6%
Describe each video with AlexNet features (an early image network), learn a linear map (ridge regression) from those features
to the full EEG (**encoding**: video → predicted brain response), then find the test recording whose EEG correlates best
with each prediction. It's twice chance, but weak.

## 1. Better image features (M1), 4.2% / 16.9%
Replace AlexNet with **CLIP**, a model trained on 2 billion image–text pairs, so its features describe *meaning* ("a dog
on a beach"). EEG mostly carries the gist of what was seen, so meaning-level features match it better.

## 2. A hidden bug: regularization did nothing (M2), 5.4% / 21.1%
Ridge regression has a knob (alpha) that stops it from fitting noise. The recipe re-scaled the features after PCA so that every
direction was equally strong. Then alpha can only shrink all predictions *uniformly*, and the correlation matching ignores
that, so **the knob had no effect at all**. Removing the extra scaling made alpha work as intended: weak, noisy directions get
shrunk more.

## 3. The biggest single jump: decode instead of encode (M3), 14.0% / 36.0%
Instead of predicting 17 × 300 EEG numbers from the video (most of which are noise), we predict the **video's feature vector
from the EEG**: a ridge map from 17 channels × 55 time points (50–600 ms) to CLIP's 512 numbers. Then we compare the predicted
vector with each candidate's real vector by **cosine similarity**, i.e. by direction, since ridge makes predictions short.
- **Why it works:** the comparison happens in a clean, meaningful space instead of in noisy EEG space, and the decoder
  learns by itself which channels and time points carry information.
- **Check:** training on shuffled EEG–video pairs gives chance, so there's no leakage.

## 4. Removing correlated EEG noise (M5, M8a), 14.7% / 42.2%
EEG noise is shared across neighbouring electrodes: many channels pick up the same disturbance. From the training data we
estimate, **for each time point, how the noise is shared across the 17 electrodes** (from what the model *can't* explain),
and transform the EEG so that this shared noise becomes uncorrelated and evenly spread. The real signal then stands out.
Top-5 rose by 6 points. (It also nearly doubled the old encoding model: 5.4 → 8.8% top-1.)

## 5. An ensemble of different views (M11, M13, M17), 20.8% / 48.0%
We run the same decoder recipe with **different feature models** and add up their scores, with weights chosen by
cross-validation:
- **DINOv2** (self-supervised image model: shapes and objects),
- **V-JEPA 2** (a *video* model: motion and dynamics),
- a **contrastive head** (trained to rank the right video first, CLIP-style, instead of regressing).

Each sees the video differently, so their mistakes differ, and the sum is better than any single model. A bigger CLIP was
*worse*, so different beats bigger.

## 6. Correcting "hub" videos and the train/test shift (M18), 24.0% / 57.1%
- **Hubs:** some candidate videos score high for *every* recording. **CSLS** lowers each candidate's score by how popular it
  is across all recordings, so a video only wins if it fits *this* recording specifically. It uses no labels and was
  **approved by the organizers**.
- **Test-set centering:** the test recordings differ systematically from the training ones (cleaner, probably averaged
  over repeats, with a flipped average response). Subtracting the test set's own mean (allowed: unsupervised normalization)
  removes that shared offset.

## 7. Low-level vision and sound (M20–M25), 32.4% / 66.7%
Our electrodes all sit over the **visual cortex**, which reacts strongly to *how things look*, not only to what they mean.
- **AlexNet conv5 + hand-made descriptors** (edges, contrast, colour, spatial frequency, motion), krabiki-51: spatial
  low-level structure, which semantic models wash out.
- **lowav**: brightness, contrast, edges, motion **and loudness, onsets and sound texture per 100 ms**, Alikhan: the timing
  of visual and audio changes.
- **CLAP** (an audio model), Alikhan: the sound of the clip, which is the most different view of all.

Low-level members became the backbone of the ensemble. For the weakest subjects (5 and 6), cross-validation gives them the
most weight.

## 8. Learning a little from the other subjects (M26, M28), 33.8% / 68.1% (final)
Each subject has only ~900 noisy training recordings, and the decoders are data-limited. All six people watched largely the same
training videos, so each subject's decoder now also learns from the **other subjects' training recordings, at a small weight**
(5–20%, chosen per subject by cross-validation). Brains differ, so treating everyone equally hurts, but a little shared data
helps every member on every subject: cross-validation picked a pooling weight above 0 in 36 of 36 subject × member cases, and the **weakest subjects (5 and 6) gained the most**. The final model pools the three heaviest decoders (DINOv2, AlexNet conv5, lowav). Pooling all six added nothing more. Cross-validation never sees a held-out video through another subject, so the
evaluation stays zero-shot like the real test.

---

## What we tried that didn't make it (and why that's useful)
| Idea | Result |
|---|---|
| More frames per video (8 → 32) | no gain: frames within a 3 s clip are nearly identical for CLIP |
| Bigger CLIP (ViT-L/14) | worse than B-32: EEG carries gist, not fine detail |
| Wider / extra time windows | worse: the information sits in the first ~600 ms |
| Public THINGS-EEG2 data (same 17 channels) | transfers only ~2× chance: photos flashed for 100 ms ≠ 3 s videos |
| EEGNet (trained EEG network) | weaker than ridge alone; data-limited (its gap to ridge shrinks with more data) |
| Motion gate (trust video models for moving clips) | every model decodes *moving* clips worse, so the premise doesn't hold |
| Reliability weighting, ERP subtraction, time jitter | no gain on training CV, or not confirmed on dev |

## How we know it's real
- **Training-only selection:** every weight, alpha and member was chosen by cross-validation on training videos. Dev only
  confirmed.
- **Shuffle controls:** with video–EEG pairs scrambled in training, the full pipeline falls to chance (1.0% / 4.7%).
- **Independent reproductions:** key results were re-implemented independently and matched exactly. A teammate reproduced
  them on different hardware.
- **Honest expectation:** subjects 5 and 6 are the two weakest by cross-validation. The expected score there is roughly
  **~26% top-1 / ~60% top-5 for subject 5 and ~18% / ~49% for subject 6**, an estimate and probably optimistic.
