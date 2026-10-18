# Inference: ranked candidate lists from the trained model

This folder turns the challenge test EEG into the submission files. It needs **only numpy and scipy, no GPU, no
network, no PyTorch and none of the pretrained vision / audio models**: the fitted parameters of our final model and
the features of the 102 test videos are stored in [`model/`](model/). The training that produced those parameters is in
[`../training/`](../training/).

## Run

```bash
pip install -r inference/requirements.txt          # numpy==2.4.6, scipy==1.17.1 (Python 3.11)
python inference/predict.py --data-root /path/to/eeg-mind-reading-challenge --out predictions
```

`--data-root` is the challenge folder (it must contain `tst/sub-05_tst.npy` and `tst/sub-06_tst.npy`). Options:
`--subjects 5 6` (default), `--model inference/model` (default), `--no-verify` (skip the file hash check).

Output: `predictions/sub-05_ranked_candidate_list.npy` and `predictions/sub-06_ranked_candidate_list.npy`, each an `int64`
array of shape (102, 102). Row *i* belongs to the *i*-th recording of `tst/sub-0X_tst.npy` (original order), and the row
lists all video IDs 1001..1102 from most to least likely. The script validates the shape, the integer dtype and that every
row is a permutation of 1001..1102 before it finishes.

Runtime: about 2 seconds for both subjects on an 8-thread workstation CPU (measured), 265 MB peak memory.

## Check against the development subjects

With `--subjects 1 2 3 4` the script also scores its output against `tst_stim/` (the development labels, used for
scoring only; they never enter the model):

```
sub-01: ... top1=0.353 top5=0.716 median_rank=2.0
sub-02: ... top1=0.363 top5=0.686 median_rank=3.0
sub-03: ... top1=0.265 top5=0.657 median_rank=3.0
sub-04: ... top1=0.373 top5=0.667 median_rank=2.5
```

Mean over the four: **33.8% top-1, 68.1% top-5**, median rank 2.6 (chance 1.0% / 4.9% / 51.5).

## What the model is

The final ensemble ("v4b") has seven members. Each maps the first 1.5 s of EEG (17 channels, decimated to 100 Hz, noise-whitened)
to a feature space of the stimulus videos with ridge regression, except one contrastive head; the test videos are ranked
by similarity in each space, and the seven rankings are combined with fixed weights.

| member | features | weight |
|---|---|---|
| pooled ridge decoder | DINOv2 ViT-B/14 patch tokens | 0.2 |
| ridge decoder | V-JEPA 2 ViT-L (16 frames) | 0.1 |
| contrastive two-tower head | CLIP ViT-B/32 image embeddings | 0.1 |
| pooled ridge decoder | AlexNet conv5 maps (6 x 6 pooled) | 0.2 |
| ridge decoder | 255 hand-made low-level visual descriptors | 0.1 |
| ridge decoder | CLAP audio embeddings | 0.1 |
| pooled ridge decoder | per-100 ms low-level audio-visual features | 0.2 |

"Pooled" means the target subject's training trials plus the other subjects' training trials at a small weight. Every
parameter is estimated from the training recordings only. The test data enter in two label-free ways: the mean of the
102 test recordings is replaced by the training mean (test-time centering), and each similarity matrix is rescaled with
CSLS (k = 5). Unsupervised normalization of the test data is allowed by the challenge rules, and the organizers approved CSLS. The full description of the steps is the docstring of
[`predict.py`](predict.py); the public models, their exact versions and revisions, the per-subject hyperparameters and
the file formats are in [`model/README.md`](model/README.md) and [`model/meta.json`](model/meta.json).

## Where the files come from

`model/` is written by [`training/scripts/export_bundle.py`](../training/scripts/export_bundle.py) (model card by
`training/scripts/bundle_readme.py`), which refits every member from the training recordings with the hyperparameters
chosen by training-only cross-validation, and checks each refit against the live training pipeline and the cached
similarity matrices of the ensemble run (largest difference 3e-9). Command and details: the docstring of `export_bundle.py`
and [`../training/README.md`](../training/README.md). Every file is below 50 MB (largest: 34.7 MB) and
committed as a plain git file, so a normal `git clone` is enough. `meta.json` lists the SHA-256 of every file;
`predict.py` verifies them before use.

## Verification

```bash
pip install -r inference/requirements.txt
EEG_DATA_ROOT=/path/to/eeg-mind-reading-challenge python -m pytest inference/test_predict.py -q
```

The tests check that (1) every model file matches its hash and size limit, (2) `meta.json` is consistent with the files,
(3) the output validator and CSLS behave as specified, (4) sub-01..04 reproduce 138 / 408 top-1 and 278 / 408 top-5 hits
(33.8% / 68.1%), and (5) sub-05 and sub-06 are **bit-identical** to the committed
[`submissions/final_v4b_csls/`](../submissions/final_v4b_csls/) files. Tests 4 and 5 need the challenge data and are skipped
without it. Result when this folder was produced: 6 passed.
