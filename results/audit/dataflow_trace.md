# Code-level data-flow trace: headline `final_csls` (fin5_ctr_m5, track-matching 7297a59)

Audited in a detached checkout of 7297a59 (`/home/kparpie/Krabiki-audit`). The headline path is `scripts/ensemble.py` with
members `decw:clip_b32`, `decw:dinov2_b_patch_all`, `decw:vjepa2_l_all:16`, `con:clip_b32:two_tower_d256`,
`decw:clip_b32_mid8_all:8`, plus `--center-test --csls 5 --csls-stage member`. The candidate 6th member
`decw:alexnet_p6:8` (krabiki-51, uncommitted `src/eegmr/lowlevel.py`) uses the same `decw` code path and is covered too.

Method: `grep` for every use of `load_test`, `load_test_labels` / `tst_stim`, `TEST_IDS`, `center_like`, `csls` in `src/`
and `scripts/{ensemble,predict}.py`, then reading each call chain into `decoding.py`, `contrastive.py`, `gpu.py`,
`matching.py` and `features.py`.

## (a) Test EEG (`load_test`)
| Where | What is computed from it | Allowed? |
|---|---|---|
| `ensemble.py:456` `measured = preprocess(load_test(s))` | crop to 1.5 s + decimation (per recording, no statistics) | yes |
| `ensemble.py:232` `center_like(measured, eeg)` (`--center-test`) | the mean over the 102 test recordings (per channel × time) is replaced by the training mean. Uses no labels; transductive over the unlabeled test set | yes, rule 5 (unsupervised normalization of test recordings) |
| `decoding.DecodingVariant.similarity` → `decoder_input(eeg, window, whiteners)` | per-timepoint whitening with Σ⁻¹ᐟ² estimated on **training** OOF residuals; standardization with the **training** mean/std inside `gpu.ridge_basis` / `RidgeBasis.project` | yes |
| `contrastive.prepare(eeg_trn, eeg_new, ...)` | test inputs standardized with **training** statistics (`gpu.standardize(X_new, X_trn)`) | yes |
| `ensemble.py:470` `csls(t, 5)` per member | CSLS on each member's 102 × 102 test matrix: mean of the 5 largest similarities per row and per column. Uses no labels; transductive | yes, per the organizers (2026-10-18, see M17-final) |

No statistic that is **fitted** (scaler, PCA, whitener, ridge, alpha, ensemble weight, early stopping) sees test EEG. The two
transductive operations (centering, CSLS) use the unlabeled test set only and are permutation-equivariant (control 3b).

## (b) Dev labels (`load_test_labels` / `tst_stim`)
| Where | Use |
|---|---|
| `data.py:34` | loader only |
| `ensemble.py:469` `true_ids = load_test_labels(s)` | used only **after** `write_submission`, in `score_submission_file` / `evaluate` for the reported rows (`continue` when None) |
| `ensemble.py:486` | list of dev subjects for the printed report table |
| `compare_matching.compare_table` | reloads labels to print the paired Wilcoxon table (reporting) |
| `predict.py:78` | after `write_submission`, scoring only (not on the headline path) |

The weight search, the strategy choice (LOFO), the alphas, the whiteners and early stopping never call `load_test_labels`.
The training-only CV block (`ensemble.py` "1. training-only CV") calls only `load_train`.

## (c) Test-video features (`TEST_IDS`)
| Where | Use |
|---|---|
| `ensemble.py:465` `load_features(..., TEST_IDS, ...)` | candidate representations of the 102 test videos |
| `DecodingVariant.similarity` | candidates standardized with **training-video** statistics (`gpu.standardize(cand, features_trn)`) |
| `contrastive.prepare` | `V_new` standardized with `V_trn` statistics |
| `features.py` (all extractors), `lowlevel.pool_cached_alexnet_conv2` | computed **per video** from that video's frames only; no statistic across videos. Checked numerically for `alexnet_p6_f8.npz`: bit-identical to re-pooling `alexnet_f8.npz`, and test rows are unchanged when all training rows are zeroed |

## Templates across subjects
Each subject is processed independently: its own training data, folds, member fits and test similarities. Nothing from
sub-01..04's **test** data (EEG or labels) reaches sub-05/06. Global ensemble weights are chosen from the **training-only**
CV similarities of all six subjects (including sub-05/06's own training data, which is allowed).

## Caveats (not leaks)
- Member alphas are chosen by CV on each subject's full training set and then reused in the outer CV folds (not nested),
  so CV scores are slightly optimistic. No test data is involved.
- CSLS in the CV is applied to held-out folds of about 200 videos; at test time to 102.
