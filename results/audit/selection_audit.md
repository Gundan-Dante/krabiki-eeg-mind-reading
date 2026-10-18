# Selection audit: which choices were made by train-CV vs by looking at dev

Sources: MODEL_BACKLOG.md (M0–M18, M14, M17-final), PROGRESS.md, IMPLEMENTATION_AUDIT.md, CONTRASTIVE_AUDIT.md, the
results/*.csv files, and this session's own history. "dev" = sub-01..04 test sets with labels.

| Decision in the headline pipeline | How it was made | Dev-influenced? |
|---|---|---|
| Ensemble weights (0.0 / 0.4 / 0.3 / 0.2 / 0.1) and strategy (global) | exhaustive simplex search on training-only CV; strategy by leave-one-fold-out on training folds | **no** |
| Member alphas, whiteners, early stopping | training-only CV / training data | **no** |
| Window 50–600 ms (samples 5:60) | **first picked by comparing windows on dev** (early encoding runs, 2026-10-17); later confirmed by train-CV (encoding sweep: 5:60 best of 12 windows; the decoder's CV picked 5:60 for every subject) | **yes, initially**; CV-confirmed |
| Crop 0–1.5 s | dev (early); irrelevant for every window-restricted head (bit-identical at fixed alpha) | yes, but no effect |
| Decoding over encoding (M3) | dev and train-CV agree | partly (both) |
| Removing the post-PCA scaler (M2) | found analytically; confirmed by CV and dev | partly |
| Noise-whitened decoder input (M8a) | train-CV (better on all 6 subjects); dev confirms | no (dev confirmation only) |
| **Rejecting reliability weighting (M8b)** | **CV preferred it on all 6 subjects; it was rejected because dev did not confirm** | **yes** (dev overrode CV) |
| Frame count / pooling (M6) | CV and dev both flat; kept the default | no effect |
| Feature sets / candidate members (DINOv2 patch vs CLS, V-JEPA, block-8 CLS, L-14 rejected) | proposed by CV, each confirmed on dev before entering the candidate pool; the final weights by CV | **candidate pool partly dev-influenced** (e.g. L-14 dropped "CV +0.0032, dev p 0.14") |
| Contrastive two-tower member (M12) | CV picked w = 0.75 vs decw; dev confirmed; the member then entered the CV search | candidate pool dev-checked |
| CSLS adoption (M18) and its stage (per member) | CV (dCV −0.006) + dev (p ~1e-12); stage `member` vs `blend` by LOFO (+0.003–0.004) | partly; the stage was chosen by CV |
| CSLS k = 5 | **k = 5 / 10 / 20 tied by CV (≤ 0.0005)**; M18 recommended k = 10; the final grid used k = 5 (fin5_ctr_m5); M17-final says "tied" | **unclear** which rule broke the tie; k is insensitive on dev too (17.4 / 17.4 / 17.2 top-1) |
| Test-time centering | M18: **dCV −0.0002 (≈ 0); adopted on dev p = 4e-4** ("CV understates it"); in the final grid LOFO +0.0008 with CSLS | **yes, mainly dev** at adoption; small CV support later |
| Not taking s3 (EEGNet) into the headline | LOFO gain like-for-like +0.0004 **and** "dev shows no top-1/top-5 trade" rule: dev p = 0.24 | **partly dev** (the rule includes a dev check) |
| AlexNet p6 6th member (fin6x, candidate v2) | LOFO 0.21235 → 0.23619 (training-only); weights by CV | no (per krabiki-51); its candidacy came from the low-level hypothesis, not dev |

**Summary:** the final weights, alphas and statistics are training-only. But the **candidate set** (which members and which
post-processing options were allowed into the search) was repeatedly screened on dev: window, centering, CSLS, member
pool, M8b rejection, s3 rule. That is legitimate model selection under rule 4, but it makes the dev numbers
(24.0 / 57.1) **optimistic estimates** of sub-05/06 performance. The effect is small per decision; across about 10
dev-screened decisions it could add up to a few points.
