# Submission sets (sub-05 / sub-06)

| Directory | Model | Dev (sub-01..04) top-1 / top-5 / median | Status |
|---|---|---|---|
| **`final_v4b_csls/`** | **FINAL (v4b, M28).** Subject-pooled DINOv2-patch, AlexNet conv5 and lowav decoders + V-JEPA 2, hand-crafted descriptors, CLAP + contrastive two-tower; global CV weights .2/.1/.1/.2/.1/.1/.2 (pooled patch / V-JEPA / contrastive / pooled conv5 / descriptors / CLAP / pooled lowav); test centering; CSLS k=5 per member | 33.8 / 68.1 / 2.6 | **final** (LOFO 0.2872; best training CV on sub-05 / sub-06; verified: rescore, files == saved matrices) |
| **`final_v4_csls/`** | v4a (M28): only DINOv2-patch pooled. v3 with the DINOv2-patch decoder replaced by a **subject-pooled** one (own trials + other subjects' training trials at λ chosen per subject by CV, zero-shot CV); global CV weights pooled-patch .2 / V-JEPA .1 / contrastive .1 / AlexNet conv5 .2 / descriptors .1 / CLAP .1 / lowav .2; test centering; CSLS k=5 per member | 33.3 / 68.9 / 2.5 | final (verified: rescore, files == saved matrices, pooling code uses only other subjects' training data) |
| **`final_v3_csls/`** | Previous headline (v3, M25). Whitened decoders on DINOv2-B patch, V-JEPA 2-L 16 f, AlexNet conv5 (6×6 pool), hand-crafted low-level descriptors, CLAP audio, low-level AV per 100 ms (lowav) + contrastive two-tower on CLIP; per-subject CV weights; test centering; CSLS k=5 per member | 32.4 / 66.7 / 3.0 | final (audited: rescore, shuffle control) |
| `final_v3_nocsls/` | v3 backup: same 7 members, no centering, no CSLS, global weights | 25.2 / 58.1 / 4.1 | backup |
| `final_csls/` | Previous headline (M17 + block-8). Whitened decoders on CLIP B-32 (w 0.0), DINOv2-B patch (0.4), V-JEPA 2-L 16 f (0.3), CLIP block-8 CLS (0.1) + contrastive two-tower on CLIP (0.2, 3 seeds); test-set centering; CSLS k=5 per member (allowed by the organizers) | 24.0 / 57.1 / 4.25 | final |
| `final_nocsls/` | Backup without CSLS: the same members + Stage 3 EEGNet, no centering | 19.1 / 50.0 / 5.5 | backup |
| `decw_clip_b32_f8_t1.5s_w5-60/` | Single whitened CLIP decoder (M8a) | 14.7 / 42.2 / 7.5 | simple fallback |

Weights chosen by training-only CV (leave-one-fold-out strategy rule). Dev labels were used only to report.
Reproduce the headline (branch track-matching, commit 7297a59):
`PYTHONPATH=src:scripts python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 --member decw:clip_b32_mid8_all:8 --max-seconds 1.5 --center-test --csls 5 --csls-stage member --experiment fin5_ctr_m5`

Reproduce v3 (branch track-matching, commit 30edc9a): `PYTHONPATH=src:scripts python scripts/ensemble.py --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 --member decw:alexnet_c5p6:8 --member decw:lowlevel_all:1 --member decw:clap_all:1 --member decw:lowav_all:1 --max-seconds 1.5 --sim-cache results/similarity/_cache --center-test --csls 5 --csls-stage member --experiment v3_ctr_m5`

Reproduce v4 (track-matching 49cc41d): pooled member via `scripts/pool_member.py` (track-pool), then `scripts/ensemble.py` with `--member pdecw:dinov2_b_patch_all` in place of `decw:dinov2_b_patch_all` (see PROGRESS.md "FINAL v4" on track-matching).
