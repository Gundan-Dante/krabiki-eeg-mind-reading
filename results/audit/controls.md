# Controls on the headline `final_csls` (fin5_ctr_m5, track-matching 7297a59), raw evidence

Run from a detached checkout of 7297a59 (`/home/kparpie/Krabiki-audit`), audited code unchanged. Data loading was patched
from outside by `scripts/audit_controls.py`, and results were evaluated by `scripts/audit_compare.py` → `controls.json`.

| Item | Result |
|---|---|
| **3b row permutation** (test recordings of sub-01 and sub-05 permuted, `perms.json`; full scoring rerun incl. centering + per-member CSLS + blend) | **PASS**: output rows == baseline rows[perm], **bit-identical, 102/102 rows** for both subjects |
| **3c determinism, test side** (member test similarities of sub-01 and sub-05 recomputed from scratch, incl. the contrastive member's 3 SGD seeds) | **PASS**: all 5 members bit-identical (max abs diff 0.0) to krabiki-51's cache |
| **3c determinism, CV side** (6 CV-fold files recomputed from scratch before the full run was stopped: all 5 members of sub-01 + decw CLIP of sub-02, alphas + fold similarities) | **PASS**: bit-identical |
| **3d re-score of committed files** | **PASS**: rerun `sub-05` == committed `final_csls/sub-05` and rerun `sub-01` == `fin5_ctr_m5_cv/sub-01`, 102/102 rows identical; dev top-1 of sub-01 29.4% in both |
| 3a label-shuffle control | **not run here** (krabiki-e1 is running it); `audit_controls.py --mode shuffle` implements it |
| 4 fp16 vs fp32 feature extraction | **not run** (timebox); `audit_controls.py --mode fp16` implements it |

Scope notes: the full from-scratch rerun of all six subjects was stopped for time. Determinism was checked on sub-01 /
sub-05 (test) and on sub-01 (+ one sub-02 member) for the CV, all on the same machine and GPU (RTX 2000 Ada, CUDA 12.8,
torch 2.11.0). Cross-hardware determinism is not tested. Within `audit_perm`, the printed dev table (from
`compare_table`, which reloads labels) is not meaningful for the permuted subjects; the check compares rows directly.
