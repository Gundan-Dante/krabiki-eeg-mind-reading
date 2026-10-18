#!/usr/bin/env bash
# Audit + bundle runbook for a CV-weighted ensemble produced by scripts/ensemble.py in another worktree.
# Fill in the variables (from the producing session's report), then run:  bash scripts/audit_runbook.sh <step>
# Steps: setup | rescore | rerun | permute | compare | shuffle | bundle | infer | package   (or "all")
set -euo pipefail

: "${COMMIT:?commit that produced the submission}"            # e.g. 7297a59
: "${SRC_TREE:?worktree holding its results/ and submissions/}" # e.g. /home/kparpie/Krabiki-matching
: "${EXP:?ensemble --experiment name}"                         # e.g. fin5_ctr_m5
: "${FINAL_DIR:?submission dir with the final sub-05/06}"      # e.g. final_csls
: "${MEMBERS:?member specs, space separated}"                  # e.g. "decw:clip_b32 decw:dinov2_b_patch_all ..."
: "${WEIGHTS:?final weights, integers summing to 10}"          # e.g. "0 4 3 2 1"
FLAGS=${FLAGS:-"--max-seconds 1.5 --center-test --csls 5 --csls-stage member"}
REPO=/home/kparpie/Krabiki
AUD=${AUD:-/home/kparpie/Krabiki-audit-$COMMIT}
PY=$REPO/.venv/bin/python
OUT=${OUT:-$REPO/results/audit_$EXP}
BUNDLE=${BUNDLE:-$REPO/bundle/krabiki_${FINAL_DIR}_bundle}
DATA="--root $REPO/eeg-mind-reading-challenge --feature-dir $REPO/features"
MEM_ARGS=$(for m in $MEMBERS; do printf -- "--member %s " "$m"; done)
CACHE=$SRC_TREE/results/similarity/_cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$OUT"

step_setup() {
  [ -d "$AUD" ] || git -C $REPO worktree add --detach "$AUD" "$COMMIT"
  mkdir -p "$AUD/audit_out"
}
step_rescore() {  # saved blended matrices -> ranked lists == written submissions; dev scores
  PYTHONPATH=$REPO/src $PY $REPO/scripts/audit_compare.py --out "$OUT/rescore.json" rescore \
    --sim-dir "$SRC_TREE/results/similarity/${EXP}_cv" --sub-dir "$SRC_TREE/submissions/${EXP}_cv"
}
prep_cache() {  # copy the producer's cache; drop the test similarities of sub-01 / sub-05 so they are recomputed
  rm -rf "$AUD/audit_out/$1" && mkdir -p "$AUD/audit_out/$1" && cp "$CACHE"/*.npz "$AUD/audit_out/$1/"
  rm -f "$AUD/audit_out/$1"/sub-01_*_test.npz "$AUD/audit_out/$1"/sub-05_*_test.npz
}
step_rerun() {
  prep_cache cache_rerun
  (cd "$AUD" && $PY $REPO/scripts/audit_controls.py --code-dir . --mode rerun -- $MEM_ARGS $FLAGS $DATA \
     --sim-cache audit_out/cache_rerun --experiment audit_rerun --results audit_out/r.csv --cv-out audit_out/r_cv.csv) \
     > "$OUT/rerun.log" 2>&1
}
step_permute() {
  prep_cache cache_perm
  (cd "$AUD" && $PY $REPO/scripts/audit_controls.py --code-dir . --mode rerun --permute --subjects-recompute 1 5 \
     --perm-out audit_out/perms.json -- $MEM_ARGS $FLAGS $DATA --sim-cache audit_out/cache_perm \
     --experiment audit_perm --results audit_out/p.csv --cv-out audit_out/p_cv.csv) > "$OUT/perm.log" 2>&1
  cp "$AUD/audit_out/perms.json" "$OUT/"
}
step_compare() {
  PYTHONPATH=$REPO/src $PY $REPO/scripts/audit_compare.py --out "$OUT/determinism.json" determinism \
    --new-cache "$AUD/audit_out/cache_rerun" --ref-cache "$CACHE" --subjects 1 5 \
    --new-sub "$AUD/submissions/audit_rerun_cv" --ref-sub "$SRC_TREE/submissions/$FINAL_DIR"
  PYTHONPATH=$REPO/src $PY $REPO/scripts/audit_compare.py --out "$OUT/permutation.json" permutation \
    --perms "$AUD/audit_out/perms.json" --base-sub "$AUD/submissions/audit_rerun_cv" --perm-sub "$AUD/submissions/audit_perm_cv"
}
step_shuffle() {  # training pairs permuted, alphas re-chosen, the FINAL weights fixed (variant "ref"); dev subjects only
  # one process per dev subject in parallel (4 threads each); the fixed --ref-weights make per-subject runs equivalent
  for s in 1 2 3 4; do
    (cd "$AUD" && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       $PY $REPO/scripts/audit_controls.py --code-dir . --mode shuffle -- $MEM_ARGS $FLAGS $DATA \
       --ref-weights $WEIGHTS --subjects $s --experiment audit_shuffle_s$s --results audit_out/s$s.csv \
       --cv-out audit_out/s${s}_cv.csv) > "$OUT/shuffle_s$s.log" 2>&1 &
  done
  wait
  for s in 1 2 3 4; do
    PYTHONPATH=$REPO/src $PY $REPO/scripts/audit_compare.py --out "$OUT/shuffle_s$s.json" score \
      --sub-dir "$AUD/submissions/audit_shuffle_s${s}_ref" --subjects $s
  done
}
step_bundle() {  # SUBJECT_WEIGHTS=<json {"1": [...], ...}> overrides WEIGHTS for per-subject strategies
  W_ARGS=${SUBJECT_WEIGHTS:+--subject-weights $SUBJECT_WEIGHTS}
  $PY $REPO/scripts/export_bundle.py --code-dir "$AUD" --cache "$CACHE" --out "$BUNDLE" $MEM_ARGS \
    ${W_ARGS:---weights $WEIGHTS} $FLAGS --source "$COMMIT, $EXP ($FINAL_DIR)" $DATA > "$OUT/export.log" 2>&1
  cp $REPO/scripts/infer_bundle.py "$BUNDLE/"
}
step_package() {  # docs + schematics + README, then zip (run after step_infer has verified the bundle)
  VIZ=${VIZ:-/home/kparpie/Krabiki-viz/presentation}
  for f in ${DOCS:-ARCHITECTURE.md IMPROVEMENTS.md EEGNET.md AUDIT.md PRESENTATION.md}; do [ -f $REPO/$f ] && cp $REPO/$f "$BUNDLE/"; done
  mkdir -p "$BUNDLE/schematics" && cp $VIZ/FIGURES.md "$BUNDLE/schematics/"
  for f in ${FIGS:-fig1_progress fig2_encoding_vs_decoding fig4_ensemble_anatomy fig5_expected_score fig7_recall_at_k}; do
    cp $VIZ/figures/$f.png "$BUNDLE/schematics/"; done
  $PY $REPO/scripts/bundle_readme.py --bundle "$BUNDLE" --title "$FINAL_DIR" --dev "${DEV_LINE:-see bundle_meta.json}" \
    --expected-dir "submissions/$FINAL_DIR" --verified "${VERIFIED_LINE:-}"
  (cd "$(dirname "$BUNDLE")" && rm -f "$(basename "$BUNDLE").zip" && zip -qr -9 "$(basename "$BUNDLE").zip" "$(basename "$BUNDLE")")
  ls -la "$BUNDLE.zip"
}
step_infer() {  # CPU only; must equal the committed final sub-05/06
  CUDA_VISIBLE_DEVICES="" $PY -I "$BUNDLE/infer_bundle.py" --bundle "$BUNDLE" --root $REPO/eeg-mind-reading-challenge \
    --subjects 1 2 3 4 5 6 --out "$OUT/bundle_out" | tee "$OUT/infer.log"
  PYTHONPATH=$REPO/src $PY $REPO/scripts/audit_compare.py --out "$OUT/bundle_vs_final.json" determinism \
    --new-cache /nonexistent --ref-cache /nonexistent --subjects 5 6 --new-sub "$OUT/bundle_out" \
    --ref-sub "$SRC_TREE/submissions/$FINAL_DIR"
}

case "${1:-all}" in
  all) step_setup; step_rescore; step_rerun; step_permute; step_compare; step_shuffle; step_bundle; step_infer; step_package ;;
  *) "step_$1" ;;
esac
