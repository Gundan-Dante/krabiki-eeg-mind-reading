#!/usr/bin/env bash
# Full training pipeline of the final model (v4b): videos + training EEG -> feature caches -> members -> CV-weighted
# ensemble -> sub-0X_ranked_candidate_list.npy, then export of the fitted parameters for inference/.
#
# Usage (from anywhere):  bash training/run_pipeline.sh [stage ...]
#   stages: features | decw | pool | ensemble | export | verify    (default: all, in this order)
# Environment (all optional):
#   DATA      challenge data folder               (default: <repo>/eeg-mind-reading-challenge)
#   FEATURES  feature cache folder                (default: <repo>/features)
#   WORK      similarity caches, logs, outputs    (default: <repo>/work)
#   PY        Python 3.11 interpreter with training/requirements.txt installed (default: python)
#   SUBJECTS  subjects to fit                     (default: "1 2 3 4 5 6"; the weight search needs all six)
#   JOBS      subjects processed in parallel in the decw / pool / ensemble stages (default: 1). Every subject's
#             members depend only on that subject's data (pooling: the others' TRAINING data), so the caches and
#             the outputs are identical for any JOBS; 3 fits an 8 GB GPU.
# Needs one CUDA GPU (8 GB is enough); everything also runs on CPU, much slower.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
DATA=$(realpath -m "${DATA:-$REPO/eeg-mind-reading-challenge}")
FEATURES=$(realpath -m "${FEATURES:-$REPO/features}")
WORK=$(realpath -m "${WORK:-$REPO/work}")
PY=${PY:-python}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6"}
JOBS=${JOBS:-1}
CACHE=$WORK/similarity_cache
EXP=v4b_ctr_m5
export PYTHONPATH="$HERE/src:$HERE/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
mkdir -p "$WORK" "$FEATURES"
IO=(--root "$DATA" --feature-dir "$FEATURES")
FLAGS=(--max-seconds 1.5 --center-test --sim-cache "$CACHE")

# The seven members of the final model, in weight order; global weights 0.2/0.1/0.1/0.2/0.1/0.1/0.2.
MEMBERS=(pdecw:dinov2_b_patch_all decw:vjepa2_l_all:16 con:clip_b32:two_tower_d256 pdecw:alexnet_c5p6:8
         decw:lowlevel_all:1 decw:clap_all:1 pdecw:lowav_all:1)
POOLED=(dinov2_b_patch_all:8 alexnet_c5p6:8 lowav_all:1)   # features of the subject-pooled members

log() { echo "[$(date +%T)] $*"; }

per_subject() {  # per_subject NAME CMD...: run CMD once per subject (appending --subjects S), JOBS at a time
  local name=$1; shift
  if [ "$JOBS" -le 1 ]; then "$@" --subjects $SUBJECTS; return; fi
  local threads=$(( $(nproc) / JOBS )); [ $threads -gt 8 ] && threads=8; [ $threads -lt 1 ] && threads=1
  local pids=() s
  for s in $SUBJECTS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads \
      "$@" --subjects $s > "$WORK/log_${name}_sub-$s.txt" 2>&1 &
    pids+=($!)
  done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "per-subject $name failed, see $WORK/log_${name}_sub-*.txt"; return 1; }
}
member_args() { for m in "$@"; do printf -- "--member\n%s\n" "$m"; done; }

stage_features() {  # frozen public models + hand-crafted descriptors, one cache per feature set (GPU, ~1 h)
  log "features -> $FEATURES"
  $PY "$HERE/scripts/extract_features.py" --model clip_b32 --frames 8 "${IO[@]}"            # contrastive head targets
  $PY "$HERE/scripts/extract_features.py" --model dinov2_b --all-frames --frames 8 "${IO[@]}" # -> dinov2_b_patch_all_f8
  $PY "$HERE/scripts/extract_features.py" --model vjepa2_l --frames 16 "${IO[@]}"            # -> vjepa2_l_all_f16
  $PY "$HERE/scripts/lowlevel_features.py" alexnet-layers --layers c5 --pool 6 "${IO[@]}"     # -> alexnet_c5p6_f8
  $PY "$HERE/scripts/lowlevel_features.py" descriptors "${IO[@]}"                              # -> lowlevel_all_f1
  $PY "$HERE/scripts/extract_audio.py" --model clap "${IO[@]}"                                 # -> clap_all_f1
  $PY "$HERE/scripts/extract_audio.py" --model lowlevel "${IO[@]}"                             # -> lowav_all_f1
}

stage_decw() {  # per-subject decoder alphas (and CV fold matrices) of the features that get pooled
  log "unpooled decoders of the pooled feature sets (alphas for pool_member.py)"
  local specs=(); for f in "${POOLED[@]}"; do specs+=("decw:$f"); done
  (cd "$WORK" && per_subject decw $PY "$HERE/scripts/ensemble.py" $(member_args "${specs[@]}") "${FLAGS[@]}" "${IO[@]}" \
     --cv-only --experiment pool_alphas)
}

stage_pool() {  # subject-pooled decoders: own training trials + the other subjects' training trials at weight lambda
  for spec in "${POOLED[@]}"; do
    log "pooled member ${spec%:*}"
    per_subject "pool_${spec%:*}" $PY "$HERE/scripts/pool_member.py" --features "${spec%:*}" --frames "${spec#*:}" \
      --decw-cache "$CACHE" --cache "$CACHE" "${IO[@]}"
  done
}

stage_ensemble() {  # remaining members, CV weight search, test similarities, CSLS, ranked lists
  log "ensemble $EXP"
  if [ "$JOBS" -gt 1 ]; then  # fill the member caches (CV folds + test) per subject in parallel; the run below reads them
    local unpooled=(); for m in "${MEMBERS[@]}"; do [[ $m == pdecw:* ]] || unpooled+=("$m"); done
    (cd "$WORK" && per_subject members $PY "$HERE/scripts/ensemble.py" $(member_args "${unpooled[@]}") "${FLAGS[@]}" \
       "${IO[@]}" --experiment members_only --results members_only.csv)
  fi
  (cd "$WORK" && $PY "$HERE/scripts/ensemble.py" $(member_args "${MEMBERS[@]}") "${FLAGS[@]}" "${IO[@]}" \
     --csls 5 --csls-stage member --experiment $EXP)
  log "ranked lists: $WORK/submissions/${EXP}_cv/"
}

stage_export() {  # fitted parameters -> inference/model (numpy-only predictor)
  log "export -> $REPO/inference/model"
  $PY "$HERE/scripts/export_bundle.py" --code-dir "$HERE" --out "$REPO/inference/model" --cache "$CACHE" \
    $(member_args "${MEMBERS[@]}") --weights 0.2 0.1 0.1 0.2 0.1 0.1 0.2 \
    --max-seconds 1.5 --center-test --csls 5 --csls-stage member "${IO[@]}"
}

stage_verify() {  # the rebuilt sub-05/06 lists against the committed submission
  log "verify against $REPO/submissions/final_v4b_csls"
  $PY - "$WORK/submissions/${EXP}_cv" "$REPO/submissions/final_v4b_csls" <<'EOF'
import sys
from pathlib import Path
import numpy as np
new, ref = map(Path, sys.argv[1:])
for s in (5, 6):
    a, b = (np.load(d / f"sub-{s:02d}_ranked_candidate_list.npy") for d in (new, ref))
    print(f"sub-{s:02d}: identical={np.array_equal(a, b)}  differing entries={(a != b).sum()}  "
          f"same top-1={np.mean(a[:, 0] == b[:, 0]):.3f}  same top-5 set={np.mean([set(x[:5]) == set(y[:5]) for x, y in zip(a, b)]):.3f}")
EOF
}

STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(features decw pool ensemble export verify)
for st in "${STAGES[@]}"; do "stage_$st"; done
log "done"
