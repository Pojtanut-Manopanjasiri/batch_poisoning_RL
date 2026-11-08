#!/bin/bash
# Tuning sweep: every GRID_* combination x TUNE_SEEDS per learner, scored on seed 997, then pick_best.py.
# Usage: bash scripts/hyak/sweep.sh [config file] [--dry-run | --local] [KEY=VALUE ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$@"
HP_FROM=                                  # a sweep starts from config.sh's values
RUN_DIR=$OUT_ROOT/${RUN_NAME}_tune

grid() {  # arg: learner -> one line of key=value words per config (c0, c1, ...)
  local a b c
  case $1 in
    fevi) for a in $GRID_FEVI_LR; do for b in $GRID_FEVI_FIT_STEPS; do for c in $GRID_FEVI_M; do
            echo "lr=$a fevi_fit_steps=$b fevi_M=$c"; done; done; done ;;
    trpo) for a in $GRID_TRPO_DELTA; do for b in $GRID_TRPO_VF_ITERS; do
            echo "trpo_delta=$a trpo_vf_iters=$b"; done; done ;;
    bdq) for a in $GRID_BDQ_GRAD_STEPS; do for b in $GRID_BDQ_LR; do
           echo "n_grad_steps=$a lr=$b"; done; done ;;
  esac
}

lines() {
  local l s i hp
  job_line refs "${TUNE_SEEDS%% *}" "$RUN_DIR/out/refs.pkl" "--tune "
  for l in $LEARNERS; do
    i=0
    while read -r hp; do
      for s in $TUNE_SEEDS; do
        job_line "$l" "$s" "$RUN_DIR/out/${l}_c${i}_s$s.pkl" \
          "--tune --tag c$i $(learner_hp "$l")$(printf -- '--hp %s ' $hp)"
      done
      i=$((i + 1))
    done < <(grid "$l")
  done
}

prepare_run lines
submit_jobs "$JOBS" "$PY $REPO/scripts/pick_best.py $RUN_DIR"
