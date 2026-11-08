#!/bin/bash
# Final runs: one Slurm task per (seed, learner) plus the references per seed, then merge_runs.py.
# Usage: bash scripts/hyak/submit.sh [config file] [--dry-run | --local] [KEY=VALUE ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$@"
RUN_DIR=$OUT_ROOT/$RUN_NAME

lines() {
  local s l trials
  for s in $(expand_seeds "$SEEDS"); do
    job_line refs "$s" "$RUN_DIR/out/refs_s$s.pkl"
    for l in $LEARNERS; do
      trials=""
      [[ -n $HP_FROM ]] && trials="--trials $(tuned "$l" trials) "
      job_line "$l" "$s" "$RUN_DIR/out/${l}_s$s.pkl" "$(learner_hp "$l")$trials"
    done
  done
}

prepare_run lines
follow="$PY $REPO/scripts/merge_runs.py $RUN_DIR --suffix=$RESULT_SUFFIX"
[[ $EXPORT_EPISODES != 0 ]] && follow+=" && $PY $REPO/scripts/export_predictions.py $RUN_DIR --episodes $EXPORT_EPISODES"
submit_jobs "$JOBS" "$follow"
