#!/bin/bash
# Edit-cost sweep: one submit.sh run per value in LAMS, then lam_summary.py compares the policies across them.
# Usage: bash scripts/hyak/lam_sweep.sh [config file] [--dry-run | --local] [KEY=VALUE ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$@"
SUM_DIR=$OUT_ROOT/${RUN_NAME}_lam
dirs=() ids=()
for lam in $LAMS; do
  run=${RUN_NAME}_lam$lam
  bash "$HERE/submit.sh" FEVI_EXACT_TEST=true "$@" LAM="$lam" RUN_NAME="$run"      # later settings win
  dirs+=("$OUT_ROOT/$run")
  [[ $MODE == submit ]] && ids+=("$(cat "$OUT_ROOT/$run/then.jobid")")
done
summary="$PY $REPO/scripts/lam_summary.py ${dirs[*]} --out $SUM_DIR"
case $MODE in
  dry) echo; echo "then, after every run is merged: $summary" ;;
  local) eval "$summary" ;;
  submit)
    mkdir -p "$SUM_DIR/logs"
    echo "$summary" > "$SUM_DIR/then.txt"
    sbatch_args "$SUM_DIR"
    "${sb[@]}" --job-name="${RUN_NAME}_lam" --dependency=afterok:"$(IFS=:; echo "${ids[*]}")" --array=0 \
      "$HERE/array.slurm" "$SUM_DIR/then.txt" > /dev/null
    echo "summary job submitted: table and figure in $SUM_DIR once every run has finished" ;;
esac
