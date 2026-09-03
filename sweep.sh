#!/usr/bin/env bash
# Run run.py across every (task, mode, seed) combination. Sequential by default; set
# PARALLEL=1 to run JOBS at a time instead. Activate the project's conda env first
# (conda activate batch-poisoning-rl).
#
#   ./sweep.sh                                     # default grid, one run at a time
#   PARALLEL=1 JOBS=8 ./sweep.sh
#   TASKS="regression mlp" MODES="sort" SEEDS="0 1 2 3 4" ./sweep.sh
#   EXTRA_ARGS="--K 40" ./sweep.sh                 # forwarded to every run.py call
#   EXTRA_ARGS="--device mps" ./sweep.sh           # GPU for BDQ/TRPO + FVI's TD-fit step
#
# Each run's stdout/stderr goes to $OUTDIR/<task>_<mode>_seed<seed>.log.
#
# Note: with PARALLEL=1 and --device mps, concurrent jobs share one GPU context (MPS
# has no real multi-process isolation, unlike CUDA) -- if that's slower than sequential,
# drop PARALLEL for GPU runs.

# no `set -e`: one run failing shouldn't stop the rest of the sweep or skip the
# "all runs finished" line below.
set -uo pipefail

export TASKS=${TASKS:-"regression logistic mlp"}
export MODES=${MODES:-"sort lowrank"}
export SEEDS=${SEEDS:-"0 1 2"}
export PARALLEL=${PARALLEL:-0}
export JOBS=${JOBS:-4}
export OUTDIR=${OUTDIR:-results}
export EXTRA_ARGS=${EXTRA_ARGS:-}

ejobs=1
if [[ "$PARALLEL" == 1 || "$PARALLEL" == true ]]; then ejobs="$JOBS"; fi

mkdir -p "$OUTDIR"

for task in $TASKS; do
  for mode in $MODES; do
    for seed in $SEEDS; do
      echo "$task $mode $seed"
    done
  done
done | xargs -P "$ejobs" -n 3 bash -c '
  task=$1; mode=$2; seed=$3
  log="$OUTDIR/${task}_${mode}_seed${seed}.log"
  echo "[$task/$mode/seed$seed] -> $log"
  python -u src/run.py --task "$task" --mode "$mode" --seed "$seed" $EXTRA_ARGS 2>&1 | tee "$log"
  echo "[$task/$mode/seed$seed] done (exit ${PIPESTATUS[0]})"
' _

echo "all runs finished; logs in $OUTDIR/"

# combine each (task, mode)'s seeds into one results.json (skipped if --no-save is in
# EXTRA_ARGS, since there'd be nothing to read)
if [[ "$EXTRA_ARGS" != *"--no-save"* ]]; then
  for task in $TASKS; do
    for mode in $MODES; do
      python src/aggregate.py --task "$task" --mode "$mode"
    done
  done
fi
