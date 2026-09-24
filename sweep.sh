#!/usr/bin/env bash
# Run run.py across every (task, mode, learner, seed) combination. Sequential by default; set
# PARALLEL=1 to run JOBS at a time instead. Activate the project's conda env first
# (conda activate batch-poisoning-rl).
#
#   ./sweep.sh                                     # default grid, one run at a time
#   PARALLEL=1 JOBS=8 ./sweep.sh
#   TASKS="regression mlp" MODES="greedy" SEEDS="0 1 2 3 4" ./sweep.sh
#   LEARNERS="fevi fvi" ./sweep.sh                 # FEVI vs the FVI ablation
#   EXTRA_ARGS="--K 40" ./sweep.sh                 # forwarded to every run.py call
#   EXTRA_ARGS="--device cuda" PARALLEL=1 JOBS=4 ./sweep.sh   # GPU; each job opens a
#                                                  # CUDA context, so keep JOBS small
#
# Each run's stdout/stderr goes to $OUTDIR/<task>_<mode>_<learner>_seed<seed>.log, and
# its results.json to $RESULTS_DIR (default output/), where aggregate.py folds them.
# One line per run as it finishes (costs, or FAILED with the log tail), then a tally.
#
# PARALLEL=1 pins each run to a single torch thread (TORCH_NUM_THREADS=1); without it
# every one of the JOBS processes grabs all cores and they contend.

# no `set -e`: one run failing shouldn't stop the rest of the sweep or skip the tally.
set -uo pipefail

export TASKS=${TASKS:-"regression logistic mlp"}
export MODES=${MODES:-"greedy exact"}
export LEARNERS=${LEARNERS:-"fevi"}
export SEEDS=${SEEDS:-"0 1 2"}
export PARALLEL=${PARALLEL:-0}
export JOBS=${JOBS:-4}
export OUTDIR=${OUTDIR:-results}
export RESULTS_DIR=${RESULTS_DIR:-output}
export EXTRA_ARGS=${EXTRA_ARGS:-}

ejobs=1
export STREAM=true
if [[ "$PARALLEL" == 1 || "$PARALLEL" == true ]]; then
  ejobs="$JOBS"
  STREAM=false          # interleaved parallel output is unreadable; it goes to the logs
  export TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fi

mkdir -p "$OUTDIR"

echo "=== sweep: tasks=[$TASKS]  modes=[$MODES]  learners=[$LEARNERS]  seeds=[$SEEDS]  jobs=$ejobs"
echo "    run.py args: ${EXTRA_ARGS:-(none)}"
echo "    logs -> $OUTDIR/    results -> $RESULTS_DIR/"
[[ "$EXTRA_ARGS" == *"--no-save"* ]] && \
  echo "    NOTE --no-save: nothing is written; costs appear only in the logs"
echo

export FAILS=$(mktemp)
trap 'rm -f "$FAILS"' EXIT

run_one() {
  local task=$1 mode=$2 learner=$3 seed=$4 start=$SECONDS rc costs
  local tag="$task/$mode/$learner/seed$seed"
  local log="$OUTDIR/${task}_${mode}_${learner}_seed${seed}.log"
  if [[ "$STREAM" == true ]]; then
    python -u src/run.py --task "$task" --mode "$mode" --learner "$learner" \
      --seed "$seed" --outdir "$RESULTS_DIR" $EXTRA_ARGS 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  else
    python -u src/run.py --task "$task" --mode "$mode" --learner "$learner" \
      --seed "$seed" --outdir "$RESULTS_DIR" $EXTRA_ARGS > "$log" 2>&1
    rc=$?
  fi
  if [[ $rc -eq 0 ]]; then
    costs=$(grep -E '^  [A-Za-z-]+ *: -?[0-9.]+$' "$log" |
            awk '{printf "%s%s %.3f", sep, $1, $3; sep = " | "}')
    echo "[$tag] ok in $((SECONDS - start))s   $costs"
  else
    echo "$tag" >> "$FAILS"
    echo "[$tag] FAILED (exit $rc) after $((SECONDS - start))s -- last lines of $log:"
    tail -n 3 "$log" | sed 's/^/      /'
  fi
}
export -f run_one

start_all=$SECONDS
for task in $TASKS; do
  for mode in $MODES; do
    for learner in $LEARNERS; do
      for seed in $SEEDS; do
        echo "$task $mode $learner $seed"
      done
    done
  done
done | xargs -P "$ejobs" -n 4 bash -c 'run_one "$@"' _

elapsed=$((SECONDS - start_all))
ntotal=$(( $(echo $TASKS | wc -w) * $(echo $MODES | wc -w) * $(echo $LEARNERS | wc -w) \
           * $(echo $SEEDS | wc -w) ))
nfail=$(wc -l < "$FAILS")
echo
echo "=== $((ntotal - nfail))/$ntotal runs ok in $((elapsed / 60))m$((elapsed % 60))s   (logs in $OUTDIR/)"
(( nfail > 0 )) && echo "    failed: $(paste -sd' ' "$FAILS")"

# combine each (task, mode)'s seeds into one results.json (skipped if --no-save is in
# EXTRA_ARGS, since there'd be nothing to read)
if [[ "$EXTRA_ARGS" != *"--no-save"* ]] && (( nfail < ntotal )); then
  for task in $TASKS; do
    for mode in $MODES; do
      echo
      python src/aggregate.py --task "$task" --mode "$mode" --outdir "$RESULTS_DIR"
    done
  done
fi
