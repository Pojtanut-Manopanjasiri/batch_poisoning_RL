#!/usr/bin/env bash
# Sensitivity analysis: attack strength vs edit price c_edit.
# Retrains the tier (+ BDQ/TRPO if requested via EXTRA_ARGS) at every value, same
# fairness protocol as run_seeds.sh/sweep.sh. Activate the project's conda env first
# (conda activate batch-poisoning-rl). Can be run from anywhere -- cd's to the repo
# root itself.
#
#   ./sensitivity/c_edit.sh                                # default axis/task/seeds
#   ./sensitivity/c_edit.sh --seeds 5
#   VALUES="0 0.05 0.1 0.2" ./sensitivity/c_edit.sh
#   TASK=logistic SEEDS=5 ./sensitivity/c_edit.sh           # SEEDS env var works too
#   EXTRA_ARGS="--bdq --trpo" ./sensitivity/c_edit.sh       # add baselines at every point
#   EXTRA_ARGS="--device mps" ./sensitivity/c_edit.sh
#   ./sensitivity/c_edit.sh --seeds 5 --jobs 8               # more concurrent seeds (default: 4)
#
# --seeds > 1 trains that value's seeds concurrently (run_seeds.sh --parallel), one
# value at a time; --seeds 1 stays sequential (nothing to parallelize).
#
# Each value gets its own ./run_seeds.sh call (seeds trained, then folded into one
# output/<task>_<mode>_<tag>_results_<timestamp>.json and cleaned up by aggregate.py,
# tag = a c_edit/eta/m/T/d/rank/k_max summary) BEFORE the next value starts. Every call
# passes --sens-axis c_edit, recorded verbatim in each seed's results.json and carried
# through by aggregate.py -- the notebook's sensitivity loader reads this tag directly
# rather than inferring axis membership from config content, so it can't be confused by
# some other axis's sweep files sitting in the same output/ folder at an overlapping
# config (e.g. this axis's own c_edit=0.1 baseline point vs another axis's baseline
# point, which share every other field). aggregate.py also groups strictly by each
# seed's own recorded config (+ this tag), so even running several sensitivity/*.sh
# scripts at once (interleaved in output/) won't mix one sweep's seeds into another's
# summary -- running one at a time here just keeps output/ tidier (fewer pending
# per-seed directories at any moment).

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -uo pipefail

TASK=${TASK:-linear}
MODE=${MODE:-sort}
SEEDS=${SEEDS:-3}
JOBS=${JOBS:-4}
VALUES=${VALUES:-"0.0 0.025 0.05 0.1 0.2 0.4"}
EXTRA_ARGS=${EXTRA_ARGS:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --jobs)  JOBS="$2"; shift 2 ;;
    *) echo "unknown argument: $1 (other options are env vars, see script header)" >&2; exit 1 ;;
  esac
done

par=""
[[ "$SEEDS" -gt 1 ]] && par="--parallel --jobs $JOBS"

echo "=== c_edit sensitivity: task=$TASK mode=$MODE seeds=$SEEDS values=[$VALUES] ==="
for v in $VALUES; do
  echo "--- c_edit=$v ---"
  ./run_seeds.sh --task "$TASK" --mode "$MODE" --seeds "$SEEDS" --c-edit "$v" --sens-axis c_edit $par $EXTRA_ARGS
done
echo "=== c_edit sensitivity done ==="
