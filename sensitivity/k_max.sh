#!/usr/bin/env bash
# Sensitivity analysis: attack strength vs the hard edit budget k_max (deletes/step;
# duplicates match 1:1 under balance=True, so total edits per step is 2*k_max).
# c_edit *prices* edits; k_max *caps* them. Forces --c-edit 0 by default so the budget
# is isolated as the only thing constraining edits (nothing is "too expensive", only
# "not allowed") -- override BASE_C_EDIT if you want both active at once. Activate the
# project's conda env first (conda activate batch-poisoning-rl). Can be run from
# anywhere -- cd's to the repo root itself.
#
#   ./sensitivity/k_max.sh
#   ./sensitivity/k_max.sh --seeds 5
#   VALUES="1 3 5" ./sensitivity/k_max.sh
#   BASE_C_EDIT=0.05 ./sensitivity/k_max.sh    # budget + price both active
#   TASK=nn SEEDS=3 ./sensitivity/k_max.sh
#   ./sensitivity/k_max.sh --seeds 5 --jobs 8   # more concurrent seeds (default: 4)
#
# --seeds > 1 trains that value's seeds concurrently (run_seeds.sh --parallel), one
# value at a time; --seeds 1 stays sequential (nothing to parallelize).
#
# See c_edit.sh's header for the aggregate.py/output naming notes -- same here.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -uo pipefail

TASK=${TASK:-linear}
MODE=${MODE:-sort}
SEEDS=${SEEDS:-1}
JOBS=${JOBS:-4}
VALUES=${VALUES:-"1 2 3 4 5"}
BASE_C_EDIT=${BASE_C_EDIT:-0}
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

echo "=== k_max sensitivity: task=$TASK mode=$MODE seeds=$SEEDS values=[$VALUES] c_edit=$BASE_C_EDIT ==="
for v in $VALUES; do
  echo "--- k_max=$v ---"
  ./run_seeds.sh --task "$TASK" --mode "$MODE" --seeds "$SEEDS" --k-max "$v" --c-edit "$BASE_C_EDIT" --sens-axis k_max $par $EXTRA_ARGS
done
echo "=== k_max sensitivity done ==="
