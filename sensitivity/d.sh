#!/usr/bin/env bash
# Sensitivity analysis: attack strength vs problem dimension d.
# For regression/logistic, d is theta's own dimension. For the mlp task, d is the
# victim's INPUT dimension -- theta actually lives in the flattened-parameter space
# (D = h*d + h + h + 1 for hidden width h), so d still controls problem size but the
# effective theta-dimension scales faster than d itself. Activate the project's conda
# env first (conda activate batch-poisoning-rl). Can be run from anywhere -- cd's to
# the repo root itself.
#
#   ./sensitivity/d.sh
#   ./sensitivity/d.sh --seeds 5
#   VALUES="2 5 10 20" ./sensitivity/d.sh
#   TASK=nn SEEDS=3 ./sensitivity/d.sh
#   ./sensitivity/d.sh --seeds 5 --jobs 8   # more concurrent seeds (default: 4)
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
VALUES=${VALUES:-"3 5 8 12"}
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

echo "=== d sensitivity: task=$TASK mode=$MODE seeds=$SEEDS values=[$VALUES] ==="
for v in $VALUES; do
  echo "--- d=$v ---"
  ./run_seeds.sh --task "$TASK" --mode "$MODE" --seeds "$SEEDS" --d "$v" --sens-axis d $par $EXTRA_ARGS
done
echo "=== d sensitivity done ==="
