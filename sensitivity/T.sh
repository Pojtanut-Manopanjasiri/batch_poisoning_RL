#!/usr/bin/env bash
# Sensitivity analysis: attack strength vs horizon T (planning length).
# At T=1 the method's decomposition must reduce exactly to greedy (V_hat(.,T)=0, so
# alpha collapses to the immediate term) -- a useful sanity check on this axis. Larger T
# is where the learned cost-to-go actually earns its keep over a myopic policy. Activate
# the project's conda env first (conda activate batch-poisoning-rl). Can be run from
# anywhere -- cd's to the repo root itself.
#
#   ./sensitivity/T.sh
#   ./sensitivity/T.sh --seeds 5
#   VALUES="1 5 10 20" ./sensitivity/T.sh
#   TASK=nn SEEDS=3 ./sensitivity/T.sh
#   ./sensitivity/T.sh --seeds 5 --jobs 8   # more concurrent seeds (default: 4)
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
VALUES=${VALUES:-"1 2 4 6 10 14"}
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

echo "=== T sensitivity: task=$TASK mode=$MODE seeds=$SEEDS values=[$VALUES] ==="
for v in $VALUES; do
  echo "--- T=$v ---"
  ./run_seeds.sh --task "$TASK" --mode "$MODE" --seeds "$SEEDS" --T "$v" --sens-axis T $par $EXTRA_ARGS
done
echo "=== T sensitivity done ==="
