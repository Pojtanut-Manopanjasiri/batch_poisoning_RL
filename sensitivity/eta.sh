#!/usr/bin/env bash
# Sensitivity analysis: attack strength vs the victim's own learning rate eta.
# eta sets how far one SGD step moves the victim (phi_i = -(eta/m)*grad_i), independent
# of anything the attacker controls -- alpha_i and B both scale with it. Activate the
# project's conda env first (conda activate batch-poisoning-rl). Can be run from
# anywhere -- cd's to the repo root itself.
#
#   ./sensitivity/eta.sh
#   ./sensitivity/eta.sh --seeds 5
#   VALUES="0.1 0.2 0.4" ./sensitivity/eta.sh
#   TASK=nn SEEDS=3 ./sensitivity/eta.sh
#   EXTRA_ARGS="--bdq --trpo" ./sensitivity/eta.sh
#   ./sensitivity/eta.sh --seeds 5 --jobs 8       # more concurrent seeds (default: 4)
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
VALUES=${VALUES:-"0.05 0.1 0.2 0.4 0.8"}
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

echo "=== eta sensitivity: task=$TASK mode=$MODE seeds=$SEEDS values=[$VALUES] ==="
for v in $VALUES; do
  echo "--- eta=$v ---"
  ./run_seeds.sh --task "$TASK" --mode "$MODE" --seeds "$SEEDS" --eta "$v" --sens-axis eta $par $EXTRA_ARGS
done
echo "=== eta sensitivity done ==="
