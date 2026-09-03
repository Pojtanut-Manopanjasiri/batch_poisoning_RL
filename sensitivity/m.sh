#!/usr/bin/env bash
# Sensitivity analysis: attack strength vs batch size m (action space = 3^m).
# The joint action space grows exponentially in m -- the reason BDQ/TRPO exist at all --
# while each point's own influence shrinks as phi_i ~ 1/m (the victim averages the
# batch). SCALE_CEDIT (default on) holds c_edit*m constant across values so the total
# edit budget's *price* doesn't confound with the batch-size effect; set it to 0 to
# sweep m at a fixed c_edit instead. Activate the project's conda env first (conda
# activate batch-poisoning-rl). Can be run from anywhere -- cd's to the repo root.
#
#   ./sensitivity/m.sh
#   ./sensitivity/m.sh --seeds 5
#   VALUES="4 8 12 16" ./sensitivity/m.sh
#   SCALE_CEDIT=0 ./sensitivity/m.sh
#   TASK=logistic SEEDS=3 ./sensitivity/m.sh
#   ./sensitivity/m.sh --seeds 5 --jobs 8   # more concurrent seeds (default: 4)
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
VALUES=${VALUES:-"4 6 8 10 14"}
BASE_C_EDIT=${BASE_C_EDIT:-0.1}   # regression task's default c_edit; override if TASK != linear
BASE_M=${BASE_M:-10}              # the m that BASE_C_EDIT was calibrated at
SCALE_CEDIT=${SCALE_CEDIT:-1}     # 1: hold c_edit*m constant.  0: fixed c_edit at every m
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

echo "=== m sensitivity: task=$TASK mode=$MODE seeds=$SEEDS values=[$VALUES] scale_cedit=$SCALE_CEDIT ==="
for v in $VALUES; do
  if [[ "$SCALE_CEDIT" == 1 ]]; then
    c_edit=$(python3 -c "print($BASE_C_EDIT * $BASE_M / $v)")
    echo "--- m=$v  (c_edit=$c_edit, holding c_edit*m=${BASE_C_EDIT}*${BASE_M}) ---"
  else
    c_edit=$BASE_C_EDIT
    echo "--- m=$v  (c_edit=$c_edit, fixed) ---"
  fi
  ./run_seeds.sh --task "$TASK" --mode "$MODE" --seeds "$SEEDS" --m "$v" --c-edit "$c_edit" --sens-axis m $par $EXTRA_ARGS
done
echo "=== m sensitivity done ==="
