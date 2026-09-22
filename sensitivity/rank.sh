#!/usr/bin/env bash
# Sensitivity analysis: attack strength (and training time) vs low-rank's kept rank r.
# Only meaningful for the lowrank tier (--rank is ignored by sort/exact), so this script
# always forces --mode lowrank regardless of $MODE. r is bounded by min(m, d) -- pick
# VALUES accordingly for your task's m/d. Note: unlike the original notebook's rank
# sweep (which trains ONE V_hat and only varies the selector at eval time), this
# retrains a fresh V_hat at every r, since run.py's --mode/--rank affect the tier used
# to collect rollouts too -- slower, but consistent with every other axis here (every
# learner retrained at every point). Activate the project's conda env first (conda
# activate batch-poisoning-rl). Can be run from anywhere -- cd's to the repo root.
#
# EXCLUDED from run_all.sh's default AXES: for the regression task's default d=5, m=10,
# the interaction matrix B is provably rank <= d = 5 (decomposition.py builds it as
# Phi @ (.) @ Phi.T with Phi an m x d matrix), and its spectrum is dominated ~3-4x by the
# top eigendirection -- so r=1..5 pick nearly-identical actions (measured >96% agreement
# even at r=1) and the sweep is flat well within seed noise. Only worth running by hand
# with a larger --d (raises the rank ceiling) or against the mlp task (richer Hessian
# structure) -- see AXES="... rank ..." in run_all.sh's header to add it back in.
#
#   ./sensitivity/rank.sh
#   ./sensitivity/rank.sh --seeds 5
#   VALUES="1 2 3" ./sensitivity/rank.sh
#   TASK=logistic SEEDS=3 ./sensitivity/rank.sh
#   ./sensitivity/rank.sh --seeds 5 --jobs 8   # more concurrent seeds (default: 4)
#
# --seeds > 1 trains that value's seeds concurrently (run_seeds.sh --parallel), one
# value at a time; --seeds 1 stays sequential (nothing to parallelize).
#
# See c_edit.sh's header for the aggregate.py/output naming notes -- same here.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -uo pipefail

TASK=${TASK:-linear}
SEEDS=${SEEDS:-1}
JOBS=${JOBS:-4}
VALUES=${VALUES:-"1 2 3 4 5"}
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

echo "=== rank sensitivity: task=$TASK mode=lowrank seeds=$SEEDS values=[$VALUES] ==="
for v in $VALUES; do
  echo "--- rank=$v ---"
  ./run_seeds.sh --task "$TASK" --mode lowrank --seeds "$SEEDS" --rank "$v" --sens-axis rank $par $EXTRA_ARGS
done
echo "=== rank sensitivity done ==="
