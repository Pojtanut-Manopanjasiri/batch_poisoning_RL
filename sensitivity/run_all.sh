#!/usr/bin/env bash
# Run every sensitivity axis in this folder, one after another. This is the slow,
# comprehensive path -- each axis is its own script (c_edit.sh, eta.sh, m.sh, T.sh,
# d.sh) and can be run standalone instead; see their headers for
# axis-specific options (VALUES, SCALE_CEDIT, BASE_C_EDIT, ...). Activate the project's
# conda env first (conda activate batch-poisoning-rl).
#
#   ./sensitivity/run_all.sh                        # every axis, defaults
#   ./sensitivity/run_all.sh --seeds 5               # forwarded to every axis script
#   ./sensitivity/run_all.sh --seeds 5 --jobs 8      # each axis's seeds train concurrently
#   TASK=logistic SEEDS=1 ./sensitivity/run_all.sh   # env vars work too
#   AXES="c_edit m T" ./sensitivity/run_all.sh       # only run these axes
#   EXTRA_ARGS="--device cuda" ./sensitivity/run_all.sh
#
# --seeds > 1 makes every axis train that value's seeds concurrently (each script's own
# run_seeds.sh --parallel, default 4 jobs, override with --jobs); --seeds 1 stays
# sequential. See any axis script's header for details.

cd "$(dirname "${BASH_SOURCE[0]}")"
set -uo pipefail   # no -e: one axis failing shouldn't cancel the rest

AXES=${AXES:-"c_edit eta m T d"}
seeds=""   # plain strings, not arrays: bash 3.2's `set -u` treats a never-populated
jobs=""    # array as unbound, so this project sticks to strings for optional argv

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) seeds="--seeds $2"; shift 2 ;;
    --jobs)  jobs="--jobs $2"; shift 2 ;;
    *) echo "unknown argument: $1 (other options are env vars, see script header)" >&2; exit 1 ;;
  esac
done

echo "=== sensitivity sweep: axes=[$AXES]  task=${TASK:-linear}  seeds=${SEEDS:-<per-axis default>} ==="
for axis in $AXES; do
  script="./${axis}.sh"
  if [[ ! -f "$script" ]]; then
    echo "!! unknown axis '$axis' (no $script) -- skipping" >&2
    continue
  fi
  echo
  echo "############################################################"
  echo "# axis: $axis"
  echo "############################################################"
  bash "$script" $seeds $jobs
done
echo
echo "=== all requested axes finished ==="
