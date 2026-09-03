#!/usr/bin/env bash
# Run run.py for many seeds of ONE experiment. Sequential by default; add --parallel
# to train multiple seeds at once. Activate the project's conda env first
# (conda activate batch-poisoning-rl).
#
#   ./run_seeds.sh --task linear --seeds 5                          # one at a time
#   ./run_seeds.sh --task nn --seeds 3 --parallel --jobs 2 --mode lowrank --K 40
#   ./run_seeds.sh --task logistic --seeds 4 --m 6 --T 8 --c-edit 0.05
#   ./run_seeds.sh --task linear --seeds 2 -- --diagnostics --no-save   # anything else
#
# --task      linear | logistic | nn    (maps to run.py's regression | logistic | mlp)
# --seeds     how many seeds to run, i.e. seeds 0..N-1              [default: 1]
# --parallel  train seeds concurrently instead of one at a time     [default: off]
# --jobs      how many seeds to train at once (only with --parallel) [default: 4]
# --outdir    where per-seed LOG files go (this script's own output) [default: results]
#             run.py's own results.json is separate: it auto-names an output/... dir per
#             run (task/mode/seed/timestamp) -- not settable here, see run.py --no-save
#
# Every run.py override is available directly here too, forwarded only if you pass it
# (otherwise run.py falls back to its own task default), so you don't need `--`:
#   --mode --rank --d --m --T --eta --noise --c-edit --weight-decay --K --dual-iters
#   --victim-hidden --init-scale --k-max --no-hessian --diagnostics --no-save --verbose
#   --bdq --trpo --trpo-iters --device --sens-axis
#
# --sens-axis NAME: tag every seed's results.json with which sensitivity/*.sh axis
# produced it (e.g. "c_edit", "m") -- sensitivity/*.sh scripts set this themselves, no
# need to pass it by hand unless you're sweeping something outside those scripts.
#
# --device auto|cpu|mps|cuda: speeds up BDQ/TRPO and FVI's TD-fit step (default "auto"
# picks a GPU if one is available). With --parallel, each seed's process gets its own
# GPU context -- fine on CUDA, but MPS has no real multi-process isolation, so several
# concurrent --device mps jobs will contend for the same GPU; if that looks slower than
# sequential, drop --parallel for GPU runs.
#
# Anything after a lone `--` is ALSO forwarded verbatim, for flags not listed above
# (e.g. ones added to run.py later).

set -uo pipefail

task=""
nseeds=1
parallel=false
jobs=4
outdir="results"
extra=""

# run.py overrides -- unset ("") means "don't forward; let run.py use its own default"
mode="" rank="" d="" m="" T="" eta="" noise="" c_edit="" weight_decay=""
K="" dual_iters="" victim_hidden="" init_scale="" trpo_iters="" device="" k_max=""
sens_axis=""
no_hessian=false diagnostics=false no_save=false verbose=false bdq=false trpo=false

usage() {
  echo "Usage: $0 --task {regression,logistic,nn} [--seeds N] [--parallel] [--jobs N] [--outdir DIR]" >&2
  echo "          [--mode M] [--rank R] [--d D] [--m M] [--T T] [--eta E] [--noise N] [--c-edit C]" >&2
  echo "          [--weight-decay W] [--K K] [--dual-iters D] [--victim-hidden H] [--init-scale S]" >&2
  echo "          [--k-max K] [--no-hessian] [--diagnostics] [--no-save] [--verbose]" >&2
  echo "          [--bdq] [--trpo] [--trpo-iters N] [--device {auto,cpu,mps,cuda}]" >&2
  echo "          [--sens-axis NAME] [-- extra run.py args]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)          task="$2"; shift 2 ;;
    --seeds)         nseeds="$2"; shift 2 ;;
    --parallel)      parallel=true; shift ;;
    --jobs)          jobs="$2"; shift 2 ;;
    --outdir)        outdir="$2"; shift 2 ;;
    --mode)          mode="$2"; shift 2 ;;
    --rank)          rank="$2"; shift 2 ;;
    --d)             d="$2"; shift 2 ;;
    --m)             m="$2"; shift 2 ;;
    --T)             T="$2"; shift 2 ;;
    --eta)           eta="$2"; shift 2 ;;
    --noise)         noise="$2"; shift 2 ;;
    --c-edit)        c_edit="$2"; shift 2 ;;
    --weight-decay)  weight_decay="$2"; shift 2 ;;
    --K)             K="$2"; shift 2 ;;
    --dual-iters)    dual_iters="$2"; shift 2 ;;
    --victim-hidden) victim_hidden="$2"; shift 2 ;;
    --init-scale)    init_scale="$2"; shift 2 ;;
    --no-hessian)    no_hessian=true; shift ;;
    --diagnostics)   diagnostics=true; shift ;;
    --no-save)       no_save=true; shift ;;
    --verbose)       verbose=true; shift ;;
    --bdq)           bdq=true; shift ;;
    --trpo)          trpo=true; shift ;;
    --trpo-iters)    trpo_iters="$2"; shift 2 ;;
    --device)        device="$2"; shift 2 ;;
    --k-max)         k_max="$2"; shift 2 ;;
    --sens-axis)     sens_axis="$2"; shift 2 ;;
    --)              shift; extra="$*"; break ;;
    -h|--help)       usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

ejobs=1
if [[ "$parallel" == true ]]; then ejobs="$jobs"; fi

case "$task" in
  linear)                    task=regression ;;
  nn)                        task=mlp ;;
  logistic|regression|mlp)   ;;                  # already a valid run.py --task value
  "") echo "must pass --task {linear,logistic,nn}" >&2; usage ;;
  *)  echo "unknown task: $task (expected linear|logistic|nn)" >&2; usage ;;
esac

mkdir -p "$outdir"

# forward only the run.py overrides actually set, in the same "None means default" spirit
run_py_args=""
[[ -n "$mode"          ]] && run_py_args+=" --mode $mode"
[[ -n "$rank"          ]] && run_py_args+=" --rank $rank"
[[ -n "$d"             ]] && run_py_args+=" --d $d"
[[ -n "$m"             ]] && run_py_args+=" --m $m"
[[ -n "$T"             ]] && run_py_args+=" --T $T"
[[ -n "$eta"           ]] && run_py_args+=" --eta $eta"
[[ -n "$noise"         ]] && run_py_args+=" --noise $noise"
[[ -n "$c_edit"        ]] && run_py_args+=" --c-edit $c_edit"
[[ -n "$weight_decay"  ]] && run_py_args+=" --weight-decay $weight_decay"
[[ -n "$K"             ]] && run_py_args+=" --K $K"
[[ -n "$dual_iters"    ]] && run_py_args+=" --dual-iters $dual_iters"
[[ -n "$victim_hidden" ]] && run_py_args+=" --victim-hidden $victim_hidden"
[[ -n "$init_scale"    ]] && run_py_args+=" --init-scale $init_scale"
[[ -n "$trpo_iters"    ]] && run_py_args+=" --trpo-iters $trpo_iters"
[[ -n "$device"        ]] && run_py_args+=" --device $device"
[[ -n "$k_max"         ]] && run_py_args+=" --k-max $k_max"
[[ -n "$sens_axis"     ]] && run_py_args+=" --sens-axis $sens_axis"
[[ "$no_hessian"  == true ]] && run_py_args+=" --no-hessian"
[[ "$diagnostics" == true ]] && run_py_args+=" --diagnostics"
[[ "$no_save"     == true ]] && run_py_args+=" --no-save"
[[ "$verbose"     == true ]] && run_py_args+=" --verbose"
[[ "$bdq"         == true ]] && run_py_args+=" --bdq"
[[ "$trpo"        == true ]] && run_py_args+=" --trpo"
run_py_args+=" $extra"

export TASK="$task" OUTDIR="$outdir" EXTRA_ARGS="$run_py_args"

seq 0 $((nseeds - 1)) | xargs -P "$ejobs" -n 1 bash -c '
  seed=$1
  log="$OUTDIR/${TASK}_seed${seed}.log"
  echo "[$TASK seed $seed] -> $log"
  python -u src/run.py --task "$TASK" --seed "$seed" $EXTRA_ARGS 2>&1 | tee "$log"
  echo "[$TASK seed $seed] done (exit ${PIPESTATUS[0]})"
' _

echo "all $nseeds seed(s) finished; logs in $outdir/"

# combine every seed's results.json into one (skipped if --no-save was passed in the
# extra args, since there'd be nothing to read)
if [[ "$run_py_args" != *"--no-save"* ]]; then
  python src/aggregate.py --task "$task" --mode "${mode:-sort}"
fi
