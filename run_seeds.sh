#!/usr/bin/env bash
# Run run.py for many seeds of ONE experiment. Sequential by default; add --parallel
# to train multiple seeds at once. Activate the project's conda env first
# (conda activate batch-poisoning-rl).
#
#   ./run_seeds.sh --task linear --seeds 5                          # one at a time
#   ./run_seeds.sh --task nn --seeds 3 --parallel --jobs 2 --mode exact --K 40
#   ./run_seeds.sh --task logistic --seeds 4 --m 6 --T 8 --c-edit 0.05
#   ./run_seeds.sh --task linear --seeds 2 -- --diagnostics --no-save   # anything else
#
# --task      linear | logistic | nn    (maps to run.py's regression | logistic | mlp)
# --seeds     how many seeds to run, i.e. seeds 0..N-1              [default: 1]
# --parallel  train seeds concurrently instead of one at a time     [default: off]
# --jobs      how many seeds to train at once (only with --parallel) [default: 4]
# --outdir    where per-seed LOG files go                           [default: results]
# --results-dir  where run.py writes each seed's results.json and aggregate.py folds
#             them into one file                                     [default: output]
#
# Prints a header with the resolved settings, one line per seed as it finishes (its
# costs, or FAILED with the log tail), a final tally, and aggregate.py's summary table.
# With --parallel each seed's full output goes only to its log; sequential runs also
# stream it to the terminal.
#
# Every run.py override is available directly here too, forwarded only if you pass it
# (otherwise run.py falls back to its own task default), so you don't need `--`:
#   --mode --learner --d --m --T --eta --noise --c-edit --weight-decay --K --L --B
#   --victim-hidden --init-scale --no-hessian --diagnostics --no-save --verbose
#   --bdq --trpo --clairvoyant --device --sens-axis
#   --history --history-episodes --trace-episodes --eval-episodes
#
# --sens-axis NAME: tag every seed's results.json with which sensitivity/*.sh axis
# produced it (e.g. "c_edit", "m") -- sensitivity/*.sh scripts set this themselves, no
# need to pass it by hand unless you're sweeping something outside those scripts.
#
# --parallel pins each seed to a single torch thread (TORCH_NUM_THREADS=1). Without
# that, every one of the --jobs processes grabs all cores and they contend, which is
# routinely slower than running the seeds one at a time.
#
# --device auto|cpu|cuda (default cpu). With --parallel and --device cuda, every job
# opens its own CUDA context (a few hundred MB each) on the one GPU, so keep --jobs
# small enough to fit in GPU memory.
#
# Anything after a lone `--` is ALSO forwarded verbatim, for flags not listed above
# (e.g. ones added to run.py later).

set -uo pipefail

task=""
nseeds=1
parallel=false
jobs=4
outdir="results"
results_dir="output"
extra=""

# run.py overrides -- unset ("") means "don't forward; let run.py use its own default"
mode="" learner="" d="" m="" T="" eta="" noise="" c_edit="" weight_decay=""
K="" victim_hidden="" init_scale="" L="" B="" device=""
history_episodes="" trace_episodes="" eval_episodes=""
sens_axis=""
no_hessian=false diagnostics=false no_save=false verbose=false bdq=false trpo=false
clairvoyant=false history=false

usage() {
  echo "Usage: $0 --task {regression,logistic,nn} [--seeds N] [--parallel] [--jobs N] [--outdir DIR]" >&2
  echo "          [--results-dir DIR]" >&2
  echo "          [--mode M] [--learner {fevi,fvi}] [--d D] [--m M] [--T T] [--eta E] [--noise N] [--c-edit C]" >&2
  echo "          [--weight-decay W] [--K K] [--L L] [--B B] [--victim-hidden H] [--init-scale S]" >&2
  echo "          [--no-hessian] [--diagnostics] [--no-save] [--verbose] [--bdq] [--trpo]" >&2
  echo "          [--clairvoyant] [--device {auto,cpu,cuda}] [--history]" >&2
  echo "          [--history-episodes N] [--trace-episodes N] [--eval-episodes N]" >&2
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
    --results-dir)   results_dir="$2"; shift 2 ;;
    --mode)          mode="$2"; shift 2 ;;
    --learner)       learner="$2"; shift 2 ;;
    --d)             d="$2"; shift 2 ;;
    --m)             m="$2"; shift 2 ;;
    --T)             T="$2"; shift 2 ;;
    --eta)           eta="$2"; shift 2 ;;
    --noise)         noise="$2"; shift 2 ;;
    --c-edit)        c_edit="$2"; shift 2 ;;
    --weight-decay)  weight_decay="$2"; shift 2 ;;
    --K)             K="$2"; shift 2 ;;
    --L)             L="$2"; shift 2 ;;
    --B)             B="$2"; shift 2 ;;
    --victim-hidden) victim_hidden="$2"; shift 2 ;;
    --init-scale)    init_scale="$2"; shift 2 ;;
    --no-hessian)    no_hessian=true; shift ;;
    --diagnostics)   diagnostics=true; shift ;;
    --no-save)       no_save=true; shift ;;
    --verbose)       verbose=true; shift ;;
    --bdq)           bdq=true; shift ;;
    --trpo)          trpo=true; shift ;;
    --clairvoyant)   clairvoyant=true; shift ;;
    --device)        device="$2"; shift 2 ;;
    --history)       history=true; shift ;;
    --history-episodes) history_episodes="$2"; shift 2 ;;
    --trace-episodes)   trace_episodes="$2"; shift 2 ;;
    --eval-episodes)    eval_episodes="$2"; shift 2 ;;
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
[[ -n "$learner"       ]] && run_py_args+=" --learner $learner"
[[ -n "$d"             ]] && run_py_args+=" --d $d"
[[ -n "$m"             ]] && run_py_args+=" --m $m"
[[ -n "$T"             ]] && run_py_args+=" --T $T"
[[ -n "$eta"           ]] && run_py_args+=" --eta $eta"
[[ -n "$noise"         ]] && run_py_args+=" --noise $noise"
[[ -n "$c_edit"        ]] && run_py_args+=" --c-edit $c_edit"
[[ -n "$weight_decay"  ]] && run_py_args+=" --weight-decay $weight_decay"
[[ -n "$K"             ]] && run_py_args+=" --K $K"
[[ -n "$L"             ]] && run_py_args+=" --L $L"
[[ -n "$B"             ]] && run_py_args+=" --B $B"
[[ -n "$victim_hidden" ]] && run_py_args+=" --victim-hidden $victim_hidden"
[[ -n "$init_scale"    ]] && run_py_args+=" --init-scale $init_scale"
[[ -n "$sens_axis"     ]] && run_py_args+=" --sens-axis $sens_axis"
[[ -n "$device"        ]] && run_py_args+=" --device $device"
[[ -n "$history_episodes" ]] && run_py_args+=" --history-episodes $history_episodes"
[[ -n "$trace_episodes"   ]] && run_py_args+=" --trace-episodes $trace_episodes"
[[ -n "$eval_episodes"    ]] && run_py_args+=" --eval-episodes $eval_episodes"
[[ "$no_hessian"  == true ]] && run_py_args+=" --no-hessian"
[[ "$diagnostics" == true ]] && run_py_args+=" --diagnostics"
[[ "$no_save"     == true ]] && run_py_args+=" --no-save"
[[ "$verbose"     == true ]] && run_py_args+=" --verbose"
[[ "$bdq"         == true ]] && run_py_args+=" --bdq"
[[ "$trpo"        == true ]] && run_py_args+=" --trpo"
[[ "$clairvoyant" == true ]] && run_py_args+=" --clairvoyant"
[[ "$history"     == true ]] && run_py_args+=" --history"
run_py_args+=" $extra"

no_save_any=false
[[ "$run_py_args" == *"--no-save"* ]] && no_save_any=true

given_or_default() { if [[ -n "$1" ]]; then echo "$1"; else echo "$2 (default)"; fi; }
echo "=== run_seeds: task=$task  learner=$(given_or_default "$learner" fevi)  mode=$(given_or_default "$mode" greedy)  seeds=0..$((nseeds - 1))  jobs=$ejobs $([[ "$parallel" == true ]] && echo parallel || echo sequential)  device=$(given_or_default "$device" cpu)"
args_shown=$(echo $run_py_args)          # collapse the leading/doubled spaces
echo "    run.py args: ${args_shown:-(none)}"
echo "    logs -> $outdir/    results -> $results_dir/"
[[ "$no_save_any" == true ]] && echo "    NOTE --no-save: nothing is written; costs appear only in the logs"
echo

# mode/learner in the log name: otherwise a greedy run and an exact one (or fvi and
# fevi) of the same task overwrite each other's logs
FAILS=$(mktemp)
trap 'rm -f "$FAILS"' EXIT
export TASK="$task" OUTDIR="$outdir" EXTRA_ARGS="$run_py_args --outdir $results_dir" \
       RUN_TAG="${mode:-greedy}_${learner:-fevi}" FAILS \
       STREAM=$([[ "$parallel" == true ]] && echo false || echo true)

# one torch thread per seed when several run at once -- N processes each spawning N
# intra-op threads oversubscribes the cores and is routinely slower than sequential
if [[ "$parallel" == true ]]; then
  export TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fi

run_one_seed() {
  local seed=$1 log="$OUTDIR/${TASK}_${RUN_TAG}_seed${1}.log" start=$SECONDS rc costs
  if [[ "$STREAM" == true ]]; then
    python -u src/run.py --task "$TASK" --seed "$seed" $EXTRA_ARGS 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  else
    python -u src/run.py --task "$TASK" --seed "$seed" $EXTRA_ARGS > "$log" 2>&1
    rc=$?
  fi
  if [[ $rc -eq 0 ]]; then
    # run.py's cost lines look like "  first-order : 13.6746"
    costs=$(grep -E '^  [A-Za-z-]+ *: -?[0-9.]+$' "$log" |
            awk '{printf "%s%s %.3f", sep, $1, $3; sep = " | "}')
    echo "[$TASK seed $seed] ok in $((SECONDS - start))s   $costs"
  else
    echo "$seed" >> "$FAILS"
    echo "[$TASK seed $seed] FAILED (exit $rc) after $((SECONDS - start))s -- last lines of $log:"
    tail -n 3 "$log" | sed 's/^/      /'
  fi
}
export -f run_one_seed

start_all=$SECONDS
seq 0 $((nseeds - 1)) | xargs -P "$ejobs" -n 1 bash -c 'run_one_seed "$1"' _

elapsed=$((SECONDS - start_all))
nfail=$(wc -l < "$FAILS")
nok=$((nseeds - nfail))
tally="=== $nok/$nseeds seeds ok in $((elapsed / 60))m$((elapsed % 60))s"
(( nfail > 0 )) && tally+="; failed: $(sort -n "$FAILS" | paste -sd, -)"
echo
echo "$tally   (logs in $outdir/)"

# combine every seed's results.json into one (skipped if nothing was saved)
if [[ "$no_save_any" == false ]] && (( nok > 0 )); then
  echo
  python src/aggregate.py --task "$task" --mode "${mode:-greedy}" --outdir "$results_dir"
fi
