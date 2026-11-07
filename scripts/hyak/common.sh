# Helpers sourced by submit.sh, sweep.sh and setup_env.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

load_config() {  # args: [config file] [--dry-run | --local] [KEY=VALUE ...]
  local cfg=$HERE/config.sh arg
  MODE=submit
  OVERRIDES=()
  for arg in "$@"; do
    case $arg in
      --dry-run) MODE=dry ;;
      --local) MODE=local ;;
      *=*) OVERRIDES+=("$arg") ;;
      *) cfg=$arg ;;
    esac
  done
  source "$cfg"
  for arg in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do
    [[ -v ${arg%%=*} ]] || { echo "unknown setting ${arg%%=*} (see config.sh)" >&2; exit 1; }
    printf -v "${arg%%=*}" '%s' "${arg#*=}"
  done
  CONFIG_FILE=$cfg
  PY=$CONDA_ENV/bin/python
}

expand_seeds() {  # "0-4 7" -> 0 1 2 3 4 7
  local x
  for x in ${1//,/ }; do
    if [[ $x == *-* ]]; then seq "${x%-*}" "${x#*-}"; else echo "$x"; fi
  done
}

shared_args() {  # victim, attack, budget and network flags for every job
  local a="--name $NAME --m $M --T $T --eta $ETA --env_seed $ENV_SEED --L $L --B $B --lam $LAM --balance $BALANCE"
  a+=" --victim_hidden $VICTIM_HIDDEN --victim_depth $VICTIM_DEPTH --victim_act $VICTIM_ACT"
  a+=" --init_scale $INIT_SCALE --target_scale $TARGET_SCALE --target_steps $TARGET_STEPS"
  if [[ -n $DATASET ]]; then
    a+=" --dataset $DATASET --dataset_n $DATASET_N --dataset_pca $DATASET_PCA --dataset_noise $DATASET_NOISE"
    a+=" --dataset_test $DATASET_TEST"
    [[ $TASK == mlp ]] && a+=" --task mlp"
  else
    a+=" --task $TASK --d $D --noise $NOISE --theta_star_norm $THETA_STAR_NORM"
  fi
  a+=" --train_episodes $TRAIN_EPISODES --N_episodes $EPISODES_PER_ITER --eval_episodes $EVAL_EPISODES"
  echo "$a --hidden $HIDDEN --weight_decay $WEIGHT_DECAY --myopic_prior $MYOPIC_PRIOR${EXTRA:+ $EXTRA}"
}

tuned() {  # args: learner, hp | trials -> that entry of HP_FROM for NAME
  python3 - "$HP_FROM" "$NAME" "$1" "$2" <<'EOF'
import json, sys
path, name, learner, field = sys.argv[1:]
s = json.load(open(path))["selected"][name][learner]
fmt = lambda v: ",".join(map(str, v)) if isinstance(v, list) else v
print(" ".join(f"{k}={fmt(v)}" for k, v in s["hp"].items()) if field == "hp" else s["trials"])
EOF
}

learner_hp() {  # arg: learner -> --hp flags: config.sh values, then HP_FROM's tuned ones, then EXTRA_<LEARNER>
  local kv x
  case $1 in
    fevi) kv="lr=$FEVI_LR fevi_fit_steps=$FEVI_FIT_STEPS fevi_M=$FEVI_M" ;;
    trpo) kv="trpo_delta=$TRPO_DELTA trpo_vf_iters=$TRPO_VF_ITERS trpo_vf_lr=$TRPO_VF_LR trpo_actor=$TRPO_ACTOR" ;;
    bdq) kv="lr=$BDQ_LR n_grad_steps=$BDQ_GRAD_STEPS batch_size=$BDQ_BATCH" ;;
  esac
  [[ -n $HP_FROM ]] && kv+=" $(tuned "$1" hp)"
  case $1 in
    fevi) kv+=" $EXTRA_FEVI" ;;
    trpo) kv+=" $EXTRA_TRPO" ;;
    bdq) kv+=" $EXTRA_BDQ" ;;
  esac
  for x in $kv; do printf -- '--hp %s ' "$x"; done
}

job_line() {  # args: learner, seed, output pickle, [more flags]
  echo "$PY $REPO/scripts/run_experiment.py $(shared_args) --learner $1 --seed $2 ${4:-}--out $3"     # later --hp wins
}

prepare_run() {  # arg: function printing the job lines -> JOBS file (RUN_DIR/jobs.txt unless --dry-run)
  local lines
  lines=$("$1")
  if [[ $MODE == dry ]]; then
    JOBS=$(mktemp)
    echo "$lines" > "$JOBS"
    return
  fi
  [[ -x $PY ]] || { echo "no python at $PY: run setup_env.sh or set CONDA_ENV" >&2; exit 1; }
  JOBS=$RUN_DIR/jobs.txt
  if [[ -f $JOBS && $(cat "$JOBS") != "$lines" ]]; then
    echo "$RUN_DIR was run with other settings: pick a new RUN_NAME or delete that directory" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR/out" "$RUN_DIR/logs"
  echo "$lines" > "$JOBS"
  { cat "$CONFIG_FILE"; echo "# command-line overrides"; printf '%s\n' ${OVERRIDES[@]+"${OVERRIDES[@]}"}; } \
    > "$RUN_DIR/config.sh"
}

submit_jobs() {  # args: jobs file, command to run once every job succeeded
  local jobs=$1 then=$2 n id line
  n=$(wc -l < "$jobs")
  local sb=(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" --mem="$MEM"
            --output="$RUN_DIR/logs/%x_%A_%a.out")
  [[ -n $CONSTRAINT ]] && sb+=(--constraint="$CONSTRAINT")
  case $MODE in
    dry)
      cat "$jobs"
      echo
      echo "${sb[*]} --job-name=$RUN_NAME --array=0-$((n - 1))%$MAX_PARALLEL $HERE/array.slurm $RUN_DIR/jobs.txt"
      echo "then, after all $n jobs succeed: $then"
      rm -f "$jobs" ;;
    local)
      export OMP_NUM_THREADS=1
      while IFS= read -r line; do eval "$line"; done < "$jobs"
      eval "$then" ;;
    submit)
      echo "$then" > "$RUN_DIR/then.txt"
      id=$("${sb[@]}" --job-name="$RUN_NAME" --array=0-$((n - 1))%"$MAX_PARALLEL" "$HERE/array.slurm" "$jobs")
      "${sb[@]}" --job-name="${RUN_NAME}_then" --dependency=afterok:"$id" --array=0 \
        "$HERE/array.slurm" "$RUN_DIR/then.txt" > /dev/null
      echo "submitted $n jobs as array $id (+ follow-up); logs in $RUN_DIR/logs; watch with: squeue -u $USER" ;;
  esac
}
