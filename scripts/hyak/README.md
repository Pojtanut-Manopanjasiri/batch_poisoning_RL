# Running the experiments on Hyak (klone)

Each Slurm array task trains one learner on one seed (or tests null / greedy), using 1 CPU with
one thread. A follow-up job then combines the outputs into the same results pickle the notebooks write.
Every setting lives in [config.sh](config.sh).

## One-time setup

1. Clone the repo on klone, for example under `/gscratch/<group>/$USER/`.
2. Edit `config.sh`:
   - Set `ACCOUNT` and `PARTITION`; `hyakalloc` lists the ones you can use.
   - Point `CONDA_ENV` at a path under `/gscratch`, because your home directory's quota is small.
3. If `conda` isn't available, install [Miniforge](https://github.com/conda-forge/miniforge) into `/gscratch`.
4. Start an interactive node and run the setup script. It builds the env from `environment.yml`, downloads
   the real datasets into `data/` and checks `gurobipy`:
   ```bash
   salloc -A <account> -p <partition> -c 2 --mem 8G -t 1:00:00
   bash scripts/hyak/setup_env.sh
   ```

## Settings

- **Config files:** the default is `config.sh`. Pass another file as the first argument;
  [examples/logistic.sh](examples/logistic.sh), [examples/wine.sh](examples/wine.sh) and
  [examples/mlp.sh](examples/mlp.sh) each load `config.sh` and change a few values.
- **Neural-net victim:** `TASK=mlp` selects it. Set its shape with `VICTIM_HIDDEN`, `VICTIM_DEPTH` and
  `VICTIM_ACT`. `TARGET_STEPS=N` makes the attack target the victim's own SGD on flipped labels for N steps.
- **One-off overrides:** add `KEY=VALUE` arguments, for example
  `bash scripts/hyak/submit.sh D=15 LAM=0.5 RUN_NAME=linear_d15`.
- **Fields not in `config.sh`:** use `EXTRA` for flags that apply to every learner, and `EXTRA_FEVI`,
  `EXTRA_TRPO`, `EXTRA_BDQ` for one learner. `python scripts/run_experiment.py --help` lists every field.
- **Dry run first:** `--dry-run` prints the job lines and the `sbatch` commands without submitting.
  `--local` runs everything on the current machine; use it with a tiny budget.

## Tuning (optional)

```bash
bash scripts/hyak/sweep.sh                                    # linear victim
bash scripts/hyak/sweep.sh scripts/hyak/examples/logistic.sh  # logistic victim
```

- Each sweep runs every `GRID_*` combination on each of the `TUNE_SEEDS`, scores it on seed 997, then
  writes `results/hyak/<RUN_NAME>_tune/selected.json`.
- To use one sweep's picks, pass that file as `HP_FROM` to `submit.sh`.
- The notebook reads both victims from a single file. Combine the sweeps with:
  `python scripts/pick_best.py results/hyak/linear_d5_tune results/hyak/logistic_d5_tune --out results/fair_tuning/selected.json`.

## Final runs

```bash
bash scripts/hyak/submit.sh HP_FROM=results/hyak/linear_d5_tune/selected.json
bash scripts/hyak/submit.sh scripts/hyak/examples/logistic.sh HP_FROM=results/hyak/logistic_d5_tune/selected.json
squeue -u $USER                                               # progress; logs in results/hyak/<RUN_NAME>/logs
```

When every task succeeds, the merge job writes `results/hyak/<RUN_NAME>/<name>_..._fair.pkl`.

It also writes `<name>_..._fair_predictions/seed<s>.json` (`EXPORT_EPISODES`, default 1): the victim's test
points, their true labels, and each policy's predictions from the models left after the first test episode. Read one
with `j = json.load(open(...)); pd.DataFrame(j["features"])`. To export more episodes later:
`python scripts/export_predictions.py results/hyak/<RUN_NAME> --episodes all`.

## Edit-cost sweep

```bash
bash scripts/hyak/lam_sweep.sh scripts/hyak/examples/mlp.sh SEEDS=0-4          # add --dry-run to preview
```

- Submits one full run (`submit.sh`) per value in `LAMS`, each in `results/hyak/<RUN_NAME>_lam<value>/`.
- Each FEVI job also tests its value network with exact action selection, saved as the policy `FEVI-exact`.
  It is a test-time variant: training and checkpoint selection still use FEVI's first-order selection.
  `FEVI_EXACT_TEST=true` adds it to any `submit.sh` run.
- When every run has been merged, a last job writes `results/hyak/<RUN_NAME>_lam/`: `lam_summary.csv` and
  `cost_vs_lambda_<victim>.pdf/.png`; the tables are in that folder's `logs/`.
- Raw cost rises with lambda for every policy, because lambda prices the edits. Compare the policies at the
  same lambda: cost relative to null and to greedy, and the distance part of the cost.
- Hyperparameters are the ones in the config (or `HP_FROM`) at every lambda; they are not re-tuned per value.

## Plotting the results

[notebooks/results.ipynb](../../notebooks/results.ipynb) only reads results; it never trains. By default it loads
every merged pickle under `results/hyak/`. Set `RESULT_FILES` to choose others, `METRIC` to pick the victim's score
(accuracy/precision/recall/F1 or MSE/MAE/RMSE/R²) and `DIST_TO` for ‖θ − θ*‖ or ‖θ − θ̃‖. Pickles from older code
still give the cost plots; the victim-quality plots need results from the current code.

To use the training notebooks' result cells instead:
1. Copy the merged pickle back: `rsync klone:<repo>/results/hyak/<RUN_NAME>/*.pkl results/victims/`.
   Real datasets go in `results/real/`.
2. In the notebook, set `OVERWRITE = False` so the training cell skips any victim that already has a
   pickle, then run all cells.
3. The settings in the notebook's `VICTIMS` must give the same file name (task, m, T, gamma, lambda) as the run.

**Victim quality after the attack** is always measured on clean data (real datasets: a held-out share,
`DATASET_TEST`). It is saved as confusion counts (logistic) or error sums (regression, mlp), together with the
distance to θ* (`star_dist`, none for mlp) and to the target (`target_dist`). In each seed's `runs[i]`:
- `["victim"][policy]`: each test episode's final model `theta` and its counts; `["path"]` holds the same after
  every step; `null` and `greedy` also have `["val"]`, from their validation episodes.
- `["victim"]["initial" | "target" | "star"]`: θ₀, θ̃ and θ* (the datasets' θ* is the least-squares fit).
- `["logs"][learner]["quality"]`: the validation victims at each training checkpoint.

`experiment.victim_scores(...)` turns the counts into accuracy/precision/recall/F1 or MSE/MAE/RMSE/R², one per model.

## When something fails

- **Resubmitting:** run the same command again. Jobs whose output already exists exit immediately.
- **Changed settings:** reusing a `RUN_NAME` with different settings is refused. Pick a new `RUN_NAME`.
- **A failed task:** the follow-up job never starts and stays pending (`DependencyNeverSatisfied`).
  Read the task's log, `scancel` the follow-up job, then resubmit.
- **Comparable wall-clock:** set `CONSTRAINT` to a single node type. Timing is a secondary result, so this
  only matters for the time plots.
