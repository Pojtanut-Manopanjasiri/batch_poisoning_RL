# Settings for submit.sh / sweep.sh / setup_env.sh; override any of them per call: bash submit.sh D=15 LAM=0.5

# --- Slurm (run `hyakalloc` on klone to see your accounts and partitions)
ACCOUNT=CHANGE_ME
PARTITION=compute                  # or ckpt-all: idle nodes, preemptible (jobs requeue and restart)
TIME=04:00:00                      # per job; the slowest is BDQ with 1000 grad steps (~1 h on a laptop core)
MEM=2G
CONSTRAINT=                        # pin one node type so wall-clock times are comparable across learners
MAX_PARALLEL=100                   # array tasks running at once
CONDA_ENV=/gscratch/CHANGE_ME/$USER/envs/batch-poisoning-rl    # created by setup_env.sh

# --- Run
OUT_ROOT=$REPO/results/hyak        # each run writes to $OUT_ROOT/$RUN_NAME (sweeps: ${RUN_NAME}_tune)
RUN_NAME=linear_d5
NAME=linear                        # result label the notebook looks up: linear | logistic | <dataset>
SEEDS=0-9                          # training seeds: "0-9", "0 3 7" or "0-4 9"
LEARNERS="fevi bdq trpo"
RESULT_SUFFIX=_fair                # merged file-name suffix (experiment.ipynb: _fair, experiment_real.ipynb: empty)
EXPORT_EPISODES=1                  # predictions JSON for each policy's first N test-episode victims; all; 0 = none

# --- Victim
TASK=regression                    # regression | logistic | mlp (neural-net victim; datasets infer it unless mlp)
DATASET=                           # empty = synthetic Gaussian stream; else wine | airfoil | breast_cancer | mnist<k>
D=5                                # data dimension (synthetic; datasets use their feature count)
M=10                               # victim batch size m
T=10                               # horizon
ETA=0.2                            # victim learning rate; auto = experiment_real.ipynb's grid search (datasets only)
NOISE=0.1                          # label noise (synthetic)
THETA_STAR_NORM=2.95               # ||theta*|| (synthetic)
ENV_SEED=0                         # draws theta*, theta_0 and theta_target
DATASET_N=10000                    # pool size (datasets)
DATASET_PCA=30                     # PCA dimension (mnist)
DATASET_NOISE=0.0                  # extra label noise (datasets)
DATASET_TEST=0.2                   # held-out share the victim is scored on (mnist: its 10k test images); 0 = the pool
VICTIM_HIDDEN=8                    # mlp: hidden width
VICTIM_DEPTH=1                     # mlp: hidden layers
VICTIM_ACT=tanh                    # mlp: tanh | relu | gelu
INIT_SCALE=0.3                     # mlp: theta_0 = INIT_SCALE * randn
TARGET_SCALE=1.0                   # mlp random target: TARGET_SCALE * randn
TARGET_STEPS=none                  # none: random target; N: target = victim's SGD on flipped labels for N steps

# --- Attack
L=2                                # max copies of one point
B=none                             # per-batch edit budget ||a - 1||_1; none = only L limits it
LAM=1.0                            # edit-cost weight lambda (rho = lambda / 2m)
BALANCE=true                       # keep sum_i a_i = m

# --- Budget and evaluation
TRAIN_EPISODES=2500                # real episodes per learner
EPISODES_PER_ITER=50               # episodes collected per update (N_episodes)
EVAL_EPISODES=200                  # per validation (seed 998) and test (seed 999) evaluation

# --- Networks (all learners)
HIDDEN=128,128
WEIGHT_DECAY=1e-3
MYOPIC_PRIOR=false                 # true: TRPO/BDQ start from the separable one-step score (ablation)

# --- Learner hyperparameters (HP_FROM overrides the keys it tuned)
FEVI_LR=1e-3
FEVI_FIT_STEPS=150                 # value-net steps per stage
FEVI_M=5                           # simulated next states per sample
TRPO_DELTA=0.05                    # KL trust region
TRPO_VF_ITERS=100                  # critic steps per update
TRPO_VF_LR=1e-3
TRPO_ACTOR=alloc                   # alloc | swap
BDQ_LR=1e-3
BDQ_GRAD_STEPS=250                 # per update
BDQ_BATCH=256                      # replay minibatch
HP_FROM=                           # selected.json from sweep.sh, e.g. results/hyak/linear_d5_tune/selected.json
EXTRA=                             # more flags for every job, e.g. "--gae_lambda 0.9 --solver brute"
EXTRA_FEVI=                        # more key=value overrides for one learner, e.g. "fevi_shared=true"
EXTRA_TRPO=
EXTRA_BDQ=

# --- Tuning grid (sweep.sh): every combination x TUNE_SEEDS, scored on seed 997
TUNE_SEEDS="100 101"
GRID_FEVI_LR="1e-3 3e-3"
GRID_FEVI_FIT_STEPS="50 150"
GRID_FEVI_M="5 20"
GRID_TRPO_DELTA="0.02 0.05 0.1 0.2"
GRID_TRPO_VF_ITERS="40 100"
GRID_BDQ_GRAD_STEPS="1000 250"
GRID_BDQ_LR="3e-4 1e-3 3e-3 1e-2"
