# Wine victim of notebooks/experiment_real.ipynb (its regression hyperparameters); everything else from config.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
RUN_NAME=wine
NAME=wine
DATASET=wine
ETA=auto                           # the notebook's eta grid search
LAM=0.3
TRAIN_EPISODES=1000
SEEDS=0
RESULT_SUFFIX=
FEVI_FIT_STEPS=50                  # notebook: n_grad_steps // T
TRPO_DELTA=0.05
TRPO_VF_ITERS=100
BDQ_GRAD_STEPS=250                 # logistic datasets (breast_cancer, mnist1): TRPO_DELTA=0.02, BDQ_GRAD_STEPS=100
