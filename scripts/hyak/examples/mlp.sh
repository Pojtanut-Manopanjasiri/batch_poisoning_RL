# Neural-net victim (synthetic y = sin(x . theta*) + noise); exact greedy gains ~16% over null here.
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
RUN_NAME=mlp_d5
NAME=mlp
TASK=mlp
ETA=0.5
NOISE=0.1
THETA_STAR_NORM=none
LAM=0.1                            # lambda=1 makes every edit cost more than it gains
VICTIM_HIDDEN=8
VICTIM_DEPTH=1                     # at depth 2 here, myopic greedy does worse than null
VICTIM_ACT=tanh
TARGET_STEPS=100
