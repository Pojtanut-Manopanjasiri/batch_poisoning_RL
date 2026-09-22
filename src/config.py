"""Configuration for the batch-curation poisoning experiment.

Two dataclasses:
  - EnvConfig     : the environment / victim (reconstructed from Brief I).
  - MethodConfig  : the FVI + decomposition + selection method (Brief II, sec. 6).
"""
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment and victim setup.

    The problem recap in Brief II fully pins down the environment except for the
    victim's per-point loss.  We use least-squares linear regression, the minimal
    choice that keeps the transition action-linear (no clipping / normalisation).
    """
    d: int = 5          # parameter dimension  (theta in R^d)
    m: int = 8          # batch size           (points per step)
    T: int = 6          # finite horizon
    eta: float = 0.2    # victim learning rate
    noise: float = 0.1  # regression: label-noise std;  logistic: label-flip probability
    seed: int = 0
    # "regression" (least squares) | "logistic" (binary CE) | "mlp" (small neural net)
    task: str = "regression"
    victim_hidden: int = 8     # mlp task: hidden width of the victim network
    init_scale: float = 0.3    # mlp task: std of the victim's initial weights
    target_scale: float = 1.0  # mlp task: std of the target weights; > init_scale so
                               # f(.;theta_target) is visibly non-linear


@dataclass
class MethodConfig:
    # --- value learning (FVI) ---
    hidden: tuple = (128, 128)
    time_encoding: str = "remaining"  # (T-t) in [-1,1], Pardo et al. 2018 | "elapsed" t/T
    K_iters: int = 20          # outer FVI iterations
    N_episodes: int = 50       # rollouts collected per iteration
    n_grad_steps: int = 500    # SGD steps per fit round
    batch_size: int = 256      # minibatch size for the fit (not in the brief)
    buffer_capacity: int = 20000   # replay buffer: transitions kept across iterations
    ema_decay: float | None = None # Polyak (EMA) averaging of value-net weights; None = off
    lr: float = 1e-3
    weight_decay: float = 0.0  # L2 penalty on value-net weights (Adam); constrains curvature
    eps_start: float = 0.5
    eps_end: float = 0.05

    # --- TRPO baseline (trpo.py); on-policy, hard KL trust region ---
    gamma: float = 1.0              # finite horizon, undiscounted
    gae_lambda: float = 0.95
    trpo_delta: float = 0.01        # KL budget per update
    trpo_damping: float = 0.1       # Fisher damping (conditions the CG solve)
    trpo_cg_iters: int = 10
    trpo_backtracks: int = 10       # line-search halvings before rejecting the step
    trpo_minibatch: int = 64        # critic regression minibatch
    trpo_vf_iters: int = 40         # critic regression steps per iteration
    trpo_vf_lr: float = 1e-3

    # --- decomposition ---
    use_hessian_in_B: bool = True   # False => Gram-only ablation (much faster)

    # --- FEVI (train_fevi; Algorithm 1) -------------------------------------
    # N/M are Algorithm 1's sample sizes.  fevi_outer is NOT in Algorithm 1: it
    # regenerates {R_t} under the improved policy (Algorithm 1 takes them as given).
    fevi_N: int = 200               # states sampled per stage  (N)
    fevi_M: int = 5                 # fresh batches per state   (M)
    fevi_outer: int = 5             # collect -> backward-pass rounds
    fevi_fit_steps: int = 300       # gradient steps per stage regression
    fevi_shared: bool = False       # one net with t as a feature, pooled across stages,
                                    # instead of a separate omega_t per stage

    # --- selection ---
    select_mode: str = "greedy"     # "greedy" (Algorithm 2) | "exact"
    solver: str = "gurobi"          # "gurobi" | "brute" (exact tier)

    # --- constraints (paper notation) ---
    L: int = 2                      # repetition limit: a_i in {0, ..., L}
    B: int | None = None            # budget on ||a - 1||_1; None = unbounded
    balance: bool = True            # batch-size preservation: sum_i a_i = m
    c_edit: float = 0.0             # per-edit cost; the paper's lambda / 2m
    lam: float = 1.0                # weight of the nefarious term


# --------------------------------------------------------------------------- #
# Display names
# --------------------------------------------------------------------------- #
# The method is named after the selector it actually runs, so a plot legend or a
# results table says which problem was solved rather than the generic "method".
# These are DISPLAY names only -- `select_mode` keeps its short identifiers.
TIER_NAME = {"greedy": "first-order", "exact": "exact"}
TIER_MODE = {v: k for k, v in TIER_NAME.items()}     # inverse: display name -> select_mode


def tier_name(mode):
    """Display name for a select_mode ('greedy' -> 'first-order', ...)."""
    try:
        return TIER_NAME[mode]
    except KeyError:
        raise KeyError(f"unknown select_mode {mode!r}; expected one of {list(TIER_NAME)}")
