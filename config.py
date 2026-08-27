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
    target_scale: float = 1.0  # mlp task: std of the attacker's target weights
                                # (bigger than init_scale so f(.;theta_target) is
                                # visibly non-linear -- small weights keep tanh
                                # in its near-linear regime)


@dataclass
class MethodConfig:
    # --- value learning (FVI) ---
    hidden: tuple = (128, 128)
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

    # --- PPO baseline (ppo.py); on-policy, so no replay / no max over actions ---
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2           # clipped-surrogate range
    ppo_epochs: int = 10            # passes over each on-policy batch
    ppo_minibatch: int = 64
    gamma: float = 1.0              # finite horizon, undiscounted
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01      # exploration (replaces the eps schedule)
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- TRPO baseline (trpo.py); hard KL trust region instead of PPO's clipping ---
    trpo_delta: float = 0.01        # KL budget per update
    trpo_damping: float = 0.1       # Fisher damping (conditions the CG solve)
    trpo_cg_iters: int = 10
    trpo_backtracks: int = 10       # line-search halvings before rejecting the step
    trpo_vf_iters: int = 40         # critic regression steps per iteration
    trpo_vf_lr: float = 1e-3

    # --- decomposition ---
    use_hessian_in_B: bool = True   # False => Gram-only ablation (much faster)

    # --- selection ---
    select_mode: str = "sort"       # "sort" | "lowrank" | "exact"
    rank_r: int = 2                 # rank kept in lowrank
    dual_iters: int = 20            # fixed-point iterations in lowrank
    solver: str = "gurobi"          # "gurobi" | "brute" (exact tier)

    # --- constraints ---
    balance: bool = True            # #deletes == #duplicates
    k_max: int | None = None        # hard budget on #deletes
    c_edit: float = 0.0             # per-edit perturbation cost
    lam: float = 1.0                # weight of the nefarious (target) term


# --------------------------------------------------------------------------- #
# Display names
# --------------------------------------------------------------------------- #
# The method is named after the selector it actually runs, so a plot legend or a
# results table says which problem was solved rather than the generic "method".
# These are DISPLAY names only -- `select_mode` keeps its short identifiers.
TIER_NAME = {"sort": "first-order", "lowrank": "low-rank", "exact": "exact"}
TIER_MODE = {v: k for k, v in TIER_NAME.items()}     # inverse: display name -> select_mode


def tier_name(mode):
    """Display name for a select_mode ('sort' -> 'first-order', ...)."""
    try:
        return TIER_NAME[mode]
    except KeyError:
        raise KeyError(f"unknown select_mode {mode!r}; expected one of {list(TIER_NAME)}")
