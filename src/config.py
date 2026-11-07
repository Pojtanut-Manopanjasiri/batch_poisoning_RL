from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment and victim setup."""
    d: int = 5          
    m: int = 8         
    T: int = 6          
    eta: float = 0.2    
    noise: float = 0.1  
    seed: int = 0
    theta_star_norm: float | None = None   #
    dataset: str | None = None    
    dataset_n: int = 10000          
    dataset_noise: float = 0.0     
    dataset_pca: int | None = 30
    dataset_test: float = 0.2              # held-out share for scoring the victim (mnist: its 10k test images); 0: the pool
    task: str = "regression"
    victim_hidden: int = 8
    victim_depth: int = 1                  # mlp victim: hidden layers of width victim_hidden
    victim_act: str = "tanh"               # mlp victim: tanh | relu | gelu
    init_scale: float = 0.3
    target_scale: float = 1.0
    target_steps: int | None = None        # None: random target; N: victim's SGD on flipped labels for N steps
    device: str = "cpu"      


@dataclass
class MethodConfig:
    hidden: tuple = (128, 128)
    time_encoding: str = "remaining"  
    K_iters: int = 20      
    N_episodes: int = 50    
    n_grad_steps: int = 500  
    batch_size: int = 256     
    buffer_capacity: int = 20000  
    ema_decay: float | None = None 
    lr: float = 1e-3
    weight_decay: float = 0.0 
    eps_start: float = 0.5
    eps_end: float = 0.05
    rfevi_backup: str = "optimality"   
    rfevi_targets: int | None = None   

    gamma: float = 1.0             
    gae_lambda: float = 0.95
    trpo_delta: float = 0.01        
    trpo_damping: float = 0.1       
    trpo_cg_iters: int = 10
    trpo_backtracks: int = 10      
    trpo_minibatch: int = 64        
    trpo_vf_iters: int = 40         
    trpo_vf_lr: float = 1e-3
    trpo_actor: str = "alloc"       # "alloc": exact Gibbs allocation (alloc_policy.py) | "swap"
    myopic_prior: bool = False      # TRPO/BDQ: add the separable one-step score (bdq._myopic_scores), learn only the correction
    trpo_prior_beta: float = 5.0    # TRPO with myopic_prior: initial inverse temperature (learned)

    fevi_N: int = 200              
    fevi_M: int = 5                
    fevi_outer: int = 5            
    fevi_fit_steps: int = 300       
    fevi_shared: bool = False       
    select_mode: str = "greedy"     
    solver: str = "gurobi"          

    L: int = 2                     
    B: int | None = None            
    balance: bool = True            
    lam: float = 0.0                

TIER_NAME = {"greedy": "first-order", "exact": "exact"}
TIER_MODE = {v: k for k, v in TIER_NAME.items()}  


def tier_name(mode):
    """Display name for a select_mode ('greedy' -> 'first-order', ...)."""
    try:
        return TIER_NAME[mode]
    except KeyError:
        raise KeyError(f"unknown select_mode {mode!r}; expected one of {list(TIER_NAME)}")
