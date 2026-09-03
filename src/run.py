import argparse
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import torch

from agents import GreedyAgent, MethodAgent, NullAgent, RandomAgent
from bdq import BDQAgent, train_bdq
from config import EnvConfig, MethodConfig, tier_name
from diagnostics import run_all_diagnostics
from env import Environment
from fvi import rollout, train_fvi
from trpo import TRPOAgent, train_trpo

def evaluate(env, agent, rng, n_episodes=100, name=None, verbose=False):
    """Mean episode cost of a policy, and its mean decision latency (seconds/action,
    timed around agent.act() only -- see fvi.rollout). Prints running mean/std every
    ~20% of episodes when verbose (in addition to train_fvi's own per-iteration
    progress, this covers the final evaluation stage, which otherwise runs silently)."""
    log_every = max(1, n_episodes // 5)
    costs, decision_times = [], []
    for i in range(1, n_episodes + 1):
        costs.append(rollout(env, agent, rng, decision_times=decision_times)[1])
        if verbose and (i % log_every == 0 or i == n_episodes):
            print(f"    [{name}] episode {i:3d}/{n_episodes}: "
                  f"mean={np.mean(costs):.4f}  std={np.std(costs):.4f}")
    return float(np.mean(costs)), float(np.mean(decision_times))


def resolve_device(name):
    """"auto" -> cuda if available, else mps, else cpu. Anything else passes through."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# fields sensitivity/*.sh sweeps -- embedded in the saved output's directory/file names
# (see config_tag below) so a listing of output/ is self-describing, and aggregate.py
# can tell two different sweep points apart even if their timestamps collide.
CONFIG_FIELDS = ["c_edit", "eta", "m", "T", "d", "rank_r", "k_max"]


def _fmt(x):
    if x is None:
        return "na"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def config_tag(cfg):
    """cfg: a dict with at least CONFIG_FIELDS (e.g. merged env_config + method_config)."""
    return (f"ce{_fmt(cfg['c_edit'])}_eta{_fmt(cfg['eta'])}_m{_fmt(cfg['m'])}"
            f"_T{_fmt(cfg['T'])}_d{_fmt(cfg['d'])}_r{_fmt(cfg['rank_r'])}_k{_fmt(cfg['k_max'])}")


TASK_DEFAULTS = {
    "regression": dict(d=5, m=10, T=10, eta=0.2, noise=0.1, c_edit=0.1, weight_decay=1e-3,
                        dual_iters=20, K_iters=30, victim_hidden=8, init_scale=0.3, k_max=None),
    "logistic":   dict(d=5, m=10, T=10, eta=1.0, noise=0.1, c_edit=0.02, weight_decay=1e-3,
                        dual_iters=20, K_iters=20, victim_hidden=8, init_scale=0.3, k_max=None),
    "mlp":        dict(d=5, m=10, T=10, eta=0.3, noise=0.1, c_edit=0.02, weight_decay=1e-3,
                        dual_iters=20, K_iters=50, victim_hidden=8, init_scale=0.3, k_max=None),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="regression", choices=list(TASK_DEFAULTS))
    p.add_argument("--mode", default="sort", choices=["sort", "lowrank", "exact"])
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--no-hessian", action="store_true", help="Gram-only ablation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diagnostics", action="store_true", help="also write the 3 figures")
    p.add_argument("--no-save", action="store_true",
                    help="don't write results.json (output dir is auto-named; not user-settable)")
    p.add_argument("--verbose", action="store_true", help="print FVI's per-iteration training progress")
    p.add_argument("--bdq", action="store_true", help="also train a BDQ agent (slow: ~13s/iter)")
    p.add_argument("--trpo", action="store_true", help="also train a TRPO agent (converges later than the others)")
    p.add_argument("--trpo-iters", type=int, default=None,
                    help="override TRPO's K_iters (default: max(K_iters, 50))")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="auto picks cuda > mps > cpu. Speeds up BDQ/TRPO's batched grad "
                         "steps and FVI's TD-fit step; rollout/decomposition always run "
                         "on CPU regardless (measured not to benefit from a GPU there)")

    # overrides for the task's default env/method settings -- unset (None) means "use
    # the task's default"; useful for sweeping one axis at a time via sweep.sh
    p.add_argument("--d", type=int, default=None,
                    help="problem dimension (input dim for mlp; theta's dim otherwise)")
    p.add_argument("--m", type=int, default=None, help="batch size: points per step")
    p.add_argument("--T", type=int, default=None, help="horizon: steps per episode")
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--noise", type=float, default=None)
    p.add_argument("--c-edit", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--K", type=int, default=None, help="override K_iters")
    p.add_argument("--dual-iters", type=int, default=None)
    p.add_argument("--victim-hidden", type=int, default=None, help="mlp task only")
    p.add_argument("--init-scale", type=float, default=None, help="mlp task only")
    p.add_argument("--k-max", type=int, default=None,
                    help="hard budget on #deletes per step (None: unconstrained, only c_edit prices edits)")
    p.add_argument("--sens-axis", default=None,
                    help="label recorded verbatim in results.json (e.g. 'c_edit', 'm') identifying "
                         "which sensitivity/*.sh sweep produced this run; unset for a plain run. "
                         "Lets the notebook trust this tag instead of inferring axis membership "
                         "from config content, which can't tell one axis's own deliberate "
                         "co-variation (e.g. m.sh moving c_edit with m) apart from an unrelated "
                         "sweep that happens to share the same value.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    defaults = TASK_DEFAULTS[args.task]

    def pick(flag_val, key):
        return flag_val if flag_val is not None else defaults[key]

    env_cfg = EnvConfig(task=args.task,
                         d=pick(args.d, "d"),
                         m=pick(args.m, "m"),
                         T=pick(args.T, "T"),
                         eta=pick(args.eta, "eta"),
                         noise=pick(args.noise, "noise"),
                         victim_hidden=pick(args.victim_hidden, "victim_hidden"),
                         init_scale=pick(args.init_scale, "init_scale"))

    cfg = MethodConfig(select_mode=args.mode, rank_r=args.rank,
                        use_hessian_in_B=not args.no_hessian,
                        k_max=pick(args.k_max, "k_max"),
                        c_edit=pick(args.c_edit, "c_edit"),
                        weight_decay=pick(args.weight_decay, "weight_decay"),
                        dual_iters=pick(args.dual_iters, "dual_iters"),
                        K_iters=pick(args.K, "K_iters"))

    env = Environment(env_cfg, cfg)
    device = resolve_device(args.device)

    print(f"== task={args.task}  mode={cfg.select_mode}  seed={args.seed}  "
          f"K_iters={cfg.K_iters}  use_hessian_in_B={cfg.use_hessian_in_B}  "
          f"bdq={args.bdq}  trpo={args.trpo}  device={device} ==")
    # eval_seed is FIXED per agent (not one shared/advancing generator): every policy
    # is scored on the identical held-out episodes, and evaluating any one agent in
    # isolation later (e.g. after reloading a checkpoint) reproduces its number exactly,
    # independent of which other agents were evaluated before it. The same fixed
    # eval_seed drives each trainer's own per-iteration held-out eval too, so a tier's
    # training curve (history) ends at the same number as its final `costs` entry.
    eval_seed = args.seed + 100

    t0 = time.perf_counter()
    V_psi, hist = train_fvi(env, cfg, seed=args.seed, verbose=args.verbose,
                             return_history=True, eval_episodes=100, eval_seed=eval_seed,
                             device=device)
    train_times = {tier_name(cfg.select_mode): time.perf_counter() - t0}
    histories = {tier_name(cfg.select_mode): hist}

    baselines = {
        "null": NullAgent(),
        "random": RandomAgent(torch.Generator().manual_seed(eval_seed + 1),
                              cfg.balance, cfg.k_max),
        "greedy": GreedyAgent(env.victim, env.theta_target, cfg),
        tier_name(cfg.select_mode): MethodAgent(V_psi, env.victim, env.theta_target,
                                                cfg, torch.Generator().manual_seed(eval_seed + 1)),
    }

    if args.bdq:
        if args.verbose:
            print("\ntraining BDQ...")
        t0 = time.perf_counter()
        net, histories["BDQ"] = train_bdq(env, cfg, seed=args.seed, verbose=args.verbose,
                                           return_history=True, best_checkpoint=True,
                                           eval_episodes=100, eval_seed=eval_seed,
                                           device=device)
        train_times["BDQ"] = time.perf_counter() - t0
        baselines["BDQ"] = BDQAgent(net, env, cfg, torch.Generator().manual_seed(eval_seed + 1))

    if args.trpo:
        trpo_iters = args.trpo_iters if args.trpo_iters is not None else max(cfg.K_iters, 50)
        cfg_trpo = replace(cfg, K_iters=trpo_iters)
        if args.verbose:
            print(f"\ntraining TRPO ({trpo_iters} iters)...")
        t0 = time.perf_counter()
        net, histories["TRPO"] = train_trpo(env, cfg_trpo, seed=args.seed, verbose=args.verbose,
                                             return_history=True, best_checkpoint=True,
                                             eval_episodes=100, eval_seed=eval_seed,
                                             device=device)
        train_times["TRPO"] = time.perf_counter() - t0
        baselines["TRPO"] = TRPOAgent(net, env, cfg_trpo, torch.Generator().manual_seed(eval_seed + 1),
                                      deterministic=True)

    print("\n== Mean episode cost (100 episodes) ==")
    costs, decision_times = {}, {}
    for name, agent in baselines.items():
        if args.verbose:
            print(f"  evaluating {name}...")
        costs[name], decision_times[name] = evaluate(
            env, agent, torch.Generator().manual_seed(eval_seed),
            n_episodes=100, name=name, verbose=args.verbose)
        print(f"  {name:12s}: {costs[name]:.4f}  (decision: {decision_times[name] * 1000:.3f} ms)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = config_tag({**asdict(env_cfg), **asdict(cfg)})
    run_dir = os.path.join("output", f"{args.task}_{cfg.select_mode}_{tag}_seed{args.seed}_{timestamp}")

    if not args.no_save:
        os.makedirs(run_dir, exist_ok=True)
        result_path = os.path.join(run_dir, "results.json")
        with open(result_path, "w") as f:
            json.dump({
                "task": args.task, "mode": cfg.select_mode, "seed": args.seed,
                "eval_seed": eval_seed, "sens_axis": args.sens_axis,
                "env_config": asdict(env_cfg), "method_config": asdict(cfg),
                "costs": costs, "history": histories, "train_times": train_times,
                "decision_times": decision_times,
            }, f, indent=2)
        print(f"\nsaved results to {result_path}")

    # --- diagnostics + the three required figures (opt-in: skip by default so a
    # parallel sweep over many seeds doesn't regenerate plots for every run) ---
    if args.diagnostics:
        os.makedirs(run_dir, exist_ok=True)   # may not exist yet if --no-save was passed
        print("\n== Diagnostics ==")
        summary = run_all_diagnostics(env, cfg, V_psi, outdir=run_dir)
        print("  tier costs:", {k: round(v, 4) for k, v in summary["cost"].items()})
        print("  tier times:", {k: round(v, 5) for k, v in summary["time"].items()})
        print(f"  figures written to {run_dir}/fig1_spectrum.png, "
              f"fig2_rel_interaction.png, fig3_tiers.png")


if __name__ == "__main__":
    main()
