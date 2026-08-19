"""End-to-end experiment: train the FVI value function, compare the method against
the baselines, and produce the three diagnostic figures.

    python run.py                 # sort selection (default)
    python run.py --mode exact    # exact BQP selection (brute force here; no gurobi)
    python run.py --mode lowrank --rank 2

This reconstructs a minimal Brief I environment (linear-regression victim) and
runs the Brief II method on top of it.
"""
import argparse

import torch

from agents import GreedyAgent, MethodAgent, NullAgent, RandomAgent
from config import EnvConfig, MethodConfig, tier_name
from diagnostics import run_all_diagnostics
from env import Environment
from fvi import evaluate, train_fvi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="sort", choices=["sort", "lowrank", "exact"])
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--no-hessian", action="store_true", help="Gram-only ablation")
    p.add_argument("--K", type=int, default=None, help="override K_iters")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env_cfg = EnvConfig()
    cfg = MethodConfig(select_mode=args.mode, rank_r=args.rank,
                       use_hessian_in_B=not args.no_hessian)
    if args.K is not None:
        cfg.K_iters = args.K

    env = Environment(env_cfg, cfg)

    print(f"== Training FVI (mode={cfg.select_mode}, "
          f"use_hessian_in_B={cfg.use_hessian_in_B}) ==")
    V_psi = train_fvi(env, cfg, seed=args.seed)

    # --- compare against baselines (mean episode cost; lower is better) ---
    rng = torch.Generator().manual_seed(args.seed + 100)
    baselines = {
        "null": NullAgent(),
        "random": RandomAgent(rng, cfg.balance, cfg.k_max),
        "greedy": GreedyAgent(env.victim, env.theta_target, cfg),
        tier_name(cfg.select_mode): MethodAgent(V_psi, env.victim, env.theta_target,
                                                cfg, rng),
    }
    print("\n== Mean episode cost (100 episodes) ==")
    for name, agent in baselines.items():
        print(f"  {name:12s}: {evaluate(env, agent, rng, n_episodes=100):.4f}")

    # --- diagnostics + the three required figures ---
    print("\n== Diagnostics ==")
    summary = run_all_diagnostics(env, cfg, V_psi, outdir=args.outdir)
    print("  tier costs:", {k: round(v, 4) for k, v in summary["cost"].items()})
    print("  tier times:", {k: round(v, 5) for k, v in summary["time"].items()})
    print(f"  figures written to {args.outdir}/fig1_spectrum.png, "
          f"fig2_rel_interaction.png, fig3_tiers.png")


if __name__ == "__main__":
    main()
