"""End-to-end experiment: train the value function (FEVI by default), compare the method
against the baselines, and write a per-run results.json for aggregate.py to fold. The file
is rewritten after every evaluated agent, so an interrupted run keeps what finished.

    python run.py                                  # FEVI + greedy (Algorithm 2) selection
    python run.py --learner fvi                    # the FVI ablation
    python run.py --task mlp --m 6 --T 8 --bdq     # override the config, add a baseline

Each run gets its own output/<task>_<mode>_<tag>_seed<n>_<timestamp>/ directory, so
concurrent seeds never write the same path. Set TORCH_NUM_THREADS=1 when running seeds
in parallel (run_seeds.sh --parallel does this) -- otherwise every process grabs all
cores and they contend.
"""
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime

import torch

from agents import (ClairvoyantAgent, GreedyAgent, MethodAgent, NullAgent, RandomAgent,
                    clairvoyant_search_size)
from aggregate import config_tag
from config import EnvConfig, MethodConfig, tier_name
from diagnostics import run_all_diagnostics
from env import Environment
from fvi import evaluate, trace_episodes, train_fevi, train_fvi

LEARNERS = {"fvi": train_fvi, "fevi": train_fevi}

ENV_OVERRIDES = ["d", "m", "T", "eta", "noise", "victim_hidden", "init_scale"]
METHOD_OVERRIDES = ["c_edit", "weight_decay", "L", "B"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="regression",
                   choices=["regression", "logistic", "mlp"])
    p.add_argument("--mode", default="greedy", choices=["greedy", "exact"])
    p.add_argument("--learner", default="fevi", choices=sorted(LEARNERS),
                   help="fevi (Algorithm 1 backward induction) | fvi (TD sweeps, ablation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="output")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                   help="auto = cuda if available, else cpu")

    # config overrides -- unset means "keep the dataclass default"
    p.add_argument("--d", type=int)
    p.add_argument("--m", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--eta", type=float)
    p.add_argument("--noise", type=float)
    p.add_argument("--victim-hidden", type=int)
    p.add_argument("--init-scale", type=float)
    p.add_argument("--c-edit", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--L", type=int, help="repetition limit: a_i in {0, ..., L}")
    p.add_argument("--B", type=int, help="budget on ||a - 1||_1")
    p.add_argument("--K", type=int, help="override K_iters")
    p.add_argument("--no-hessian", action="store_true", help="Gram-only ablation")

    # extra baselines (trained on demand -- each roughly doubles the run)
    p.add_argument("--bdq", action="store_true")
    p.add_argument("--trpo", action="store_true")
    p.add_argument("--clairvoyant", action="store_true",
                   help="exact oracle with the whole stream known; O(|A|^T) per episode")
    p.add_argument("--clairvoyant-max-nodes", type=float, default=1e7,
                   help="refuse --clairvoyant when |A|^T exceeds this (checked before training)")

    p.add_argument("--diagnostics", action="store_true", help="also write the figures")
    p.add_argument("--no-save", action="store_true", help="skip results.json")
    p.add_argument("--verbose", action="store_true", help="per-iteration training logs")
    p.add_argument("--sens-axis", help="tag naming the sensitivity axis that produced it")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--history", action="store_true",
                   help="record cost per training iteration (re-evaluates each iteration)")
    p.add_argument("--history-episodes", type=int, default=20,
                   help="episodes per --history measurement")
    p.add_argument("--trace-episodes", type=int, default=5,
                   help="episodes whose per-step actions are saved; 0 disables")
    return p.parse_args()


def build_configs(args):
    def given(names):
        return {n: getattr(args, n) for n in names if getattr(args, n) is not None}

    env_cfg = EnvConfig(task=args.task, seed=args.seed, device=resolve_device(args.device),
                        **given(ENV_OVERRIDES))
    cfg = MethodConfig(select_mode=args.mode, use_hessian_in_B=not args.no_hessian,
                       **given(METHOD_OVERRIDES))
    if args.K is not None:
        cfg.K_iters = args.K
    return env_cfg, cfg


def train(fn, env, cfg, args, histories, name):
    """Train one learner, recording its cost-per-iteration curve if asked."""
    if args.history:
        model, history = fn(env, cfg, seed=args.seed, verbose=args.verbose,
                            return_history=True, eval_episodes=args.history_episodes)
        histories[name] = history
        return model
    return fn(env, cfg, seed=args.seed, verbose=args.verbose)


def check_clairvoyant(args, env_cfg, cfg):
    """Fail before training when the oracle's exhaustive search cannot finish."""
    n_actions, nodes = clairvoyant_search_size(env_cfg.m, env_cfg.T, cfg)
    if nodes > args.clairvoyant_max_nodes:
        raise SystemExit(
            f"--clairvoyant at m={env_cfg.m}, T={env_cfg.T}: |A|={n_actions} per step, up to "
            f"{nodes:.1e} search nodes per episode (limit {args.clairvoyant_max_nodes:.0e}). "
            f"Use m<=5 and T<=4, or raise --clairvoyant-max-nodes.")


def resolve_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return choice


def run_dir(args, env_cfg, cfg):
    tag = config_tag({**asdict(env_cfg), **asdict(cfg)})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{env_cfg.task}_{cfg.select_mode}_{args.learner}_{tag}_seed{args.seed}_{stamp}"
    return os.path.join(args.outdir, name)


def main():
    args = parse_args()
    # one thread per process: with --parallel, N processes x N threads thrash the cores
    threads = os.environ.get("TORCH_NUM_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    torch.manual_seed(args.seed)

    env_cfg, cfg = build_configs(args)
    if args.clairvoyant:
        check_clairvoyant(args, env_cfg, cfg)
    env = Environment(env_cfg, cfg)

    print(f"== Training {args.learner.upper()} (task={env_cfg.task}, mode={cfg.select_mode}, "
          f"use_hessian_in_B={cfg.use_hessian_in_B}, seed={args.seed}, "
          f"device={env_cfg.device}) ==")
    histories = {}
    V_psi = train(LEARNERS[args.learner], env, cfg, args, histories, args.learner)

    rng = torch.Generator().manual_seed(args.seed + 100)
    agents = {
        "null": NullAgent(),
        "random": RandomAgent(rng, cfg.balance, cfg.B, cfg.L),
        "greedy": GreedyAgent(env.victim, env.theta_target, cfg),
        tier_name(cfg.select_mode): MethodAgent(V_psi, env.victim, env.theta_target,
                                                cfg, rng),
    }
    if args.bdq:
        from bdq import BDQAgent, train_bdq
        agents["bdq"] = BDQAgent(train(train_bdq, env, cfg, args, histories, "bdq"),
                                 env, cfg, rng)
    if args.trpo:
        from trpo import TRPOAgent, train_trpo
        agents["trpo"] = TRPOAgent(train(train_trpo, env, cfg, args, histories, "trpo"),
                                   env, cfg, rng)
    if args.clairvoyant:
        agents["clairvoyant"] = ClairvoyantAgent(env.victim, env.theta_target, cfg)

    costs, traces = {}, {}
    results = {"task": env_cfg.task, "mode": cfg.select_mode, "learner": args.learner,
               "seed": args.seed, "sens_axis": args.sens_axis,
               "env_config": asdict(env_cfg), "method_config": asdict(cfg),
               "costs": costs, "history": histories, "trace": traces, "complete": False}
    out_dir = run_dir(args, env_cfg, cfg)
    if args.diagnostics or not args.no_save:
        os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.json")

    def save():
        # rewritten after every step, so an interrupted run keeps what already finished
        if not args.no_save:
            with open(path, "w") as f:
                json.dump(results, f, indent=2)

    # every agent is scored on the SAME episodes (fresh generator, identical seed)
    print(f"\n== Mean episode cost ({args.eval_episodes} episodes) ==")
    for name, agent in agents.items():
        eval_rng = torch.Generator().manual_seed(args.seed + 100)
        costs[name] = evaluate(env, agent, eval_rng, n_episodes=args.eval_episodes)
        print(f"  {name:12s}: {costs[name]:.4f}")
        save()

    # the first --trace-episodes of the SAME episodes the costs were measured on
    if args.trace_episodes:
        for name, agent in agents.items():
            trace_rng = torch.Generator().manual_seed(args.seed + 100)
            traces[name] = trace_episodes(env, agent, trace_rng, args.trace_episodes)
        save()

    if args.diagnostics:
        print("\n== Diagnostics ==")
        summary = run_all_diagnostics(env, cfg, V_psi, outdir=out_dir)
        results["tier_cost"], results["tier_time"] = summary["cost"], summary["time"]
        print("  tier costs:", {k: round(v, 4) for k, v in summary["cost"].items()})
        print("  tier times:", {k: round(v, 5) for k, v in summary["time"].items()})
        print(f"  figures written to {out_dir}/")

    results["complete"] = True
    save()
    if not args.no_save:
        print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()
