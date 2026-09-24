"""Compare every method on the same seeds: cost, wall-clock, and gradient evaluations.

    python src/compare.py                                   # 10 seeds, all methods
    python src/compare.py --seeds 5 --device cuda --workers 4
    python src/compare.py --methods null greedy fevi clairvoyant --m 4 --T 4

Each (method, seed) pair runs in its own worker process, one torch thread each.
Gradient evaluations count every Tensor.backward() and torch.autograd.grad() call,
split into training and evaluation. Within a seed every method is scored on the same
evaluation episodes, so per-seed differences are paired. clairvoyant is exact but
O(|A|^T) per episode -- keep m and T small when it's included.
"""
import argparse
import json
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import torch

METHODS = ["null", "random", "greedy", "clairvoyant", "fvi", "fevi", "bdq", "trpo"]
EVAL_SEED_OFFSET = 999

# small enough that clairvoyant's exhaustive search stays in seconds
ENV_DEFAULTS = dict(task="regression", d=3, m=4, T=4, eta=0.3, noise=0.1)
METHOD_DEFAULTS = dict(
    hidden=(64, 64), K_iters=10, N_episodes=20, n_grad_steps=150, batch_size=128,
    buffer_capacity=5000, L=2, B=None, balance=True, c_edit=0.05, lam=2.0,
    select_mode="greedy", solver="brute",
    fevi_N=60, fevi_M=3, fevi_outer=3, fevi_fit_steps=100, trpo_vf_iters=20,
)


class _GradCounter:
    backward = 0
    autograd_grad = 0

    @classmethod
    def take(cls):
        out = {"backward": cls.backward, "autograd_grad": cls.autograd_grad}
        cls.backward = cls.autograd_grad = 0
        return out


def _init_worker():
    torch.set_num_threads(1)
    orig_backward, orig_grad = torch.Tensor.backward, torch.autograd.grad

    def counted_backward(self, *a, **kw):
        _GradCounter.backward += 1
        return orig_backward(self, *a, **kw)

    def counted_grad(*a, **kw):
        _GradCounter.autograd_grad += 1
        return orig_grad(*a, **kw)

    torch.Tensor.backward = counted_backward
    torch.autograd.grad = counted_grad


def _build_agent(method, env, cfg, seed, history_episodes=0):
    """Returns (agent, cost-per-training-iteration or None)."""
    from agents import ClairvoyantAgent, GreedyAgent, MethodAgent, NullAgent, RandomAgent
    from fvi import train_fevi, train_fvi

    rng = torch.Generator().manual_seed(seed)
    kw = dict(seed=seed, verbose=False, best_checkpoint=False,
              return_history=bool(history_episodes))
    if history_episodes:
        kw["eval_episodes"] = history_episodes

    def fit(fn):
        out = fn(env, cfg, **kw)
        return out if history_episodes else (out, None)

    if method == "null":
        return NullAgent(), None
    if method == "random":
        return RandomAgent(rng, cfg.balance, cfg.B, cfg.L), None
    if method == "greedy":
        return GreedyAgent(env.victim, env.theta_target, cfg), None
    if method == "clairvoyant":
        return ClairvoyantAgent(env.victim, env.theta_target, cfg), None
    if method in ("fvi", "fevi"):
        model, hist = fit(train_fvi if method == "fvi" else train_fevi)
        return MethodAgent(model, env.victim, env.theta_target, cfg, rng), hist
    if method == "bdq":
        from bdq import BDQAgent, train_bdq
        model, hist = fit(train_bdq)
        return BDQAgent(model, env, cfg, rng), hist
    if method == "trpo":
        from trpo import TRPOAgent, train_trpo
        model, hist = fit(train_trpo)
        return TRPOAgent(model, env, cfg, rng, deterministic=True), hist
    raise ValueError(method)


def run_one(method, seed, env_kw, method_kw, eval_episodes, history_episodes=0,
            trace_episodes=0):
    from config import EnvConfig, MethodConfig
    from env import Environment
    from fvi import evaluate, trace_episodes as trace_fn

    cfg = MethodConfig(**method_kw)
    env = Environment(EnvConfig(seed=seed, **env_kw), cfg)
    sync = torch.cuda.synchronize if env.device.type == "cuda" else (lambda: None)

    _GradCounter.take()
    t0 = time.perf_counter()
    agent, history = _build_agent(method, env, cfg, seed, history_episodes)
    sync()
    train_time, train_grad = time.perf_counter() - t0, _GradCounter.take()

    t0 = time.perf_counter()
    cost = evaluate(env, agent, torch.Generator().manual_seed(seed + EVAL_SEED_OFFSET),
                    n_episodes=eval_episodes)
    sync()
    eval_time, eval_grad = time.perf_counter() - t0, _GradCounter.take()

    trace = None
    if trace_episodes:                    # first episodes of the ones just scored
        trace = trace_fn(env, agent, torch.Generator().manual_seed(seed + EVAL_SEED_OFFSET),
                         trace_episodes)

    return {"method": method, "seed": seed, "mean_cost": cost,
            "history": history, "trace": trace,
            "train_time_s": train_time, "eval_time_s": eval_time,
            "train_grad_evals": train_grad["backward"] + train_grad["autograd_grad"],
            "eval_grad_evals": eval_grad["backward"] + eval_grad["autograd_grad"],
            "train_grad": train_grad, "eval_grad": eval_grad}


def summarize(rows, methods):
    def mean(xs):
        return sum(xs) / len(xs)

    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["method"]] = r["mean_cost"]

    summary = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        costs = [r["mean_cost"] for r in rs]
        s = {"n_seeds": len(rs), "cost_mean": mean(costs),
             "cost_std": statistics.pstdev(costs),
             "train_time_s": mean([r["train_time_s"] for r in rs]),
             "eval_time_s": mean([r["eval_time_s"] for r in rs]),
             "train_grad_evals": mean([r["train_grad_evals"] for r in rs]),
             "eval_grad_evals": mean([r["eval_grad_evals"] for r in rs])}
        # paired gap to the oracle: same seed, same evaluation episodes
        gaps = [d[m] - d["clairvoyant"] for d in by_seed.values()
                if m in d and "clairvoyant" in d]
        if gaps and m != "clairvoyant":
            s["gap_to_clairvoyant"] = mean(gaps)
            s["seeds_beating_clairvoyant"] = sum(g < -1e-6 for g in gaps)
        summary[m] = s
    return summary


def print_table(summary):
    has_gap = any("gap_to_clairvoyant" in s for s in summary.values())
    head = (f"{'method':12s} {'cost (mean ± std)':>20s} {'train s':>9s} {'eval s':>8s} "
            f"{'train grads':>12s} {'eval grads':>11s}" + (f" {'gap to oracle':>14s}"
                                                           if has_gap else ""))
    print(head)
    print("-" * len(head))
    for m, s in summary.items():
        line = (f"{m:12s} {s['cost_mean']:10.3f} ± {s['cost_std']:7.3f} "
                f"{s['train_time_s']:9.2f} {s['eval_time_s']:8.3f} "
                f"{s['train_grad_evals']:12.0f} {s['eval_grad_evals']:11.0f}")
        if has_gap:
            line += f" {s['gap_to_clairvoyant']:14.4f}" if "gap_to_clairvoyant" in s \
                else f" {'—':>14s}"
        print(line)
    beaten = {m: s["seeds_beating_clairvoyant"] for m, s in summary.items()
              if s.get("seeds_beating_clairvoyant")}
    if beaten:
        print(f"\nWARNING: methods beat the exact oracle on some seeds: {beaten}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--workers", type=int,
                   help="default: all cores on cpu, 4 on cuda (one CUDA context each)")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--clairvoyant-max-nodes", type=float, default=1e7,
                   help="refuse clairvoyant when |A|^T exceeds this (checked before launch)")
    p.add_argument("--history", action="store_true",
                   help="record cost per training iteration (re-evaluates each iteration)")
    p.add_argument("--history-episodes", type=int, default=10)
    p.add_argument("--trace-episodes", type=int, default=0,
                   help="episodes whose per-step actions are saved")
    p.add_argument("--outdir", default="output")
    p.add_argument("--task", choices=["regression", "logistic", "mlp"])
    for name, typ in [("d", int), ("m", int), ("T", int), ("eta", float),
                      ("c-edit", float), ("lam", float), ("L", int), ("B", int),
                      ("K", int)]:
        p.add_argument(f"--{name}", type=typ)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    env_kw = {**ENV_DEFAULTS, "device": device}
    method_kw = dict(METHOD_DEFAULTS)
    for key in ("task", "d", "m", "T", "eta"):
        if getattr(args, key) is not None:
            env_kw[key] = getattr(args, key)
    for key in ("c_edit", "lam", "L", "B"):
        if getattr(args, key) is not None:
            method_kw[key] = getattr(args, key)
    if args.K is not None:
        method_kw["K_iters"] = args.K

    if "clairvoyant" in args.methods:
        from agents import clairvoyant_search_size
        from config import MethodConfig
        n_actions, nodes = clairvoyant_search_size(env_kw["m"], env_kw["T"],
                                                   MethodConfig(**method_kw))
        if nodes > args.clairvoyant_max_nodes:
            raise SystemExit(
                f"clairvoyant at m={env_kw['m']}, T={env_kw['T']}: |A|={n_actions} per step, "
                f"up to {nodes:.1e} search nodes per episode (limit "
                f"{args.clairvoyant_max_nodes:.0e}). Drop it from --methods, use m<=5 and "
                f"T<=4, or raise --clairvoyant-max-nodes.")

    workers = args.workers or (4 if device == "cuda" else os.cpu_count())
    jobs = [(m, s) for m in args.methods for s in range(args.seeds)]
    print(f"{len(jobs)} jobs ({len(args.methods)} methods x {args.seeds} seeds) on "
          f"{workers} workers, device={device}, cpu cores={os.cpu_count()}")

    rows, t_start = [], time.perf_counter()
    ctx = multiprocessing.get_context("spawn")        # CUDA can't survive a fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(run_one, m, s, env_kw, method_kw, args.eval_episodes,
                          args.history_episodes if args.history else 0,
                          args.trace_episodes): (m, s)
                for m, s in jobs}
        for fut in as_completed(futs):
            m, s = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                    # one failure shouldn't sink the rest
                print(f"  FAILED {m} seed {s}: {e!r}")
                continue
            rows.append(r)
            print(f"  done {m:12s} seed {s}  cost={r['mean_cost']:.4f}  "
                  f"train={r['train_time_s']:.2f}s  eval={r['eval_time_s']:.2f}s")
    wall = time.perf_counter() - t_start
    cpu_s = sum(r["train_time_s"] + r["eval_time_s"] for r in rows)
    print(f"\nfinished in {wall:.1f}s wall-clock ({cpu_s:.1f}s of work, "
          f"{cpu_s / wall:.1f}x from parallelism)\n")

    summary = summarize(rows, args.methods)
    print_table(summary)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir,
                        f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({"env_config": env_kw, "method_config": method_kw,
                   "eval_episodes": args.eval_episodes, "workers": workers,
                   "wall_clock_s": wall, "summary": summary, "runs": rows}, f, indent=2)
    print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()
