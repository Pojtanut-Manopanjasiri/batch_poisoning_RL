import argparse
import dataclasses
import os
import pickle
import sys
import typing
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import experiment as ex
from config import EnvConfig, MethodConfig
from datasets import n_features
from env import Environment
from fvi import evaluate

LEARNER = {"fevi": "FEVI", "bdq": "BDQ", "trpo": "TRPO"}
ENV_FLAG = {"seed": "env_seed"}           # EnvConfig.seed would clash with the training seed
METHOD_FIELDS = {f.name: f for f in dataclasses.fields(MethodConfig)}


def parse(field, s):
    """Command-line string -> value of the dataclass field's type ('none' -> None, hidden '128,128')."""
    if s.lower() == "none":
        return None
    t = next((a for a in typing.get_args(field.type) if a is not type(None)), field.type)
    if t is tuple:
        return tuple(int(x) for x in s.split(","))
    if t is bool:
        if s.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"--{field.name}: expected true/false, got {s!r}")
        return s.lower() in ("true", "1")
    return t(s)


def build_parser():
    p = argparse.ArgumentParser(allow_abbrev=False,
                                description="Train one learner (or test the references) on one seed and pickle the result.")
    p.add_argument("--learner", required=True, choices=[*LEARNER, "refs"])
    p.add_argument("--name", required=True, help="result label: linear, logistic, wine, ...")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--train_episodes", type=int, default=2500, help="sets K_iters = train_episodes / N_episodes")
    p.add_argument("--eval_episodes", type=int, default=200, help="episodes per validation / test evaluation")
    p.add_argument("--eval_seed", type=int, default=999, help="test episodes")
    p.add_argument("--val_seed", type=int, default=998, help="checkpoint-selection episodes")
    p.add_argument("--tune", action="store_true", help="tuning run: only score on --score_seed")
    p.add_argument("--score_seed", type=int, default=997, help="tuning-score episodes")
    p.add_argument("--tag", default="", help="config label within a sweep, e.g. c3")
    p.add_argument("--trials", type=int, default=0, help="tuning runs behind these hyperparameters")
    p.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                   help="MethodConfig override for this learner only (repeatable)")
    p.add_argument("--out", required=True, help="output pickle; the job is skipped if it exists")
    g = p.add_argument_group("victim (EnvConfig fields)")
    for f in dataclasses.fields(EnvConfig):
        g.add_argument(f"--{ENV_FLAG.get(f.name, f.name)}", dest=f"env.{f.name}", metavar="X",
                       help=f"default {f.default}" + (", or auto (datasets)" if f.name == "eta" else ""))
    g = p.add_argument_group("method (MethodConfig fields, shared by all learners)")
    for f in METHOD_FIELDS.values():
        g.add_argument(f"--{f.name}", dest=f"cfg.{f.name}", metavar="X", help=f"default {f.default}")
    return p


def victim(args):
    kw = {f.name: parse(f, v) for f in dataclasses.fields(EnvConfig)
          if (v := args[f"env.{f.name}"]) is not None and not (f.name == "eta" and v == "auto")}
    if kw.get("dataset"):                 # task and d follow from the data
        kw.setdefault("task", ex.victim_task(kw["dataset"]))
        d = n_features(kw["dataset"], kw.get("dataset_pca", EnvConfig.dataset_pca))
        if kw.setdefault("d", d) != d:
            sys.exit(f"--d {kw['d']} does not match {kw['dataset']}'s {d} features")
    ecfg = EnvConfig(**kw)
    if args["env.eta"] == "auto":
        if not ecfg.dataset:
            sys.exit("--eta auto needs --dataset")
        eta, _ = ex.best_eta(ecfg.dataset, ecfg.m, ecfg.T, ecfg.dataset_n, ecfg.dataset_noise, ecfg.dataset_pca,
                             test=ecfg.dataset_test)
        ecfg = replace(ecfg, eta=eta)
    return ecfg


def method(args):
    cfg = MethodConfig(**{k: parse(f, v) for k, f in METHOD_FIELDS.items() if (v := args[f"cfg.{k}"]) is not None})
    K, rest = divmod(args["train_episodes"], cfg.N_episodes)
    if rest:
        sys.exit(f"--train_episodes {args['train_episodes']} is not a multiple of N_episodes {cfg.N_episodes}")
    hp = {}
    for kv in args["hp"]:
        k, _, v = kv.partition("=")
        if k not in METHOD_FIELDS:
            sys.exit(f"--hp {kv}: {k} is not a MethodConfig field")
        hp[k] = parse(METHOD_FIELDS[k], v)
    return replace(cfg, K_iters=K), hp


def main():
    args = vars(build_parser().parse_args())
    out = Path(args["out"])
    if out.exists():
        print(f"{out} exists, skipping")
        return
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)        # denormals in Adam state slow CPU math ~17x
    ecfg, (cfg, hp) = victim(args), method(args)
    seed, n_eval = args["seed"], args["eval_episodes"]
    rec = dict(name=args["name"], learner=LEARNER.get(args["learner"], "refs"), seed=seed, tag=args["tag"],
               tune=args["tune"], ecfg=ecfg, cfg=cfg, hp=hp, trials=args["trials"],
               train_episodes=args["train_episodes"], eval_episodes=n_eval, eval_seed=args["eval_seed"],
               val_seed=args["val_seed"], score_seed=args["score_seed"], argv=sys.argv[1:])
    torch.manual_seed(seed)
    env = Environment(ecfg, cfg)
    score = lambda agent: float(evaluate(env, agent, torch.Generator().manual_seed(args["score_seed"]), n_eval))

    if rec["learner"] == "refs":
        agents = ex.reference_agents(env, cfg, seed)
        if args["tune"]:
            rec["score"] = {n: score(a) for n, a in agents.items()}
        else:
            rec["costs"], rec["ref"], rec["profile"], rec["victim"] = ex.evaluate_agents(
                env, agents, n_eval, args["eval_seed"], args["val_seed"])
        summary = rec.get("score") or rec["costs"]
    else:
        n = rec["learner"]
        rec["lcfg"] = lcfg = ex.learner_cfg(n, cfg, **hp)
        net, rec["hist"], rec["log"], rec["time"] = ex.train_learner(n, env, lcfg, seed, n_eval, args["val_seed"])
        agent = ex.learner_agent(n, net, env, lcfg, seed)
        if args["tune"]:
            rec["score"] = score(agent)
        else:
            costs, _, profile, models = ex.evaluate_agents(env, {n: agent}, n_eval, args["eval_seed"],
                                                           args["val_seed"])
            rec["cost"], rec["profile"], rec["victim"] = costs[n], profile[n], models[n]
        rec["params"] = ex.n_params(n, net, env, lcfg)
        summary = {n: rec.get("score", rec.get("cost")), "train_s": rec["time"]}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, out)                  # atomic: a preempted job never leaves a half-written result
    print(f"{args['name']} {args['learner']} seed {seed} {args['tag']}".rstrip(),
          {k: round(v, 4) for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
