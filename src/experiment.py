import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch

import fvi as fvi_mod
import trpo as trpo_mod
from agents import NullAgent, GreedyAgent, MethodAgent
from bdq import train_bdq, BDQAgent
from config import EnvConfig, MethodConfig
from datasets import binary_labels, dataset_size, n_features
from env import Environment, DatasetStream
from fvi import train_fevi, evaluate, rollout
from trpo import train_trpo, TRPOAgent, StateValue, PriorAllocationPolicy, max_swaps

LEARNERS = ("FEVI", "BDQ", "TRPO")
FEVI_EXACT = "FEVI-exact"                        # FEVI's value net tested with exact (MIQP) action selection
TRAIN = {"FEVI": train_fevi, "BDQ": train_bdq, "TRPO": train_trpo}
ETA_GRID = (0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)
TEST_POINTS, TEST_SEED = 10000, 1000             # synthetic victims' clean test set


def heldout_metrics(env, agent, rng, n_episodes):
    """-> mean episode cost, per-step perturbation and distance to target, and each episode's final victim."""
    cost = pert = gap = 0.0
    steps, finals = 0, []
    for _ in range(n_episodes):
        tr, c, trace = rollout(env, agent, rng, trace=True)
        cost += c
        for st in trace:
            pert += sum(abs(v - 1) for v in st["action"])
            gap += st["dist_to_target"]
            steps += 1
        finals.append(tr[-1][3])
    return cost / n_episodes, pert / steps, gap / steps, torch.stack(finals)


@contextmanager
def track_heldout(log, data):
    start, in_eval = time.perf_counter(), [0.0]

    def evaluate_logged(env, agent, rng, n_episodes=100):
        t0 = time.perf_counter()
        log["time"].append(t0 - start - in_eval[0])      # training seconds so far
        c, p, g, finals = heldout_metrics(env, agent, rng, n_episodes)
        log["cost"].append(c)
        log["pert"].append(p)
        log["gap"].append(g)
        log["quality"].append(victim_quality(env, finals, data))     # validation victims on clean data
        in_eval[0] += time.perf_counter() - t0
        return c

    saved = fvi_mod.evaluate, trpo_mod.evaluate
    fvi_mod.evaluate = trpo_mod.evaluate = evaluate_logged
    try:
        yield
    finally:
        fvi_mod.evaluate, trpo_mod.evaluate = saved


def step_profile(env, agent, n_episodes, seed):
    """-> per-step means and per-episode costs, and each episode's victim after every step (n_episodes, T, D)."""
    rng = torch.Generator().manual_seed(seed)
    runs = [rollout(env, agent, rng, trace=True) for _ in range(n_episodes)]
    eps = [r[2] for r in runs]
    cost = np.array([[st["cost"] for st in ep] for ep in eps])
    pert = np.array([[sum(abs(v - 1) for v in st["action"]) for st in ep] for ep in eps])
    gap = np.array([[st["dist_to_target"] for st in ep] for ep in eps])
    prof = {"cum": cost.cumsum(1).mean(0).tolist(), "pert": pert.mean(0).tolist(),
            "gap": gap.mean(0).tolist(), "ep": cost.sum(1).tolist()}     # ep: per-episode, for paired CIs
    return prof, torch.stack([torch.stack([tr[3] for tr in r[0]]) for r in runs])


def test_set(env):
    """Clean data the victim is scored on: the dataset's held-out points, or fresh synthetic draws."""
    if isinstance(env.stream, DatasetStream):
        s = env.stream
        X, y = (s.X_test, s.y_test) if len(s.y_test) else (s.X, s.y)       # dataset_test=0: the pool
        return X.to(env.device), y.to(env.device)
    return env.stream.sample_batch(TEST_POINTS, torch.Generator().manual_seed(TEST_SEED))


def victim_output(env, theta, X):
    """Victim model's output on X: the logit (logistic) or the predicted value."""
    return env.victim.forward(theta, X) if env.victim.task == "mlp" else X @ theta


def victim_stats(env, thetas, data):
    """Victim models (k, D) -> per-model confusion counts (logistic) or error sums (regression, mlp) on data."""
    X, y = data
    out = torch.stack([victim_output(env, th, X) for th in thetas])
    s = {"theta": thetas}
    if env.victim.task == "logistic":
        pred, y = out > 0, y.bool()                   # p > 0.5; positive class = label 1
        s.update(tp=(pred & y).sum(1), fp=(pred & ~y).sum(1), tn=(~pred & ~y).sum(1), fn=(~pred & y).sum(1))
    else:
        r, y = (out - y).double(), y.double()
        s.update(sse=(r ** 2).sum(1), sae=r.abs().sum(1), n=len(y), sst=float(((y - y.mean()) ** 2).sum()))
    return {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in s.items()}


def victim_quality(env, thetas, data):
    """victim_stats without the models, plus each model's distance to theta* (nan for mlp) and to the target."""
    s = victim_stats(env, thetas, data)
    del s["theta"]
    star = env.stream.theta_star.to(thetas)
    s["star_dist"] = ((thetas - star).norm(dim=1) if thetas.shape[1] == len(star)
                      else torch.full((len(thetas),), float("nan"))).cpu().numpy()
    s["target_dist"] = (thetas - env.theta_target).norm(dim=1).cpu().numpy()
    return s


def victim_scores(s):
    """victim_stats -> per-model accuracy/precision/recall/f1 or mse/mae/rmse/r2 (nan when undefined)."""
    if "tp" in s:
        tp, fp, tn, fn = (s[k].astype(float) for k in ("tp", "fp", "tn", "fn"))
        with np.errstate(invalid="ignore"):
            return {"accuracy": (tp + tn) / (tp + fp + tn + fn), "precision": tp / (tp + fp),
                    "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}
    mse = s["sse"] / s["n"]
    return {"mse": mse, "mae": s["sae"] / s["n"], "rmse": np.sqrt(mse), "r2": 1 - s["sse"] / s["sst"]}


def learner_cfg(name, cfg, **hp):
    if name == "FEVI":                    # FEVI collects the same episodes per update as the baselines
        cfg = replace(cfg, select_mode="greedy", fevi_outer=cfg.K_iters, fevi_N=cfg.N_episodes)
    return replace(cfg, **hp)


def train_learner(name, env, cfg, seed, eval_episodes, val_seed):
    """-> (net, validation history, held-out log, training seconds); best checkpoint on val_seed."""
    log = {"cost": [], "pert": [], "gap": [], "time": [], "quality": []}
    data = test_set(env)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with track_heldout(log, data):
        net, hist = TRAIN[name](env, cfg, seed=seed, verbose=False, return_history=True,
                                best_checkpoint=True, eval_episodes=eval_episodes, eval_seed=val_seed)
    secs = time.perf_counter() - t0
    assert np.allclose(hist, log["cost"])            # logged on the same episodes
    return net, hist, log, secs


def learner_agent(name, net, env, cfg, seed):
    rng = torch.Generator().manual_seed(seed + 1)
    if name == "FEVI":
        return MethodAgent(net, env.victim, env.theta_target, cfg, rng)
    if name == "BDQ":
        return BDQAgent(net, env, cfg, rng)
    return TRPOAgent(net, env, cfg, rng, deterministic=True)


def reference_agents(env, cfg, seed):
    rng = lambda: torch.Generator().manual_seed(seed + 1)
    agents = {"null": NullAgent(), "greedy": GreedyAgent(env.victim, env.theta_target, cfg, mode="exact")}
    if cfg.myopic_prior:                  # untrained TRPO/BDQ: where they start
        prior = PriorAllocationPolicy(env.d, env.T, max_swaps(env.m, cfg.B, cfg.L), cfg.hidden, cfg.L,
                                      env.rho, cfg.trpo_prior_beta)
        agents["prior"] = TRPOAgent(prior, env, cfg, rng())
    return agents


def evaluate_agents(env, agents, n_episodes, eval_seed, val_seed):
    """-> test costs, null/greedy learning-curve reference lines, per-step profiles, victim stats on clean data."""
    costs = {n: float(evaluate(env, a, torch.Generator().manual_seed(eval_seed), n_episodes))
             for n, a in agents.items()}
    ref, val = {}, {}
    for n in ("null", "greedy"):
        if n in agents:                               # learning-curve references, on the validation episodes
            *ref[n], val[n] = heldout_metrics(env, agents[n], torch.Generator().manual_seed(val_seed), n_episodes)
    data = test_set(env)
    profile, victim = {}, {"initial": victim_stats(env, env.theta_0[None], data),
                           "target": victim_stats(env, env.theta_target[None], data)}
    if env.stream.theta_star.numel() == env.d:        # theta*: the true parameter (datasets: least squares)
        victim["star"] = victim_stats(env, env.stream.theta_star.to(env.device)[None], data)
    for n, a in agents.items():                       # victim[n]: the models left after each test episode
        profile[n], paths = step_profile(env, a, n_episodes, eval_seed)
        victim[n] = victim_stats(env, paths[:, -1], data)
        victim[n]["theta_path"] = paths.detach().cpu().numpy()     # victim after each step t = 1..T
        E, T, D = paths.shape
        victim[n]["path"] = {k: v.reshape(E, T) if np.ndim(v) else v
                             for k, v in victim_quality(env, paths.reshape(-1, D), data).items()}
    for n, finals in val.items():                     # the references' validation victims
        victim[n]["val"] = victim_quality(env, finals, data)
    for n in agents:
        assert np.isclose(profile[n]["cum"][-1], costs[n])   # J(T-1) = test episode cost
    return costs, ref, profile, victim


def n_params(name, net, env, cfg):
    n = sum(p.numel() for p in net.parameters())
    if name == "TRPO":                    # actor + critic
        n += sum(p.numel() for p in StateValue(env.d, env.T, cfg.hidden).parameters())
    return n


def run_seed(ecfg, cfg, seed, hp, eval_episodes, eval_seed, val_seed, learners=LEARNERS):
    """Train every learner on one seed, then test them with the references; hp: {learner: overrides}."""
    torch.manual_seed(seed)
    env = Environment(ecfg, cfg)
    cfgs = {n: learner_cfg(n, cfg, **hp.get(n, {})) for n in learners}
    nets, out = {}, {"hist": {}, "logs": {}, "times": {}}
    for n in learners:
        nets[n], out["hist"][n], out["logs"][n], out["times"][n] = train_learner(
            n, env, cfgs[n], seed, eval_episodes, val_seed)
    agents = reference_agents(env, cfg, seed)
    agents.update({n: learner_agent(n, nets[n], env, cfgs[n], seed) for n in learners})
    out["costs"], out["ref"], out["profile"], out["victim"] = evaluate_agents(env, agents, eval_episodes,
                                                                               eval_seed, val_seed)
    out["params"] = {n: n_params(n, nets[n], env, cfgs[n]) for n in learners}
    return out


def victim_task(name):
    return "logistic" if binary_labels(name) else "regression"


def clean_loss(env, theta):
    X, y = env.stream.X, env.stream.y
    z = X @ theta
    if env.victim.task == "logistic":
        return float(torch.nn.functional.binary_cross_entropy_with_logits(z, y))
    return float(0.5 * ((z - y) ** 2).mean())


def dataset_victim(name, m, T, eta, n=10000, noise=0.0, pca=30, **kw):
    return EnvConfig(d=n_features(name, pca), m=m, T=T, eta=eta, task=victim_task(name), dataset=name,
                     dataset_n=n, dataset_noise=noise, dataset_pca=pca, **kw)


def eta_score(name, eta, m, T, n=10000, noise=0.0, pca=30, n_episodes=50, seed=998,
              test=EnvConfig.dataset_test):
    """Clean-training loss after T unpoisoned victim steps at learning rate eta."""
    env = Environment(dataset_victim(name, m, T, eta, n, noise, pca, dataset_test=test), MethodConfig())
    rng = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n_episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            z = env.stream.sample_batch(env.m, rng)
            theta = env.victim.update(theta, z, torch.ones(env.m)).detach()
        losses.append(clean_loss(env, theta))
    v = float(np.mean(losses))
    return v if np.isfinite(v) else np.inf


def best_eta(name, m, T, n=10000, noise=0.0, pca=30, grid=ETA_GRID, n_episodes=50, seed=998,
             test=EnvConfig.dataset_test):
    scores = [eta_score(name, e, m, T, n, noise, pca, n_episodes, seed, test) for e in grid]
    return grid[int(np.argmin(scores))], scores


def result_name(name, ecfg, lam, prior, suffix=""):
    """Results pickle name, one per victim setting."""
    tag = (f"_h{ecfg.victim_hidden}x{ecfg.victim_depth}{ecfg.victim_act}" if ecfg.task == "mlp" else "") \
        + (f"_tgt{ecfg.target_steps}" if ecfg.target_steps else "") + ("_prior" if prior else "")
    if ecfg.dataset:
        pca = f"_pca{ecfg.dataset_pca}" if ecfg.dataset.startswith("mnist") else ""
        return f"{name}{pca}_{ecfg.task}_eta{ecfg.eta:g}_m{ecfg.m}_T{ecfg.T}_n{dataset_size(ecfg.dataset, ecfg.dataset_n, ecfg.dataset_test)}" \
               f"_noise{ecfg.dataset_noise:g}_lam{lam:g}{tag}{suffix}.pkl"
    gamma = "none" if ecfg.theta_star_norm is None else f"{ecfg.theta_star_norm:g}"
    return f"{name}_m{ecfg.m}_T{ecfg.T}_gamma{gamma}_lam{lam:g}{tag}{suffix}.pkl"
