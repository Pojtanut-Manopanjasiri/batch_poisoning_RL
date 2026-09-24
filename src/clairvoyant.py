"""Exact clairvoyant for the linear victim (task="regression").

Given the whole batch stream of an episode, the T-stage problem is a mixed-integer QP:
a_{t,i} = sum_k k w_{t,i,k} with binary w, and the bilinear products w_{t,i,k} r_{t,i}
(r = residual, affine in theta_t) are linearised exactly by big-M constraints on
[lo, hi] bounds of r. The bounds come from optimisation-based bound tightening (OBBT)
under "episode cost <= U" for the best known plan's cost U, which keeps every optimal
plan feasible. Gurobi returns the best plan found and a proven lower bound on the
optimal cost; they coincide when the solve finishes within the time limit.
"""
import multiprocessing as mp
import time

import numpy as np
import torch


def _plan_cost(theta0, target, X, y, eta, lam, c_edit, actions):
    """Episode cost of a plan, in float64."""
    theta, total = theta0.copy(), 0.0
    for Xt, yt, a in zip(X, y, actions):
        a = np.asarray(a, dtype=float)
        theta = theta - eta / len(a) * (a * (Xt @ theta - yt)) @ Xt
        total += c_edit * np.abs(a - 1).sum() + 0.5 * lam * ((theta - target) ** 2).sum()
    return total


def _forward_bounds(theta0, X, y, eta, L):
    """[lo, hi] on r_{t,i} valid for every feasible plan: theta_t stays in a ball around
    theta0 whose radius grows by the largest step a feasible action can take."""
    T, m = len(X), X[0].shape[0]
    lo, hi, R = np.zeros((T, m)), np.zeros((T, m)), 0.0
    for t in range(T):
        xn = np.linalg.norm(X[t], axis=1)
        centre = X[t] @ theta0 - y[t]
        lo[t], hi[t] = centre - xn * R, centre + xn * R
        w = np.sort((np.abs(centre) + xn * R) * xn)[::-1]
        take = np.minimum(L, np.maximum(0, m - L * np.arange(m)))   # sum a = m, a_i <= L
        R += eta / m * float(take @ w)
    return lo, hi


def _build(gp, p, lo, hi, U=None, relaxed=False, threads=1):
    theta0, target, X, y, eta, lam, c_edit, L, B = p
    T, (m, d) = len(X), X[0].shape
    K = range(L + 1)
    mdl = gp.Model(env=_genv(gp, threads))
    w = mdl.addVars(T, m, L + 1, lb=0, ub=1,
                    vtype=gp.GRB.CONTINUOUS if relaxed else gp.GRB.BINARY)
    th = mdl.addVars(T + 1, d, lb=-gp.GRB.INFINITY)
    q = mdl.addVars(T, m, L + 1, lb=-gp.GRB.INFINITY)
    r, obj = {}, 0
    for j in range(d):
        mdl.addConstr(th[0, j] == float(theta0[j]))
    for t in range(T):
        for i in range(m):
            mdl.addConstr(w.sum(t, i, "*") == 1)
            r[t, i] = gp.quicksum(float(X[t][i, j]) * th[t, j] for j in range(d)) \
                - float(y[t][i])
            for k in K[1:]:                                   # q = w * r
                mdl.addConstr(q[t, i, k] <= hi[t, i] * w[t, i, k])
                mdl.addConstr(q[t, i, k] >= lo[t, i] * w[t, i, k])
                mdl.addConstr(q[t, i, k] <= r[t, i] - lo[t, i] * (1 - w[t, i, k]))
                mdl.addConstr(q[t, i, k] >= r[t, i] - hi[t, i] * (1 - w[t, i, k]))
        mdl.addConstr(gp.quicksum(k * w[t, i, k] for i in range(m) for k in K) == m)
        edits = gp.quicksum(abs(k - 1) * w[t, i, k] for i in range(m) for k in K)
        if B is not None:
            mdl.addConstr(edits <= B)
        for j in range(d):                                    # theta_{t+1} = theta_t + a @ Phi
            mdl.addConstr(th[t + 1, j] == th[t, j] - eta / m * gp.quicksum(
                float(X[t][i, j]) * k * q[t, i, k] for i in range(m) for k in K[1:]))
        obj += c_edit * edits + 0.5 * lam * gp.quicksum(
            (th[t + 1, j] - float(target[j])) * (th[t + 1, j] - float(target[j]))
            for j in range(d))
    if relaxed and U is not None:
        mdl.addQConstr(obj <= U)
    return mdl, w, r, obj


_ENVS = {}


def _genv(gp, threads):
    if threads not in _ENVS:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", threads)
        env.start()
        _ENVS[threads] = env
    return _ENVS[threads]


def _obbt(gp, p, lo, hi, U, threads):
    """One forward pass of min/max r_{t,i} over the convex relaxation with cost <= U."""
    lo, hi = lo.copy(), hi.copy()
    for t in range(1, len(p[2])):                             # r_0 is a constant
        mdl, _, r, _ = _build(gp, p, lo, hi, U, relaxed=True, threads=threads)
        for i in range(lo.shape[1]):
            for sense, arr, pick, sign in ((gp.GRB.MINIMIZE, lo, max, -1),
                                           (gp.GRB.MAXIMIZE, hi, min, 1)):
                mdl.setObjective(r[t, i], sense)
                mdl.optimize()
                if mdl.Status == gp.GRB.OPTIMAL:
                    v = mdl.ObjVal + sign * 1e-7 * (1 + abs(mdl.ObjVal))
                    arr[t, i] = pick(arr[t, i], v)
        mdl.dispose()
    return lo, hi


def solve(theta0, target, X, y, eta, lam, c_edit, L, B, warm_plans, time_limit=120.0,
          slices=(10, 20, 40), threads=1):
    """Clairvoyant plan for one episode (float64 numpy inputs).

    warm_plans: feasible plans (lists of T actions); the cheapest seeds the solver and
    sets U. The time limit is split into slices, and before each slice the bounds are
    re-tightened with the best cost found so far. Returns dict(actions, cost, bound,
    gap, time, proven).
    """
    import gurobipy as gp

    p = (theta0, target, X, y, eta, lam, c_edit, L, B)
    t0 = time.perf_counter()
    costs = [_plan_cost(theta0, target, X, y, eta, lam, c_edit, a) for a in warm_plans]
    best = int(np.argmin(costs))
    plan, U = warm_plans[best], costs[best]
    bound = -np.inf
    lo, hi = _forward_bounds(theta0, X, y, eta, L)
    slices = list(slices) + [np.inf]
    for sl in slices:
        left = time_limit - (time.perf_counter() - t0)
        if left <= 0:
            break
        lo, hi = _obbt(gp, p, lo, hi, U * (1 + 1e-9), threads)
        mdl, w, _, obj = _build(gp, p, lo, hi, threads=threads)
        mdl.setObjective(obj, gp.GRB.MINIMIZE)
        mdl.Params.MIPGap, mdl.Params.MIPGapAbs = 0.0, 1e-9
        mdl.Params.TimeLimit = max(1.0, min(sl, time_limit - (time.perf_counter() - t0)))
        for (t, i, k), v in w.items():
            v.Start = float(round(plan[t][i]) == k)
        mdl.optimize()
        if mdl.SolCount:
            cand = [[int(sum(k * round(w[t, i, k].X) for k in range(L + 1)))
                     for i in range(len(plan[0]))] for t in range(len(plan))]
            c = _plan_cost(theta0, target, X, y, eta, lam, c_edit, cand)
            if c < U:
                plan, U = cand, c
        # the bounds exclude only plans costing more than U, so the solver's bound holds
        bound = max(bound, min(mdl.ObjBound, U))
        mdl.dispose()
        if U - bound <= 1e-6 * max(1.0, abs(U)):
            break
    return {"actions": plan, "cost": U, "bound": bound,
            "gap": (U - bound) / max(1e-12, abs(U)),
            "proven": U - bound <= 1e-6 * max(1.0, abs(U)),
            "time": time.perf_counter() - t0}


def _solve_star(kw):
    return solve(**kw)


def _plan_of(env, agent, zs):
    theta, plan = env.theta_0.clone(), []
    for t, z in enumerate(zs):
        a = agent.act(theta, z, t)
        plan.append([int(v) for v in a.tolist()])
        theta = env.victim.update(theta, z, a).detach()
    return plan


def evaluate_clairvoyant(env, cfg, rng, n_episodes, warm_agents, time_limit=120.0,
                         workers=None, slices=(10, 20, 40)):
    """Clairvoyant on the first n_episodes of `rng`'s stream: the same episodes that
    fvi.evaluate(env, agent, rng, n_episodes) scores the other policies on.

    warm_agents: policies whose plans on each episode seed the solver, so the
    clairvoyant plan is never worse than any of them. Returns one record per episode.
    """
    if getattr(env.victim, "task", None) != "regression":
        raise ValueError("the exact clairvoyant needs the linear victim (task='regression')")
    jobs = []
    for _ in range(n_episodes):
        zs = [env.stream.sample_batch(env.m, rng) for _ in range(env.T)]
        ones = [[1] * env.m for _ in range(env.T)]
        with torch.enable_grad():
            warm = [ones] + [_plan_of(env, a, zs) for a in warm_agents]
        jobs.append(dict(
            theta0=env.theta_0.double().cpu().numpy(),
            target=env.theta_target.double().cpu().numpy(),
            X=[z[0].double().cpu().numpy() for z in zs],
            y=[z[1].double().cpu().numpy() for z in zs],
            eta=env.victim.eta, lam=cfg.lam, c_edit=cfg.c_edit, L=cfg.L, B=cfg.B,
            warm_plans=warm, time_limit=time_limit, slices=slices, threads=1))
    workers = workers or mp.cpu_count()
    with mp.get_context("spawn").Pool(min(workers, n_episodes)) as pool:
        return pool.map(_solve_star, jobs, chunksize=1)
