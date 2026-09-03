"""Plotting helpers for the experiment.

- plot_training_curves : method cost vs FVI iteration (training health).
- cost_vs_budget       : attack cost vs the delete budget k_max.
- plot_regression_attack : d=1 only -- how the victim's regression line moves under
  the attack (meaningful only for a single feature, where y = theta * x is a line).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents import NullAgent
from config import tier_name


def plot_training_curves(history, baseline_costs, path,
                         title="Cost across FVI iterations", name="ours"):
    """The method's held-out cost per FVI iteration vs the constant baselines.

    history        : list of per-iteration method costs (from train_fvi(return_history=True)).
    baseline_costs : {name: constant cost} for null / random / greedy.
    name           : display name of the policy (pass tier_name(cfg.select_mode)).
    """
    iters = list(range(len(history)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(iters, history, "o-", color="C0", lw=2, label=f"{name} (FVI)")

    best = min(iters, key=lambda i: history[i])
    ax.scatter([best], [history[best]], s=110, facecolors="none", edgecolors="C0",
               linewidths=2, zorder=5, label=f"best (iter {best}: {history[best]:.2f})")

    styles = {"null": ("0.4", "--"), "random": ("C3", ":"), "greedy": ("C1", "-.")}
    for name, cost in baseline_costs.items():
        color, ls = styles.get(name, ("gray", "--"))
        ax.axhline(cost, color=color, ls=ls, lw=2, label=f"{name} ({cost:.2f})")

    ax.set_xlabel("FVI iteration")
    ax.set_ylabel("mean episode cost (lower is better)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cost_vs_budget(env_cfg, method_cfg, budgets, path, seed=0,
                   eval_episodes=100, eval_seed=999):
    """Retrain + evaluate the method under each delete budget k_max, then plot cost vs budget.

    The value function depends on the constraint, so the method is retrained with
    k_max=b for each budget b. greedy is scored per budget too; null (no edits) is a
    budget-independent reference. At budget 0 no edits are allowed, so method, greedy
    and null coincide. Returns the raw costs.
    """
    from dataclasses import replace
    from env import Environment
    from fvi import train_fvi, evaluate
    from agents import GreedyAgent, MethodAgent

    budgets = list(budgets)
    name = tier_name(method_cfg.select_mode)
    method_costs, greedy_costs, null_cost = [], [], None
    for b in budgets:
        cfg_b = replace(method_cfg, k_max=int(b))
        torch.manual_seed(seed)
        env = Environment(env_cfg, cfg_b)
        V = train_fvi(env, cfg_b, seed=seed, verbose=False, best_checkpoint=True)
        method = MethodAgent(V, env.victim, env.theta_target, cfg_b,
                             torch.Generator().manual_seed(seed + 1))
        greedy = GreedyAgent(env.victim, env.theta_target, cfg_b, mode=cfg_b.select_mode)
        method_costs.append(evaluate(env, method, torch.Generator().manual_seed(eval_seed),
                                     eval_episodes))
        greedy_costs.append(evaluate(env, greedy, torch.Generator().manual_seed(eval_seed),
                                     eval_episodes))
        if null_cost is None:                        # budget-independent (keeps all points)
            null_cost = evaluate(env, NullAgent(), torch.Generator().manual_seed(eval_seed),
                                 eval_episodes)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(budgets, method_costs, "o-", color="C0", lw=2, label=f"{name} (FVI)")
    ax.plot(budgets, greedy_costs, "s-", color="C1", lw=2, label="greedy")
    ax.axhline(null_cost, color="0.4", ls="--", lw=2, label=f"null / no attack ({null_cost:.2f})")
    ax.set_xlabel("delete budget  k_max  (max #deletes = #duplicates)")
    ax.set_ylabel("mean episode cost (lower = stronger attack)")
    ax.set_title("Attack cost vs edit budget")
    ax.set_xticks(budgets)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"budgets": budgets, name: method_costs,
            "greedy": greedy_costs, "null": null_cost}


def _mean_target_gap(env, agent, rng, n_episodes):
    """Mean over episodes and steps of ||theta_t - theta_target|| (distance to target)."""
    total, count = 0.0, 0
    for _ in range(n_episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            z = env.stream.sample_batch(env.m, rng)
            a = agent.act(theta, z, t)
            theta = env.victim.update(theta, z, a).detach()
            total += float((theta - env.theta_target).norm())
            count += 1
    return total / count


def _eval_metric(env, agent, rng, n_episodes, metric):
    """metric='cost' -> mean episode cost; metric='gap' -> mean ||theta_t - target||."""
    if metric == "gap":
        return _mean_target_gap(env, agent, rng, n_episodes)
    from fvi import evaluate
    return evaluate(env, agent, rng, n_episodes)


_METRIC_YLABEL = {"cost": "mean episode cost", "gap": "mean ‖θ_t − θ̃‖"}


def _normalize_dim_series(raw, names, ds, metric, normalize, null=None):
    """Normalisation shared by every dimension-sweep plot (x-axis = d).

    normalize:
      None / False -> raw values (scale with d; log y-axis).
      "d"          -> divide by the dimension d (per-dimension; log y-axis).
      "null"/True  -> divide by the null / no-attack cost at each d (dimensionless
                      fraction, 1.0 = no attack; linear y-axis). Needs `null`.

    A per-d divisor cancels in any (a-b)/a ratio, so the right-panel "advantage"
    curves are identical under every choice -- only the left panel changes.
    Returns (vals, ylabel, subtitle, log_yscale).
    """
    if normalize in (True, "null"):
        if null is None:
            raise ValueError("normalize='null' requires the null-policy series")
        null = np.asarray(null, dtype=float)
        vals = {n: np.asarray(raw[n], dtype=float) / null for n in names}
        return vals, "cost / null  (fraction of no-attack cost)", "normalised by null", False
    if normalize == "d":
        d_arr = np.asarray(ds, dtype=float)
        vals = {n: np.asarray(raw[n], dtype=float) / d_arr for n in names}
        return vals, _METRIC_YLABEL[metric] + " / d", "per dimension", True
    vals = {n: np.asarray(raw[n], dtype=float) for n in names}
    return vals, _METRIC_YLABEL[metric], "raw; scales with d", True


def cost_vs_dimension(env_cfg, method_cfg, ds, path, seeds=(0,),
                      eval_episodes=100, eval_seed=999, metric="cost", normalize=None):
    """Sweep the model dimension d and compare all policies -- does the method's
    advantage over greedy grow with d?

    Everything except d is held fixed. method and greedy use the same selector, so the
    only difference is the learned cost-to-go. Values are averaged over `seeds`. `metric`
    is 'cost' (mean episode cost) or 'gap' (mean ||theta_t - theta_target||). The right
    panel shows the scale-free advantage (greedy - method) / greedy.

    `normalize` rescales the LEFT panel for a fair cross-d comparison: None (raw), "d"
    (per dimension), or "null"/True (fraction of the no-attack cost). See
    `_normalize_dim_series`.
    """
    from dataclasses import replace
    from env import Environment
    from fvi import train_fvi
    from agents import RandomAgent, GreedyAgent, MethodAgent

    name = tier_name(method_cfg.select_mode)
    ds, names = list(ds), ["null", "random", "greedy", name]
    costs = {n: [] for n in names}
    for d in ds:
        ecfg = replace(env_cfg, d=int(d))
        per_seed = {n: [] for n in names}
        for s in seeds:
            torch.manual_seed(s)
            env = Environment(ecfg, method_cfg)
            V = train_fvi(env, method_cfg, seed=s, verbose=False, best_checkpoint=True)
            agents = {
                "null": NullAgent(),
                "random": RandomAgent(torch.Generator().manual_seed(s + 2),
                                      method_cfg.balance, method_cfg.k_max),
                "greedy": GreedyAgent(env.victim, env.theta_target, method_cfg,
                                      mode=method_cfg.select_mode),
                name: MethodAgent(V, env.victim, env.theta_target, method_cfg,
                                  torch.Generator().manual_seed(s + 1)),
            }
            for n, ag in agents.items():
                per_seed[n].append(_eval_metric(
                    env, ag, torch.Generator().manual_seed(eval_seed), eval_episodes, metric))
        for n in names:
            costs[n].append(float(np.mean(per_seed[n])))

    greedy, method = np.array(costs["greedy"]), np.array(costs[name])
    advantage = (greedy - method) / greedy * 100.0        # % reduction vs greedy

    # left panel: normalise for a fair cross-d comparison
    vals, ylab, sub, logy = _normalize_dim_series(
        costs, names, ds, metric, normalize, null=costs["null"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {"null": ("0.4", "--", "o"), "random": ("C3", ":", "^"),
              "greedy": ("C1", "-.", "s"), name: ("C0", "-", "o")}
    for n in names:
        c, ls, mk = styles[n]
        axL.plot(ds, vals[n], marker=mk, ls=ls, color=c, lw=2, label=n)
    if normalize in (True, "null"):
        axL.axhline(1.0, color="0.7", ls=":", lw=1)
    label = "Cost" if metric == "cost" else "Target gap ‖θ_t − θ̃‖"
    axL.set_xlabel("dimension d"); axL.set_ylabel(ylab)
    if logy:
        axL.set_yscale("log")
    axL.set_title(f"{label} vs d ({sub})")
    axL.legend(fontsize=9)

    axR.plot(ds, advantage, "o-", color="C0", lw=2)
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xlabel("dimension d")
    axR.set_ylabel(f"{name} advantage over greedy  (%)")
    axR.set_title(f"(greedy - {name}) / greedy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"ds": ds, **costs, "advantage_pct": advantage.tolist()}


def cost_vs_dimension_tiers(env_cfg, method_cfg, ds, path, seeds=(0,),
                            eval_episodes=100, eval_seed=999, train_mode="sort",
                            metric="cost", normalize=None):
    """Sweep d; for each d train ONE value function (under train_mode) and evaluate that
    SAME V_hat with each selection tier -- first-order / low-rank / exact. Isolates the
    value of using the interaction B (low-rank/exact) over first-order, as a function of d.
    Keys of the returned dict are the display names, not the select_mode ids.
    Averaged over `seeds`. `metric` is 'cost' or 'gap' (mean ||theta_t - target||).

    `normalize` (same options as cost_vs_dimension) rescales the LEFT panel for a fair
    cross-d comparison: None (raw), "d" (per dimension), or "null"/True (fraction of the
    no-attack cost; the null baseline is evaluated per d only when requested). The RIGHT
    panel (improvement over sort) is a ratio and so is unaffected by the choice.
    """
    from dataclasses import replace
    from env import Environment
    from fvi import train_fvi
    from agents import MethodAgent

    ds, tiers = list(ds), ["sort", "lowrank", "exact"]
    labels = [tier_name(t) for t in tiers]        # display names, also the dict keys
    costs = {tier_name(t): [] for t in tiers}
    need_null = normalize in (True, "null")
    null_costs = [] if need_null else None
    for d in ds:
        ecfg = replace(env_cfg, d=int(d))
        per_seed = {t: [] for t in tiers}          # keyed by select_mode id while computing
        null_seed = []
        for s in seeds:
            torch.manual_seed(s)
            env = Environment(ecfg, replace(method_cfg, select_mode=train_mode))
            V = train_fvi(env, replace(method_cfg, select_mode=train_mode),
                          seed=s, verbose=False, best_checkpoint=True)
            for t in tiers:                              # same V_hat, different selector
                agent = MethodAgent(V, env.victim, env.theta_target,
                                    replace(method_cfg, select_mode=t),
                                    torch.Generator().manual_seed(s + 1))
                per_seed[t].append(_eval_metric(env, agent,
                                   torch.Generator().manual_seed(eval_seed), eval_episodes, metric))
            if need_null:                                # no-attack reference for this d
                null_seed.append(_eval_metric(env, NullAgent(),
                                 torch.Generator().manual_seed(eval_seed), eval_episodes, metric))
        for t in tiers:
            costs[tier_name(t)].append(float(np.mean(per_seed[t])))
        if need_null:
            null_costs.append(float(np.mean(null_seed)))

    sort = np.array(costs[tier_name("sort")])
    vals, ylab, sub, logy = _normalize_dim_series(costs, labels, ds, metric, normalize,
                                                  null=null_costs)
    styles = {"first-order": ("C0", "o", "-"), "low-rank": ("C1", "s", "--"),
              "exact": ("C2", "^", "-.")}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    for t in labels:
        c, mk, ls = styles[t]
        axL.plot(ds, vals[t], marker=mk, ls=ls, color=c, lw=2, label=t)
    if need_null:
        axL.axhline(1.0, color="0.7", ls=":", lw=1)
    label = "Cost" if metric == "cost" else "Target gap ‖θ_t − θ̃‖"
    axL.set_xlabel("dimension d"); axL.set_ylabel(ylab)
    if logy:
        axL.set_yscale("log")
    axL.set_title(f"{label} by tier vs d ({sub}; shared V̂, trained w/ {tier_name(train_mode)})")
    axL.legend(fontsize=9)

    for t in ["low-rank", "exact"]:                      # ratio -> normalisation-invariant
        c, mk, ls = styles[t]
        axR.plot(ds, (sort - np.array(costs[t])) / sort * 100, marker=mk, ls=ls,
                 color=c, lw=2, label=t)
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xlabel("dimension d"); axR.set_ylabel("improvement over first-order (%)")
    axR.set_title("How much the interaction B buys over first-order selection")
    axR.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    result = {"ds": ds, **costs}
    if need_null:
        result["null"] = null_costs
    return result


def plot_duality_gap(env, V_psi, cfg, path, rng, n_states=24, dual_iters=25):
    """Low-rank dual-ascent duality gap vs iteration, over states the policy visits.

    For each visited state we build (alpha, B) from V_psi and trace the gap across the
    dual-ascent iterations (selection.lowrank_gap_trace). Plots each state's trace faint
    plus the mean. The gap is non-increasing and plateaus at the residual integrality
    gap of the (nonconvex) binary problem.
    """
    from decomposition import scores_and_interaction
    from selection import lowrank_gap_trace, select_action

    traces = []
    while len(traces) < n_states:
        theta = env.theta_0.clone()
        for t in range(env.T):
            z = env.stream.sample_batch(env.m, rng)
            alpha, B = scores_and_interaction(V_psi, theta, z, t, env.victim,
                                              env.theta_target, cfg.lam,
                                              need_B=True, use_hessian=cfg.use_hessian_in_B)
            traces.append(lowrank_gap_trace(alpha, B, c_edit=cfg.c_edit, k_max=cfg.k_max,
                                            balance=cfg.balance, r=cfg.rank_r,
                                            dual_iters=dual_iters))
            a = select_action(alpha, B, cfg.select_mode, k_max=cfg.k_max, c_edit=cfg.c_edit,
                              balance=cfg.balance, r=cfg.rank_r, dual_iters=cfg.dual_iters,
                              solver=cfg.solver)
            theta = env.victim.update(theta, z, a).detach()
            if len(traces) >= n_states:
                break

    traces = np.array(traces)                       # (n_states, dual_iters)
    iters = np.arange(1, traces.shape[1] + 1)
    mean = traces.mean(0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tr in traces:
        ax.plot(iters, tr, color="0.8", lw=0.8)
    ax.plot(iters, mean, "o-", color="C0", lw=2.2, label="mean gap")
    ax.set_xlabel("dual-ascent iteration")
    ax.set_ylabel("duality gap  (primal − dual bound)")
    ax.set_title(f"Low-rank selection: duality gap over iterations (r={cfg.rank_r})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"iters": iters.tolist(), "mean_gap": mean.tolist()}


def plot_weight_decay_sweep(env_cfg, method_cfg, wds, path, seeds=(0,),
                            eval_episodes=100, eval_seed=999, n_hess_states=20):
    """Sweep weight_decay; monitor curvature (||Hessian part of B|| / ||Gram part||) and
    performance (mean episode cost). Twin axes so the trade-off is visible on one plot.
    Averaged over `seeds`.
    """
    from dataclasses import replace
    from env import Environment
    from fvi import train_fvi, evaluate
    from agents import MethodAgent
    from decomposition import scores_and_interaction
    from selection import select_action

    wds = list(wds)
    hess_ratio, cost = [], []
    for wd in wds:
        cfg = replace(method_cfg, weight_decay=wd)
        rr, cc = [], []
        for s in seeds:
            torch.manual_seed(s)
            env = Environment(env_cfg, cfg)
            V = train_fvi(env, cfg, seed=s, verbose=False)
            cc.append(evaluate(env, MethodAgent(V, env.victim, env.theta_target, cfg,
                               torch.Generator().manual_seed(s + 1)),
                               torch.Generator().manual_seed(eval_seed), eval_episodes))
            rng, ratios, cnt = torch.Generator().manual_seed(eval_seed), [], 0
            while cnt < n_hess_states:
                theta = env.theta_0.clone()
                for t in range(env.T):
                    z = env.stream.sample_batch(env.m, rng)
                    alpha, B = scores_and_interaction(V, theta, z, t, env.victim,
                                                      env.theta_target, cfg.lam,
                                                      need_B=True, use_hessian=True)
                    _, Bg = scores_and_interaction(V, theta, z, t, env.victim,
                                                   env.theta_target, cfg.lam,
                                                   need_B=True, use_hessian=False)
                    ratios.append(float((B - Bg).norm() / Bg.norm()))
                    theta = env.victim.update(
                        theta, z, select_action(alpha, B, cfg.select_mode,
                                                c_edit=cfg.c_edit, balance=cfg.balance)).detach()
                    cnt += 1
                    if cnt >= n_hess_states:
                        break
            rr.append(float(np.mean(ratios)))
        hess_ratio.append(float(np.mean(rr)))
        cost.append(float(np.mean(cc)))

    labels = ["0" if w == 0 else f"{w:g}" for w in wds]
    x = list(range(len(wds)))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    c1 = "C3"
    ax1.plot(x, hess_ratio, "o-", color=c1, lw=2, label="curvature")
    ax1.axhline(1.0, color="0.7", ls=":", lw=1)          # Hessian == Gram reference
    ax1.set_xlabel("weight decay")
    ax1.set_ylabel("||Hessian part|| / ||Gram part||", color=c1)
    ax1.set_yscale("log"); ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    c2 = "C0"
    ax2.plot(x, cost, "s--", color=c2, lw=2, label="cost")
    ax2.set_ylabel("mean episode cost (lower = stronger attack)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    ax1.set_title("Weight decay: curvature (red) vs performance (blue)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"wds": wds, "hess_ratio": hess_ratio, "cost": cost}


def _episode_batches(env, rng):
    return [env.stream.sample_batch(env.m, rng) for _ in range(env.T)]


def _simulate(env, agent, batches):
    """Run a policy on a fixed batch sequence; return the slope trajectory theta_0..T."""
    thetas = [float(env.theta_0)]
    theta = env.theta_0.clone()
    for t, z in enumerate(batches):
        theta = env.victim.update(theta, z, agent.act(theta, z, t)).detach()
        thetas.append(float(theta))
    return thetas


def _simulate_with_actions(env, agent, batches):
    """Like _simulate but also return every point (x, y) and the action taken on it."""
    theta = env.theta_0.clone()
    thetas, xs, ys, acts = [float(theta)], [], [], []
    for t, z in enumerate(batches):
        a = agent.act(theta, z, t)                       # epsilon=0 -> deterministic
        xs.append(z[0].reshape(-1)); ys.append(z[1].reshape(-1)); acts.append(a)
        theta = env.victim.update(theta, z, a).detach()
        thetas.append(float(theta))
    return (thetas, torch.cat(xs).numpy(), torch.cat(ys).numpy(),
            torch.cat(acts).numpy())


def _simulate_vec(env, agent, batches):
    """Full parameter trajectory: list of (d,) theta tensors, theta_0 .. theta_T."""
    thetas, theta = [env.theta_0.clone()], env.theta_0.clone()
    for t, z in enumerate(batches):
        theta = env.victim.update(theta, z, agent.act(theta, z, t)).detach()
        thetas.append(theta.clone())
    return thetas


def plot_attack_paramspace(env, attack_agent, path, rng, clean_agent=None):
    """Visualize a d>1 attack (no single regression line exists).

    Left : the parameter trajectory projected onto the 2D plane spanned by theta_star
           and theta_target -- clean SGD heads to theta*, the attacked run is bent
           toward the target theta_tilde.
    Right: the final regression coefficients (theta*, target, clean, attacked) per index.
    """
    if clean_agent is None:
        clean_agent = NullAgent()
    batches = _episode_batches(env, rng)
    theta_clean = _simulate_vec(env, clean_agent, batches)
    theta_atk = _simulate_vec(env, attack_agent, batches)
    star, target = env.stream.theta_star, env.theta_target

    # orthonormal basis of span(theta*, theta_tilde): u1 along theta*, u2 the rest
    u1 = star / star.norm()
    tp = target - (target @ u1) * u1
    u2 = tp / tp.norm() if float(tp.norm()) > 1e-8 else torch.zeros_like(u1)
    def proj(th):
        return (float(th @ u1), float(th @ u2))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    pc = np.array([proj(th) for th in theta_clean])
    pa = np.array([proj(th) for th in theta_atk])
    axL.plot(pc[:, 0], pc[:, 1], "b.-", lw=2, label="clean SGD")
    axL.plot(pa[:, 0], pa[:, 1], "m.-", lw=2, label="attacked")
    axL.scatter(*proj(env.theta_0), c="k", s=60, zorder=5, label="start θ₀")
    axL.scatter(*proj(star), c="g", marker="*", s=280, zorder=5, label="θ* (true)")
    axL.scatter(*proj(target), c="r", marker="X", s=150, zorder=5, label="θ̃ (target)")
    axL.set_xlabel("component along θ*")
    axL.set_ylabel("orthogonal component (toward θ̃)")
    axL.set_title(f"Parameter path projected onto span(θ*, θ̃),  d={env.d}")
    axL.legend(fontsize=9)

    idx, w = np.arange(env.d), 0.2
    axR.bar(idx - 1.5 * w, star.numpy(), w, label="θ* (true)", color="g")
    axR.bar(idx - 0.5 * w, target.numpy(), w, label="θ̃ (target)", color="r")
    axR.bar(idx + 0.5 * w, theta_clean[-1].numpy(), w, label="clean final", color="b")
    axR.bar(idx + 1.5 * w, theta_atk[-1].numpy(), w, label="attacked final", color="m")
    axR.axhline(0, color="k", lw=0.6)
    axR.set_xlabel("coefficient index i"); axR.set_ylabel("θ_i")
    axR.set_title("Final regression coefficients")
    axR.set_xticks(idx); axR.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"clean_final": theta_clean[-1].tolist(), "attacked_final": theta_atk[-1].tolist(),
            "theta_star": star.tolist(), "theta_target": target.tolist()}


def plot_regression_attack_grid(env, agents, path, rng, ncols=2):
    """One panel per policy: the fitted line, plus every data point flagged by the
    action taken on it (delete = red x, duplicate = green dot, keep = grey dot).

    agents : {name: agent}. All policies are run on the SAME batch sequence.
    """
    assert env.d == 1, "regression-line plot only makes sense for d == 1"
    from matplotlib.lines import Line2D

    batches = _episode_batches(env, rng)
    star, target = float(env.stream.theta_star), float(env.theta_target)
    all_x = torch.cat([z[0].reshape(-1) for z in batches]).numpy()
    xline = np.linspace(all_x.min(), all_x.max(), 100)

    n = len(agents)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    results = {}
    for ax, (name, agent) in zip(axes.flat, agents.items()):
        thetas, xs, ys, acts = _simulate_with_actions(env, agent, batches)
        keep, dele, dup = acts == 1, acts == 0, acts == 2
        ax.scatter(xs[keep], ys[keep], s=16, color="0.6", alpha=0.5)
        ax.scatter(xs[dele], ys[dele], marker="x", s=55, color="red", linewidths=1.8)
        ax.scatter(xs[dup], ys[dup], marker="o", s=42, color="green")
        ax.plot(xline, star * xline, "g:", lw=1.5)
        ax.plot(xline, target * xline, "r--", lw=1.5)
        ax.plot(xline, thetas[-1] * xline, "b-", lw=2)
        ax.set_title(f"{name}:  θ_T={thetas[-1]:.2f}   "
                     f"({int(dele.sum())} del, {int(dup.sum())} dup)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        results[name] = {"theta_T": thetas[-1],
                         "n_delete": int(dele.sum()), "n_dup": int(dup.sum())}
    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    legend = [
        Line2D([0], [0], marker="o", ls="None", mfc="0.6", mec="0.6", label="keep"),
        Line2D([0], [0], marker="x", ls="None", color="red", label="delete"),
        Line2D([0], [0], marker="o", ls="None", color="green", label="duplicate"),
        Line2D([0], [0], color="g", ls=":", label=f"true slope θ*={star:.2f}"),
        Line2D([0], [0], color="r", ls="--", label=f"target θ̃={target:.2f}"),
        Line2D([0], [0], color="b", label="fitted line θ_T"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("d=1 attack per policy — points flagged by the attacker's action")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return results


def plot_regression_attack(env, attack_agent, path, rng, clean_agent=None):
    """Clean (keep-all SGD) vs attacked regression line on the SAME data stream.

    Left  : data + final lines (true slope, target, clean, attacked).
    Right : the attacked line rotating from t=0 to t=T (colour = time).
    """
    assert env.d == 1, "regression-line plot only makes sense for d == 1"
    if clean_agent is None:
        clean_agent = NullAgent()

    batches = _episode_batches(env, rng)                 # same batches for both policies
    theta_clean = _simulate(env, clean_agent, batches)
    theta_atk = _simulate(env, attack_agent, batches)

    xs = torch.cat([z[0].reshape(-1) for z in batches]).numpy()
    ys = torch.cat([z[1].reshape(-1) for z in batches]).numpy()
    xline = np.linspace(xs.min(), xs.max(), 100)
    star, target = float(env.stream.theta_star), float(env.theta_target)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: final lines ---
    axL.scatter(xs, ys, s=14, alpha=0.25, color="gray", label="data")
    axL.plot(xline, star * xline, "g:", lw=2, label=f"true slope θ*={star:.2f}")
    axL.plot(xline, target * xline, "r--", lw=2, label=f"attacker target θ̃={target:.2f}")
    axL.plot(xline, theta_clean[-1] * xline, "b-", lw=2.5,
             label=f"clean SGD (θ_T={theta_clean[-1]:.2f})")
    axL.plot(xline, theta_atk[-1] * xline, "m-", lw=2.5,
             label=f"attacked (θ_T={theta_atk[-1]:.2f})")
    axL.set_xlabel("x"); axL.set_ylabel("y")
    axL.set_title("Regression line: clean vs attacked")
    axL.legend(loc="best", fontsize=9)

    # --- right: how the slope evolves over the episode ---
    steps = list(range(len(theta_atk)))
    axR.plot(steps, theta_clean, "b.-", lw=2, label="clean SGD")
    axR.plot(steps, theta_atk, "m.-", lw=2, label="attacked")
    axR.axhline(star, color="g", ls=":", lw=2, label=f"true slope θ*={star:.2f}")
    axR.axhline(target, color="r", ls="--", lw=2, label=f"target θ̃={target:.2f}")
    axR.set_xlabel("training step t"); axR.set_ylabel("slope θ_t")
    axR.set_title("How the slope changes over the attack")
    axR.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"theta_star": star, "theta_target": target,
            "clean_traj": theta_clean, "attacked_traj": theta_atk}
