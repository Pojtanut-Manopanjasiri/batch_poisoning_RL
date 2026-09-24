"""Diagnostics -- the scientific payload  (Brief II, sec. 5).

These measurements turn "we assume separability" into "we quantify what
separability costs".  Three required figures:

  1. spectrum of Gamma (log scale, averaged) + cumulative trace fraction.
  2. relative interaction |1/2 a^T Gamma a| / |a^T alpha| vs eta and vs k.
  3. selection quality vs tier: cost of greedy / exact + wall-clock.
"""
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents import MethodAgent
from config import tier_name
from decomposition import scores_and_interaction
from env import Environment
from fvi import train_fvi
from selection import _objective, select_action


# --------------------------------------------------------------------------- #
# per-step diagnostics
# --------------------------------------------------------------------------- #
def interaction_diagnostics(alpha, Gamma, a_sort, a_exact, Phi, lam, c_edit):
    lin = float(a_sort @ alpha)                       # linear-term magnitude
    pair = float(0.5 * a_sort @ Gamma @ a_sort)           # neglected interaction
    eig = torch.linalg.eigvalsh(0.5 * (Gamma + Gamma.T)).cpu().numpy()
    eig_desc = np.sort(np.abs(eig))[::-1]
    gram = lam * (Phi @ Phi.T)
    return {
        "rel_interaction": abs(pair) / (abs(lin) + 1e-12),
        "spectrum": eig_desc,
        "eff_rank": float(np.trace(Gamma.cpu().numpy()) / (np.abs(eig).max() + 1e-12)),
        "trace_frac_top1": float(eig_desc[0] / eig_desc.sum()),
        "trace_frac_top5": float(eig_desc[:5].sum() / eig_desc.sum()),
        "gram_only_pair": float(0.5 * a_sort @ gram @ a_sort),
        "action_agreement": float((a_sort == a_exact).float().mean()),
        "suboptimality": (_objective(a_sort, alpha, Gamma, c_edit)
                          - _objective(a_exact, alpha, Gamma, c_edit)),  # >= 0
    }


def _states_alpha_B(env, agent, V_psi, cfg, rng, n_episodes):
    """Roll out the policy and, at every step, recompute (alpha, Gamma, Phi)."""
    records = []
    for _ in range(n_episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            z = env.stream.sample_batch(env.m, rng)
            Phi = env.victim.phi_batch(theta, z)
            alpha, Gamma = scores_and_interaction(
                V_psi, theta, z, t, env.victim, env.theta_target, cfg.lam,
                need_Gamma=True, use_hessian=cfg.use_hessian_in_B)
            records.append((theta.cpu(), t, tuple(v.cpu() for v in z),
                            alpha.cpu(), Gamma.cpu(), Phi.cpu()))
            a = agent.act(theta, z, t)
            theta = env.victim.update(theta, z, a).detach()
    return records


# --------------------------------------------------------------------------- #
# figure 1: spectrum of Gamma
# --------------------------------------------------------------------------- #
def figure_spectrum(records, path, cfg):
    spectra = np.stack([interaction_diagnostics(
        a, Gamma, torch.ones(a.shape[0]), torch.ones(a.shape[0]), Phi, cfg.lam, cfg.c_edit
    )["spectrum"] for (_, _, _, a, Gamma, Phi) in records])
    mean_spec = spectra.mean(0)
    cum_frac = np.cumsum(mean_spec) / mean_spec.sum()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.semilogy(range(1, len(mean_spec) + 1), mean_spec, "o-", color="C0")
    ax1.set_xlabel("index k")
    ax1.set_ylabel("|eigenvalue|  (log)", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum_frac) + 1), cum_frac, "s--", color="C1")
    ax2.set_ylabel("cumulative trace fraction", color="C1")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Spectrum of Gamma (averaged over batches)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 2: relative interaction vs eta and vs k
# --------------------------------------------------------------------------- #
def _mean_rel_interaction(env, cfg, rng, n_episodes, Gamma=None):
    """Uses the myopic (V_hat == 0) decomposition so no retraining is needed."""
    vals = []
    for _ in range(n_episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            z = env.stream.sample_batch(env.m, rng)
            Phi = env.victim.phi_batch(theta, z)
            alpha = cfg.lam * (Phi @ (theta - env.theta_target))
            Gamma = cfg.lam * (Phi @ Phi.T)
            a = select_action(alpha, Gamma, "greedy", B=cfg.B, c_edit=cfg.c_edit,
                              balance=cfg.balance, L=cfg.L)
            lin = float(a @ alpha)
            pair = float(0.5 * a @ Gamma @ a)
            vals.append(abs(pair) / (abs(lin) + 1e-12))
            theta = env.victim.update(theta, z, a).detach()
    return float(np.mean(vals))


def figure_rel_interaction(env_cfg, cfg, path, etas=(0.05, 0.1, 0.2, 0.4, 0.8),
                           ks=None, n_episodes=30):
    if ks is None:
        ks = list(range(1, env_cfg.m // 2 + 1))
    rng = torch.Generator().manual_seed(1234)

    rel_eta = []
    for eta in etas:
        env = Environment(replace(env_cfg, eta=eta), cfg)
        rel_eta.append(_mean_rel_interaction(env, cfg, rng, n_episodes))

    env = Environment(env_cfg, cfg)
    rel_k = [_mean_rel_interaction(env, cfg, rng, n_episodes, Gamma=k) for k in ks]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    axL.loglog(etas, rel_eta, "o-")
    axL.loglog(etas, [rel_eta[0] * (e / etas[0]) for e in etas], "k--",
               label="slope 1 (~eta)")
    axL.set_xlabel("eta"); axL.set_ylabel("relative interaction")
    axL.set_title("rel. interaction vs eta"); axL.legend()

    axR.plot(ks, rel_k, "s-")
    axR.set_xlabel("k (budget on deletes)"); axR.set_ylabel("relative interaction")
    axR.set_title("rel. interaction vs k")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 3: selection quality vs tier
# --------------------------------------------------------------------------- #
def figure_tier_comparison(records, path, cfg):
    tiers = ["greedy", "exact"]
    costs = {t: [] for t in tiers}
    times = {t: [] for t in tiers}
    for (_, _, _, alpha, Gamma, _) in records:
        for tier in tiers:
            t0 = time.perf_counter()
            a = select_action(alpha, Gamma, tier, B=cfg.B, c_edit=cfg.c_edit,
                              balance=cfg.balance, L=cfg.L, solver=cfg.solver)
            times[tier].append(time.perf_counter() - t0)
            costs[tier].append(_objective(a, alpha, Gamma, cfg.c_edit))  # full-Gamma cost

    labels = [tier_name(t) for t in tiers]        # display names, also the dict keys
    mean_cost = [np.mean(costs[t]) for t in tiers]
    mean_time = [np.mean(times[t]) for t in tiers]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    axL.bar(labels, mean_cost, color=["#2a78d6", "#e87ba4"])
    axL.set_ylabel("mean selected cost Q(a)")
    axL.set_title("selection quality (lower is better)")
    axR.bar(labels, mean_time, color=["#2a78d6", "#e87ba4"])
    axR.set_ylabel("mean wall-clock (s)")
    axR.set_yscale("log")
    axR.set_title("selection cost (wall-clock)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"cost": dict(zip(labels, mean_cost)), "time": dict(zip(labels, mean_time))}


# --------------------------------------------------------------------------- #
# run everything
# --------------------------------------------------------------------------- #
def run_all_diagnostics(env, cfg, V_psi, outdir=".", n_episodes=30, seed=7):
    rng = torch.Generator().manual_seed(seed)
    agent = MethodAgent(V_psi, env.victim, env.theta_target, cfg, rng)
    records = _states_alpha_B(env, agent, V_psi, cfg, rng, n_episodes)

    figure_spectrum(records, f"{outdir}/fig1_spectrum.png", cfg)
    figure_rel_interaction(env._env_cfg, cfg, f"{outdir}/fig2_rel_interaction.png")
    summary = figure_tier_comparison(records, f"{outdir}/fig3_tiers.png", cfg)
    return summary
