"""Empirical check of Theorem 1: the surrogate action's one-step regret is bounded.

At a decision point (theta, Z, t) with a fixed fitted h_{t+1}, with

    F_t(a)  = (lam/2)||f(theta,Z,a) - theta_dag||^2 + c_edit||a-1||_1 + h_{t+1}(f(...))
    F~_t(a) = (lam/2)||theta - theta_dag||^2 + h_{t+1}(theta) + sum_i a_i alpha_i
              + c_edit||a-1||_1

and a_hat = argmin F~_t (Algorithm 2), a_star = argmin F_t (brute force over A), the
theorem claims  0 <= F_t(a_hat) - F_t(a_star) <= (lam + M_{t+1}) R^2 min{1, 2B/m},
where R = eta_t G and M_{t+1} is the Lipschitz constant of grad h_{t+1} on
{x : ||x - theta|| <= R}.

    python src/verify_theorem1.py --learner zero      # h == 0: M = 0 exactly, rigorous
    python src/verify_theorem1.py --learner fevi --seeds 5 --episodes 4

Unit note: phi_batch already folds eta/m into Phi, so R = m * max_i ||Phi_i|| and the
paper's (1 + M) becomes (lam + M) because the code weights the nefarious term by lam.
M is estimated by sampling Hessians inside the ball, which can only UNDERSTATE the true
supremum -- so --learner zero, where no estimation is needed, is the decisive check.
"""
import argparse
import copy
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime

import torch
import torch.nn as nn

from agents import MethodAgent
from config import EnvConfig, MethodConfig
from decomposition import scores_and_interaction
from env import Environment
from fvi import train_fevi, train_fvi
from selection import _feasible_set, select_action

CSV_FIELDS = [
    "seed", "episode", "t", "gap", "bound", "ratio", "holds",
    "G", "radius", "max_u_norm", "M_hess", "lipschitz_paper", "lipschitz_tight",
    "lipschitz_effective", "min_factor", "n_actions",
    "F_ahat", "F_astar", "Ftilde_ahat", "Ftilde_min", "surrogate_gap",
    "r_ahat", "r_astar", "u_ahat_norm", "u_astar_norm", "ahat_is_astar",
]


class ZeroValue(nn.Module):
    """h == 0 with the autograd graph intact, so grad/Hessian are exactly zero."""

    def forward(self, theta, t):
        return (theta.sum(-1) * 0.0).reshape(theta.shape[:-1])


def _ball_points(theta, radius, n, gen):
    d = theta.numel()
    v = torch.randn(n, d, generator=gen, dtype=theta.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.rand(n, 1, generator=gen) ** (1.0 / d)       # uniform in the ball
    return theta[None] + (radius * scale * v).to(theta.device)


def _hessian_spectral(V, x, t, lam):
    """(||Hess h||, ||lam I + Hess h||) at x, both spectral norms."""
    H = torch.autograd.functional.hessian(lambda z: V(z, t).reshape(()), x)
    H = 0.5 * (H + H.T)
    eig_h = torch.linalg.eigvalsh(H)
    eig_psi = eig_h + lam
    return float(eig_h.abs().max()), float(eig_psi.abs().max())


def analyze_point(V, env, cfg, theta, z, t, gen, n_hess):
    # float64 throughout: r = F - F~ cancels two O(1) numbers down to O(||u||^2), which
    # float32 cannot resolve when ||u|| is small
    theta = theta.double()
    z = (z[0].double(), z[1].double())
    target = env.theta_target.double()
    Phi = env.victim.phi_batch(theta, z)                       # (m, d), includes eta/m
    m = Phi.shape[0]
    A = _feasible_set(m, cfg.balance, cfg.B, cfg.L).to(Phi.device).double()   # (N, m)

    alpha, _ = scores_and_interaction(V, theta, z, t, env.victim, target, cfg.lam,
                                      need_Gamma=False)
    U = A @ Phi                                                # (N, d) = u(a)
    theta_next = theta[None] + U
    edits = (A - 1.0).abs().sum(1)
    with torch.no_grad():
        h_next = V(theta_next, t + 1).reshape(-1)
        h_theta = float(V(theta, t + 1))

    F = 0.5 * cfg.lam * ((theta_next - target) ** 2).sum(1) + cfg.c_edit * edits + h_next
    const = 0.5 * cfg.lam * float(((theta - target) ** 2).sum()) + h_theta
    F_tilde = A @ alpha + cfg.c_edit * edits + const
    r = F - F_tilde

    a_hat = select_action(alpha, None, "greedy", B=cfg.B, c_edit=cfg.c_edit,
                          balance=cfg.balance, L=cfg.L).double()
    i_hat = int(((A - a_hat[None]).abs().sum(1) < 1e-9).nonzero()[0])
    i_star, i_tilde = int(F.argmin()), int(F_tilde.argmin())

    G = float(Phi.norm(dim=1).max())
    radius = m * G                                             # = eta_t * G in paper units

    probes = torch.cat([theta[None], theta_next[i_hat][None], theta_next[i_star][None],
                        _ball_points(theta, radius, n_hess, gen)])
    spectra = [_hessian_spectral(V, x, t + 1, cfg.lam) for x in probes]
    m_hess = max(s[0] for s in spectra)
    lip_tight = max(s[1] for s in spectra)
    lip_paper = cfg.lam + m_hess

    u_norm = U.norm(dim=1)
    nz = u_norm > 1e-12
    lip_eff = float((2 * r[nz].abs() / u_norm[nz] ** 2).max()) if bool(nz.any()) else 0.0

    min_factor = 1.0 if cfg.B is None else min(1.0, 2.0 * cfg.B / m)
    gap = float(F[i_hat] - F[i_star])
    bound = lip_paper * radius ** 2 * min_factor

    return {
        "t": t, "gap": gap, "bound": bound,
        "ratio": gap / bound if bound > 0 else 0.0,
        "holds": bool(gap <= bound + 1e-9 and gap >= -1e-9),
        "G": G, "radius": radius, "max_u_norm": float(u_norm.max()),
        "M_hess": m_hess, "lipschitz_paper": lip_paper, "lipschitz_tight": lip_tight,
        "lipschitz_effective": lip_eff, "min_factor": min_factor,
        "n_actions": int(A.shape[0]),
        "F_ahat": float(F[i_hat]), "F_astar": float(F[i_star]),
        "Ftilde_ahat": float(F_tilde[i_hat]), "Ftilde_min": float(F_tilde[i_tilde]),
        "surrogate_gap": float(F_tilde[i_hat] - F_tilde[i_tilde]),
        "r_ahat": float(r[i_hat]), "r_astar": float(r[i_star]),
        "u_ahat_norm": float(u_norm[i_hat]), "u_astar_norm": float(u_norm[i_star]),
        "ahat_is_astar": bool(i_hat == i_star),
    }


def collect(V, env, cfg, seed, episodes, n_hess):
    """Roll the learned policy out and analyse every decision point it visits."""
    rng = torch.Generator().manual_seed(seed + 7000)
    gen = torch.Generator().manual_seed(seed + 8000)
    agent = MethodAgent(V, env.victim, env.theta_target, cfg,
                        torch.Generator().manual_seed(seed))
    V_d = copy.deepcopy(V).double()          # analysis copy; the rollout stays float32
    rows = []
    for ep in range(episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            z = env.stream.sample_batch(env.m, rng)
            row = analyze_point(V_d, env, cfg, theta, z, t, gen, n_hess)
            rows.append({"seed": seed, "episode": ep, **row})
            theta = env.victim.update(theta, z, agent.act(theta, z, t)).detach()
    return rows


def summarize(rows):
    gaps = [r["gap"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    sur = [r["surrogate_gap"] for r in rows]
    by_t = {}
    for r in rows:
        by_t.setdefault(r["t"], []).append(r["ratio"])
    return {
        "n_points": len(rows),
        "bound_violations": sum(not r["holds"] for r in rows),
        "negative_gaps": sum(g < -1e-9 for g in gaps),
        "surrogate_suboptimal": sum(s > 1e-9 for s in sur),
        "max_surrogate_gap": max(sur),
        "ahat_equals_astar_frac": sum(r["ahat_is_astar"] for r in rows) / len(rows),
        "gap_mean": sum(gaps) / len(gaps), "gap_max": max(gaps),
        "ratio_mean": sum(ratios) / len(ratios), "ratio_max": max(ratios),
        "ratio_max_by_stage": {str(t): max(v) for t, v in sorted(by_t.items())},
        "radius_containment_ok": all(r["max_u_norm"] <= r["radius"] + 1e-6 for r in rows),
        "lipschitz_estimate_ok": all(r["lipschitz_effective"] <= r["lipschitz_paper"] + 1e-6
                                     for r in rows),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--learner", default="zero", choices=["zero", "fvi", "fevi"],
                   help="zero = h == 0 (M is exactly 0, no estimation)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--n-hess", type=int, default=8,
                   help="Hessian probes inside the ball, per decision point")
    p.add_argument("--max-actions", type=int, default=20000)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--outdir", default="output")
    p.add_argument("--task", default="regression",
                   choices=["regression", "logistic", "mlp"])
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--c-edit", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--B", type=int)
    p.add_argument("--K", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg_kw = dict(task=args.task, d=args.d, m=args.m, T=args.T, eta=args.eta,
                      device=args.device)
    cfg = MethodConfig(c_edit=args.c_edit, lam=args.lam, L=args.L, B=args.B,
                       K_iters=args.K, select_mode="greedy")

    n_actions = _feasible_set(args.m, cfg.balance, cfg.B, cfg.L).shape[0]
    if n_actions > args.max_actions:
        raise SystemExit(f"|A| = {n_actions} exceeds --max-actions {args.max_actions}; "
                         "a_star needs full enumeration, so lower m or L")
    print(f"learner={args.learner}  |A|={n_actions}  m={args.m} T={args.T} d={args.d} "
          f"lam={cfg.lam} c_edit={cfg.c_edit} B={cfg.B} L={cfg.L}")

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        env = Environment(EnvConfig(seed=seed, **env_cfg_kw), cfg)
        if args.learner == "zero":
            V = ZeroValue().to(env.device)
        else:
            train = train_fvi if args.learner == "fvi" else train_fevi
            V = train(env, cfg, seed=seed, verbose=False, return_history=False,
                      best_checkpoint=False)
            torch.save(V.state_dict(),
                       os.path.join(args.outdir, f"theorem1_{args.learner}_seed{seed}.pt"))
        seed_rows = collect(V, env, cfg, seed, args.episodes, args.n_hess)
        rows.extend(seed_rows)
        bad = sum(not r["holds"] for r in seed_rows)
        print(f"  seed {seed}: {len(seed_rows)} points, {bad} violation(s), "
              f"max gap/bound = {max(r['ratio'] for r in seed_rows):.4f}")

    summary = summarize(rows)
    print("\n== Theorem 1 ==")
    print(f"  points                  : {summary['n_points']}")
    print(f"  bound violations        : {summary['bound_violations']}")
    print(f"  negative gaps           : {summary['negative_gaps']}")
    print(f"  surrogate suboptimal    : {summary['surrogate_suboptimal']} "
          f"(max {summary['max_surrogate_gap']:.2e})")
    print(f"  a_hat == a_star         : {summary['ahat_equals_astar_frac']:.1%}")
    print(f"  gap  mean / max         : {summary['gap_mean']:.6f} / {summary['gap_max']:.6f}")
    print(f"  gap/bound mean / max    : {summary['ratio_mean']:.4f} / {summary['ratio_max']:.4f}")
    print(f"  gap/bound max by stage  : {summary['ratio_max_by_stage']}")
    print(f"  ||u|| <= radius         : {summary['radius_containment_ok']}")
    print(f"  effective <= assumed L  : {summary['lipschitz_estimate_ok']}")

    csv_path = os.path.join(args.outdir, f"theorem1_{args.learner}_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(args.outdir, f"theorem1_{args.learner}_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({"learner": args.learner, "env_config": env_cfg_kw,
                   "method_config": asdict(cfg), "summary": summary}, f, indent=2)
    print(f"\nper-point rows -> {csv_path}\nsummary        -> {json_path}")


if __name__ == "__main__":
    main()
