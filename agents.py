"""Agents.

Baselines (reconstructed Brief I):
  - NullAgent      : keep everything (a = 1). No edits, no attack.
  - RandomAgent    : a uniform-ish feasible action.
  - GreedyAgent    : myopic -- minimise the one-step cost g only (V_hat = 0).

The method (Brief II):
  - MethodAgent    : scores_and_interaction (with the learned V_hat) + select_action.

Every agent exposes  act(theta, z, t, epsilon=0.0) -> a in {0,1,2}^m.
"""
import torch

from decomposition import scores_and_interaction
from env import random_feasible_action
from selection import select_action


class NullAgent:
    def act(self, theta, z, t, epsilon=0.0):
        return torch.ones(z[0].shape[0])


class RandomAgent:
    def __init__(self, rng, balance=True, k_max=None):
        self.rng, self.balance, self.k_max = rng, balance, k_max

    def act(self, theta, z, t, epsilon=0.0):
        return random_feasible_action(z[0].shape[0], self.rng, self.balance, self.k_max)


class GreedyAgent:
    """Myopic optimum of the immediate cost g (equivalently the decomposition with
    V_hat == 0: alpha_i = lam<e,phi_i>, B = lam Phi Phi^T)."""

    def __init__(self, victim, theta_target, cfg, mode="sort"):
        self.victim, self.theta_target, self.cfg, self.mode = victim, theta_target, cfg, mode

    def act(self, theta, z, t, epsilon=0.0):
        Phi = self.victim.phi_batch(theta, z)
        alpha = self.cfg.lam * (Phi @ (theta - self.theta_target))
        B = self.cfg.lam * (Phi @ Phi.T)
        return select_action(alpha, B, self.mode, k_max=self.cfg.k_max,
                             c_edit=self.cfg.c_edit, balance=self.cfg.balance,
                             r=self.cfg.rank_r, dual_iters=self.cfg.dual_iters,
                             solver=self.cfg.solver)


class MethodAgent:
    """FVI policy: build alpha, B from the learned V_hat, then select an action."""

    def __init__(self, V_psi, victim, theta_target, cfg, rng):
        self.V_psi, self.victim = V_psi, victim
        self.theta_target, self.cfg, self.rng = theta_target, cfg, rng

    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.k_max)

        need_B = self.cfg.select_mode != "sort"
        alpha, B = scores_and_interaction(
            self.V_psi, theta, z, t, self.victim, self.theta_target, self.cfg.lam,
            need_B=need_B, use_hessian=self.cfg.use_hessian_in_B)
        return select_action(alpha, B, self.cfg.select_mode, k_max=self.cfg.k_max,
                             c_edit=self.cfg.c_edit, balance=self.cfg.balance,
                             r=self.cfg.rank_r, dual_iters=self.cfg.dual_iters,
                             solver=self.cfg.solver)
