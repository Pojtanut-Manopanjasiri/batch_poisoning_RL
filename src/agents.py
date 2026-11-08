import torch

from decomposition import scores_and_interaction
from env import random_feasible_action
from selection import _feasible_set, count_feasible, select_action


class NullAgent:
    def act(self, theta, z, t, epsilon=0.0):
        return torch.ones(z[0].shape[0], device=z[0].device)


class RandomAgent:
    def __init__(self, rng, balance=True, B=None, L=2):
        self.rng, self.balance, self.B, self.L = rng, balance, B, L

    def act(self, theta, z, t, epsilon=0.0):
        return random_feasible_action(z[0].shape[0], self.rng, self.balance, self.B, self.L,
                                      device=z[0].device)


class GreedyAgent:
    """Myopic optimum of the immediate cost g (equivalently the decomposition with
    V_hat == 0: alpha_i = <e,phi_i>, Gamma = Phi Phi^T)."""

    def __init__(self, victim, theta_target, cfg, mode="greedy"):
        self.victim, self.theta_target, self.cfg, self.mode = victim, theta_target, cfg, mode

    def act(self, theta, z, t, epsilon=0.0):
        Phi = self.victim.phi_batch(theta, z)
        alpha = Phi @ (theta - self.theta_target)
        Gamma = Phi @ Phi.T
        return select_action(alpha, Gamma, self.mode, B=self.cfg.B,
                             lam=self.cfg.lam, balance=self.cfg.balance,
                             L=self.cfg.L, solver=self.cfg.solver)

class MethodAgent:
    """Value-based policy (FEVI / R-FEVI): build alpha, Gamma from V_hat, then select an action."""

    def __init__(self, V_psi, victim, theta_target, cfg, rng):
        self.V_psi, self.victim = V_psi, victim
        self.theta_target, self.cfg, self.rng = theta_target, cfg, rng

    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.B,
                                          self.cfg.L, device=z[0].device)

        need_Gamma = self.cfg.select_mode != "greedy"
        alpha, Gamma = scores_and_interaction(
            self.V_psi, theta, z, t, self.victim, self.theta_target, need_Gamma=need_Gamma)
        return select_action(alpha, Gamma, self.cfg.select_mode, B=self.cfg.B,
                             lam=self.cfg.lam, balance=self.cfg.balance,
                             L=self.cfg.L, solver=self.cfg.solver)
