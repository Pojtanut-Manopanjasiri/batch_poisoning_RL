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
    V_hat == 0: alpha_i = lam<e,phi_i>, Gamma = lam Phi Phi^T)."""

    def __init__(self, victim, theta_target, cfg, mode="greedy"):
        self.victim, self.theta_target, self.cfg, self.mode = victim, theta_target, cfg, mode

    def act(self, theta, z, t, epsilon=0.0):
        Phi = self.victim.phi_batch(theta, z)
        alpha = self.cfg.lam * (Phi @ (theta - self.theta_target))
        Gamma = self.cfg.lam * (Phi @ Phi.T)
        return select_action(alpha, Gamma, self.mode, B=self.cfg.B,
                             c_edit=self.cfg.c_edit, balance=self.cfg.balance,
                             L=self.cfg.L, solver=self.cfg.solver)


def clairvoyant_search_size(m, T, cfg):
    """(|A| per step, worst-case search nodes per episode) for ClairvoyantAgent."""
    n_actions = count_feasible(m, cfg.balance, cfg.B, cfg.L)
    return n_actions, float(n_actions) ** T


class ClairvoyantAgent:
    """Oracle: solves the T-stage problem to exact optimality given the full batch
    stream, via branch-and-bound DFS over the joint action sequence. Call `plan`
    once per episode before `act`, which just replays the resulting sequence.
    """

    def __init__(self, victim, theta_target, cfg):
        self.victim, self.theta_target, self.cfg = victim, theta_target, cfg
        self._actions, self._i = None, 0

    def _step_costs(self, theta, z, A, target):
        Phi = self.victim.phi_batch(theta, z)                    # (m, d)
        theta_next = theta.unsqueeze(0) + A @ Phi                # (N, d)
        edits = torch.abs(A - 1.0).sum(dim=1)
        nef = 0.5 * self.cfg.lam * ((theta_next - target) ** 2).sum(dim=1)
        return self.cfg.c_edit * edits + nef, theta_next

    def plan(self, theta_0, zs):
        # searched on CPU: every node reads its costs back to order and prune children
        cfg = self.cfg
        device = theta_0.device
        theta_0, target = theta_0.cpu(), self.theta_target.cpu()
        zs = [(x.cpu(), y.cpu()) for x, y in zs]
        m = zs[0][0].shape[0]
        A = _feasible_set(m, cfg.balance, cfg.B, cfg.L)

        best_cost, best_actions = float("inf"), None
        actions = [None] * len(zs)

        def rec(t, theta, cost_so_far):
            nonlocal best_cost, best_actions
            if cost_so_far >= best_cost:
                return
            if t == len(zs):
                best_cost, best_actions = cost_so_far, actions.copy()
                return
            step_cost, theta_next = self._step_costs(theta, zs[t], A, target)
            for i in torch.argsort(step_cost).tolist():
                actions[t] = A[i]
                rec(t + 1, theta_next[i].detach(), cost_so_far + float(step_cost[i]))

        rec(0, theta_0, 0.0)
        self._actions, self._i = [a.to(device) for a in best_actions], 0
        return best_cost

    def act(self, theta, z, t, epsilon=0.0):
        a = self._actions[self._i]
        self._i += 1
        return a


class MethodAgent:
    """FVI policy: build alpha, B from the learned V_hat, then select an action."""

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
            self.V_psi, theta, z, t, self.victim, self.theta_target, self.cfg.lam,
            need_Gamma=need_Gamma, use_hessian=self.cfg.use_hessian_in_B)
        return select_action(alpha, Gamma, self.cfg.select_mode, B=self.cfg.B,
                             c_edit=self.cfg.c_edit, balance=self.cfg.balance,
                             L=self.cfg.L, solver=self.cfg.solver)
