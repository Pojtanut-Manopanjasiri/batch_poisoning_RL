"""Fitted value iteration over V_bar_t(theta)  (Brief II, sec. 4.1).

Outer loop: collect rollouts under the current greedy policy (+ epsilon exploration),
then fit V_hat on Bellman targets against a frozen label network (DQN-style).

The decomposition is used for action *selection* only, never for bootstrapping:
TD targets use V_hat(theta_{t+1}) evaluated directly.
"""
import copy

import torch

from agents import MethodAgent
from value_net import ValueNet


class ReplayBuffer:
    """FIFO replay of transitions (theta, t, cost, theta_next).

    Accumulates across FVI iterations so the value fit sees data from a mixture of
    past policies -- this is what stabilises the TD bootstrap (broad coverage instead
    of collapsing onto the current policy's narrow trajectory as exploration decays).
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.theta = self.t = self.c = self.theta_next = None

    def push(self, transitions):
        theta = torch.stack([tr[0] for tr in transitions])
        t = torch.tensor([tr[1] for tr in transitions])
        c = torch.tensor([tr[2] for tr in transitions], dtype=torch.float32)
        theta_next = torch.stack([tr[3] for tr in transitions])
        if self.theta is None:
            self.theta, self.t, self.c, self.theta_next = theta, t, c, theta_next
        else:
            self.theta = torch.cat([self.theta, theta])
            self.t = torch.cat([self.t, t])
            self.c = torch.cat([self.c, c])
            self.theta_next = torch.cat([self.theta_next, theta_next])
        if len(self) > self.capacity:                    # drop oldest (FIFO)
            k = len(self) - self.capacity
            self.theta, self.t = self.theta[k:], self.t[k:]
            self.c, self.theta_next = self.c[k:], self.theta_next[k:]

    def sample(self, batch_size, rng):
        idx = torch.randint(0, len(self), (min(batch_size, len(self)),), generator=rng)
        return self.theta[idx], self.t[idx], self.c[idx], self.theta_next[idx]

    def __len__(self):
        return 0 if self.theta is None else len(self.theta)


def eps_schedule(k, K, e0, e1):
    return e0 + (e1 - e0) * (k / max(1, K - 1))


def rollout(env, agent, rng, epsilon=0.0):
    """One episode. Returns (transitions, total_cost).

    transition = (theta_t, t, cost_t, theta_{t+1}).
    """
    transitions, total = [], 0.0
    theta = env.theta_0.clone()
    for t in range(env.T):
        z = env.stream.sample_batch(env.m, rng)
        a = agent.act(theta, z, t, epsilon=epsilon)
        theta_next = env.victim.update(theta, z, a)
        c = env.cost(theta, z, a, theta_next)
        transitions.append((theta.detach(), t, c, theta_next.detach()))
        total += c
        theta = theta_next.detach()
    return transitions, total


def evaluate(env, agent, rng, n_episodes=100):
    """Mean episode cost of a policy (the objective we minimise)."""
    return sum(rollout(env, agent, rng)[1] for _ in range(n_episodes)) / n_episodes


def train_fvi(env, cfg, seed=0, verbose=True, return_history=False,
              best_checkpoint=False, ema_decay=None, eval_episodes=100, eval_seed=999):
    """Train V_hat by FVI.

    If return_history, also evaluate the method policy on a FIXED held-out set of
    episodes after every iteration and return (V_psi, history) where history is the
    list of per-iteration mean episode costs. (eval_seed is re-seeded each eval so
    all iterations are scored on the same batches -- a clean training curve.)

    If best_checkpoint, return the V_hat from the iteration with the lowest held-out
    cost (early stopping) rather than the last one -- guards against the slow drift
    that FVI can still show over long runs.

    If ema_decay is not None (e.g. 0.99), maintain a Polyak / exponential-moving-average
    of the weights across gradient steps and RETURN that averaged network. Averaging the
    SGD iterates lowers the variance of the estimate and smooths V_hat (and its curvature),
    which shrinks the Hessian part of B. ema_decay takes precedence over best_checkpoint.
    """
    rng = torch.Generator().manual_seed(seed)
    V_psi = ValueNet(env.d, env.T, cfg.hidden)
    agent = MethodAgent(V_psi, env.victim, env.theta_target, cfg, rng)
    opt = torch.optim.Adam(V_psi.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    history = []
    best_cost, best_state = float("inf"), None
    ema = copy.deepcopy(V_psi) if ema_decay is not None else None

    for k in range(cfg.K_iters):
        # --- 1. collect rollouts under current greedy policy + exploration ---
        eps = eps_schedule(k, cfg.K_iters, cfg.eps_start, cfg.eps_end)
        transitions = []
        for _ in range(cfg.N_episodes):
            transitions.extend(rollout(env, agent, rng, epsilon=eps)[0])
        buffer.push(transitions)                 # accumulate across iterations

        # --- 2. fit V_hat on TD targets (frozen label net) sampled from the buffer ---
        V_target = copy.deepcopy(V_psi)          # DQN-style, refreshed each fit round
        for _ in range(cfg.n_grad_steps):
            theta_b, t_b, c_b, theta_next_b = buffer.sample(cfg.batch_size, rng)
            with torch.no_grad():
                y = c_b + V_target(theta_next_b, t_b + 1)   # 0 if t+1 == T
            pred = V_psi(theta_b, t_b)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema is not None:                          # Polyak update: theta_ema <- d*theta_ema + (1-d)*theta
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), V_psi.parameters()):
                        pe.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

        if verbose or return_history or best_checkpoint:
            # fixed held-out eval set (same batches every iteration)
            mean_cost = evaluate(env, agent, torch.Generator().manual_seed(eval_seed),
                                 n_episodes=eval_episodes)
            history.append(mean_cost)
            if best_checkpoint and mean_cost < best_cost:
                best_cost = mean_cost
                best_state = copy.deepcopy(V_psi.state_dict())
            if verbose:
                print(f"[FVI] iter {k:2d}  eps={eps:.3f}  loss={loss.item():.4f}  "
                      f"mean_cost={mean_cost:.4f}")

    if ema is not None:
        V_psi.load_state_dict(ema.state_dict())      # return the Polyak-averaged weights
    elif best_checkpoint and best_state is not None:
        V_psi.load_state_dict(best_state)            # or the best-cost checkpoint

    return (V_psi, history) if return_history else V_psi
