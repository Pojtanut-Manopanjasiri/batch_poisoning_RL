"""Value learning over V_bar_t(theta)  (Brief II, sec. 4.1).

Two algorithms, same V(theta, t) interface:

  train_fvi  -- one net with t as a feature; TD targets vs a frozen label net,
                swept K_iters times.
  train_fevi -- Algorithm 1: backward induction t = T-1..0 over T stage nets, with
                the min over A inside the target and an M-sample batch expectation.

The decomposition is used for action *selection* only, never for bootstrapping.
"""
import copy

import torch
import torch.nn as nn

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
    V_psi = ValueNet(env.d, env.T, cfg.hidden,
                     time_encoding=getattr(cfg, "time_encoding", "elapsed"))
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


# --------------------------------------------------------------------------- #
# Fitted Expected-Value Iteration (Algorithm 1) -- explicit backward induction
# --------------------------------------------------------------------------- #
class StagedValue(nn.Module):
    """Algorithm 1's per-stage {h_t} behind ValueNet's V(theta, t) interface, so the
    decomposition / agents / rollout code is reused unchanged.

    h_T == 0 delegates to a real net at t=T: the mask zeroes it but KEEPS THE AUTOGRAD
    GRAPH ALIVE.  A detached constant breaks B's double-backward (surfaces only at
    t=T-1 with select_mode != "greedy").
    """

    def __init__(self, d, T, hidden, time_encoding="elapsed"):
        super().__init__()
        self.T = T
        self.h = nn.ModuleList([ValueNet(d, T, hidden, time_encoding=time_encoding)
                                for _ in range(T)])

    def forward(self, theta, t):
        ti = int(t) if not torch.is_tensor(t) else int(t.reshape(-1)[0])
        if ti >= self.T:
            return self.h[self.T - 1](theta, self.T)     # masked to 0, graph intact
        return self.h[ti](theta, ti)


def _collect_stage_states(env, agent, rng, n_episodes, epsilon):
    """Roll the current policy forward and bucket the visited theta_t by stage t."""
    per_stage = [[] for _ in range(env.T)]
    for _ in range(n_episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            per_stage[t].append(theta.detach().clone())
            z = env.stream.sample_batch(env.m, rng)
            a = agent.act(theta, z, t, epsilon=epsilon)
            theta = env.victim.update(theta, z, a).detach()
    return per_stage


def train_fevi(env, cfg, seed=0, verbose=True, return_history=False,
               best_checkpoint=False, eval_episodes=100, eval_seed=999):
    """Fitted Expected-Value Iteration (Algorithm 1).  Backward, t = T-1..0:

        Y_t^(n) = 1/M sum_j  min_{a in A} { g_t(theta^(n), Z^(n,j), a)
                                            + h_{t+1}(f(theta^(n), Z^(n,j), a)) }
        omega_t = argmin_omega  1/N sum_n [ h(theta^(n); omega) - Y_t^(n) ]^2

    vs train_fvi: the min is re-solved inside the target (an OPTIMALITY backup, where
    train_fvi bootstraps off the eps-greedy action actually taken); the batch
    expectation uses M fresh draws, not the one realised batch; and h_{t+1} is final
    when stage t is fit, so no frozen target and no repeated sweeps are needed.

    The min uses select_action's approximate argmin, but the target is the TRUE value
    g + h_{t+1}(f(.)) of the chosen action, not its Taylor-approximated objective.

    Algorithm 1 takes {R_t} as given; here they come from rolling the current policy
    forward, and cfg.fevi_outer repeats the whole pass so the state distribution can
    catch up.  fevi_outer=1 is Algorithm 1 exactly as written.
    """
    rng = torch.Generator().manual_seed(seed)
    # shared: one net with t as a feature instead of a separate omega_t per stage --
    # per-stage fits see only N states, which leaves grad V_hat flat.
    shared = getattr(cfg, "fevi_shared", False)
    enc = getattr(cfg, "time_encoding", "elapsed")
    V = (ValueNet(env.d, env.T, cfg.hidden, time_encoding=enc) if shared
         else StagedValue(env.d, env.T, cfg.hidden, time_encoding=enc))
    opt_shared = (torch.optim.Adam(V.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay) if shared else None)
    agent = MethodAgent(V, env.victim, env.theta_target, cfg, rng)

    cap = max(1, cfg.buffer_capacity // env.T)      # per-stage replay cap
    R = [[] for _ in range(env.T)]
    history, best_cost, best_state = [], float("inf"), None

    for k in range(cfg.fevi_outer):
        # --- regenerate {R_t} under the current policy (+ exploration) ---
        eps = eps_schedule(k, cfg.fevi_outer, cfg.eps_start, cfg.eps_end)
        fresh = _collect_stage_states(env, agent, rng, cfg.fevi_N, eps)
        for t in range(env.T):
            R[t].extend(fresh[t])
            if len(R[t]) > cap:
                R[t] = R[t][-cap:]

        # --- Algorithm 1, lines 2-10: backward induction ---
        pool_th, pool_t, pool_y = [], [], []       # shared variant: grows as we descend
        for t in reversed(range(env.T)):
            idx = torch.randint(0, len(R[t]), (min(cfg.fevi_N, len(R[t])),), generator=rng)
            thetas = torch.stack([R[t][int(i)] for i in idx])          # line 3

            Y = torch.empty(len(thetas))
            for n, theta in enumerate(thetas):                         # line 4
                acc = 0.0
                for _ in range(cfg.fevi_M):                            # line 5
                    z = env.stream.sample_batch(env.m, rng)
                    a = agent.act(theta, z, t)                         # approx argmin
                    theta_next = env.victim.update(theta, z, a).detach()
                    with torch.no_grad():                              # TRUE value of a
                        acc += env.cost(theta, z, a, theta_next) + float(V(theta_next, t + 1))
                Y[n] = acc / cfg.fevi_M                                # line 6

            # --- line 8: fit the approximator on (theta^(n), Y_t^(n)) ---
            if shared:
                # pool with stages already solved, so fitting t does not overwrite them
                pool_th.append(thetas)
                pool_t.append(torch.full((len(thetas),), float(t)))
                pool_y.append(Y)
                TH, TT, YY = torch.cat(pool_th), torch.cat(pool_t), torch.cat(pool_y)
                for _ in range(cfg.fevi_fit_steps):
                    j = torch.randint(0, len(TH), (min(cfg.batch_size, len(TH)),),
                                      generator=rng)
                    loss = ((V(TH[j], TT[j]) - YY[j]) ** 2).mean()
                    opt_shared.zero_grad()
                    loss.backward()
                    opt_shared.step()
            else:
                net = V.h[t]
                opt = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
                for _ in range(cfg.fevi_fit_steps):
                    j = torch.randint(0, len(thetas), (min(cfg.batch_size, len(thetas)),),
                                      generator=rng)
                    loss = ((net(thetas[j], t) - Y[j]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            if verbose:
                print(f"[FEVI] pass {k:2d}  stage t={t:2d}  "
                      f"mean target={float(Y.mean()):8.4f}  fit loss={float(loss):8.5f}")

        if verbose or return_history or best_checkpoint:
            mean_cost = evaluate(env, agent, torch.Generator().manual_seed(eval_seed),
                                 n_episodes=eval_episodes)
            history.append(mean_cost)
            if best_checkpoint and mean_cost < best_cost:
                best_cost = mean_cost
                best_state = copy.deepcopy(V.state_dict())
            if verbose:
                print(f"[FEVI] pass {k:2d}  eps={eps:.3f}  mean_cost={mean_cost:.4f}")

    if best_checkpoint and best_state is not None:
        V.load_state_dict(best_state)

    return (V, history) if return_history else V
