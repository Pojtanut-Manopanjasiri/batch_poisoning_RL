import copy

import torch
import torch.nn as nn

from agents import MethodAgent
from decomposition import alpha_batch
from selection import select_greedy_batch
from value_net import ValueNet


class ReplayBuffer:
    """FIFO replay of transitions (theta, t, cost, theta_next, z)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.theta = self.t = self.c = self.theta_next = self.zx = self.zy = None

    def push(self, transitions):
        theta = torch.stack([tr[0] for tr in transitions])
        t = torch.tensor([tr[1] for tr in transitions], device=theta.device)
        c = torch.tensor([tr[2] for tr in transitions], dtype=torch.float32,
                         device=theta.device)
        theta_next = torch.stack([tr[3] for tr in transitions])
        zx = torch.stack([tr[4][0] for tr in transitions])       # (n, m, d) features
        zy = torch.stack([tr[4][1] for tr in transitions])       # (n, m) labels
        if self.theta is None:
            self.theta, self.t, self.c, self.theta_next = theta, t, c, theta_next
            self.zx, self.zy = zx, zy
        else:
            self.theta = torch.cat([self.theta, theta])
            self.t = torch.cat([self.t, t])
            self.c = torch.cat([self.c, c])
            self.theta_next = torch.cat([self.theta_next, theta_next])
            self.zx, self.zy = torch.cat([self.zx, zx]), torch.cat([self.zy, zy])
        if len(self) > self.capacity:                    # drop oldest (FIFO)
            k = len(self) - self.capacity
            self.theta, self.t = self.theta[k:], self.t[k:]
            self.c, self.theta_next = self.c[k:], self.theta_next[k:]
            self.zx, self.zy = self.zx[k:], self.zy[k:]

    def sample(self, batch_size, rng):
        idx = torch.randint(0, len(self), (min(batch_size, len(self)),), generator=rng)
        idx = idx.to(self.theta.device)
        return self.theta[idx], self.t[idx], self.c[idx], self.theta_next[idx]

    def sample_states(self, n, rng):
        """n distinct stored (theta, t, z) triples, for optimality targets."""
        idx = torch.randperm(len(self), generator=rng)[:n].to(self.theta.device)
        return self.theta[idx], self.t[idx], self.zx[idx], self.zy[idx]

    def __len__(self):
        return 0 if self.theta is None else len(self.theta)


def eps_schedule(k, K, e0, e1):
    return e0 + (e1 - e0) * (k / max(1, K - 1))


def rollout(env, agent, rng, epsilon=0.0, trace=False):
    """One episode. Returns (transitions, total_cost, steps)."""
    transitions, total, steps = [], 0.0, []
    theta = env.theta_0.clone()
    zs = None
    if hasattr(agent, "plan"):
        zs = [env.stream.sample_batch(env.m, rng) for _ in range(env.T)]
        agent.plan(theta, zs)
    for t in range(env.T):
        z = zs[t] if zs is not None else env.stream.sample_batch(env.m, rng)
        a = agent.act(theta, z, t, epsilon=epsilon)
        theta_next = env.victim.update(theta, z, a)
        c = env.cost(theta, z, a, theta_next)
        transitions.append((theta.detach(), t, c, theta_next.detach(), z))
        if trace:
            steps.append({"t": t, "action": [int(v) for v in a.tolist()], "cost": c,
                          "dist_to_target": float((theta_next - env.theta_target).norm())})
        total += c
        theta = theta_next.detach()
    return transitions, total, steps


def evaluate(env, agent, rng, n_episodes=100):
    """Mean episode cost of a policy"""
    return sum(rollout(env, agent, rng)[1] for _ in range(n_episodes)) / n_episodes


def trace_episodes(env, agent, rng, n_episodes):
    """Per-step actions and costs for n_episodes, through the same rollout path."""
    return [rollout(env, agent, rng, trace=True)[2] for _ in range(n_episodes)]


def _targets(env, cfg, agent, V, theta, t, zx, zy):
    if cfg.select_mode == "greedy":
        Phi = torch.func.vmap(lambda th, x, y: env.victim.phi_batch(th, (x, y)))(theta, zx, zy)
        alpha = alpha_batch(V, theta, t + 1, Phi, env.theta_target)
        a = select_greedy_batch(alpha, cfg.lam, cfg.B, cfg.balance, cfg.L)
        theta_next = (theta + torch.einsum("sm,smd->sd", a, Phi)).detach()
        with torch.no_grad():
            return env.cost_batch(a, theta_next) + V(theta_next, t + 1)
    y = torch.empty(len(theta), device=theta.device)
    for n in range(len(theta)):
        z, tn = (zx[n], zy[n]), int(t[n])
        a = agent.act(theta[n], z, tn)
        theta_next = env.victim.update(theta[n], z, a).detach()
        with torch.no_grad():
            y[n] = env.cost(theta[n], z, a, theta_next) + float(V(theta_next, tn + 1))
    return y
   
class StagedValue(nn.Module):
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
    per_stage, batches = [[] for _ in range(env.T)], []
    for _ in range(n_episodes):
        theta = env.theta_0.clone()
        for t in range(env.T):
            per_stage[t].append(theta.detach().clone())
            z = env.stream.sample_batch(env.m, rng)
            batches.append(z)
            a = agent.act(theta, z, t, epsilon=epsilon)
            theta = env.victim.update(theta, z, a).detach()
    return per_stage, batches


def train_fevi(env, cfg, seed=0, verbose=True, return_history=False,
               best_checkpoint=False, eval_episodes=100, eval_seed=999):
    """Fitted Expected-Value Iteration (Algorithm 1). """
    rng = torch.Generator().manual_seed(seed)
    shared = getattr(cfg, "fevi_shared", False)
    enc = getattr(cfg, "time_encoding", "elapsed")
    V = (ValueNet(env.d, env.T, cfg.hidden, time_encoding=enc) if shared
         else StagedValue(env.d, env.T, cfg.hidden, time_encoding=enc)).to(env.device)
    opt_shared = (torch.optim.Adam(V.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay) if shared else None)
    agent = MethodAgent(V, env.victim, env.theta_target, cfg, rng)

    cap = max(1, cfg.buffer_capacity // env.T)      
    R = [[] for _ in range(env.T)]
    RZx = RZy = None                               
    history, best_cost, best_state = [], float("inf"), None

    for k in range(cfg.fevi_outer):
        eps = eps_schedule(k, cfg.fevi_outer, cfg.eps_start, cfg.eps_end)
        fresh, batches = _collect_stage_states(env, agent, rng, cfg.fevi_N, eps)
        for t in range(env.T):
            R[t].extend(fresh[t])
            if len(R[t]) > cap:
                R[t] = R[t][-cap:]
        zx, zy = torch.stack([z[0] for z in batches]), torch.stack([z[1] for z in batches])
        RZx = zx if RZx is None else torch.cat([RZx, zx])[-cfg.buffer_capacity:]
        RZy = zy if RZy is None else torch.cat([RZy, zy])[-cfg.buffer_capacity:]

        pool_th, pool_t, pool_y = [], [], []      
        for t in reversed(range(env.T)):
            idx = torch.randint(0, len(R[t]), (min(cfg.fevi_N, len(R[t])),), generator=rng)
            thetas = torch.stack([R[t][int(i)] for i in idx])       
            j = torch.randint(0, len(RZx), (len(thetas) * cfg.fevi_M,), generator=rng)
            th_rep = thetas.repeat_interleave(cfg.fevi_M, dim=0)    
            t_rep = torch.full((len(th_rep),), t, device=env.device)
            y = _targets(env, cfg, agent, V, th_rep, t_rep, RZx[j.to(RZx.device)],
                         RZy[j.to(RZy.device)])
            Y = y.view(len(thetas), cfg.fevi_M).mean(1)              

            if shared:
                pool_th.append(thetas)
                pool_t.append(torch.full((len(thetas),), float(t), device=env.device))
                pool_y.append(Y)
                TH, TT, YY = torch.cat(pool_th), torch.cat(pool_t), torch.cat(pool_y)
                for _ in range(cfg.fevi_fit_steps):
                    j = torch.randint(0, len(TH), (min(cfg.batch_size, len(TH)),),
                                      generator=rng).to(env.device)
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
                                      generator=rng).to(env.device)
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
