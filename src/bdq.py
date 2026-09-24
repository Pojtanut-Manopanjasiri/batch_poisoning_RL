import copy

import torch
import torch.nn as nn

from env import random_feasible_action


def _mlp(sizes):
    """Tanh MLP (same activation as ValueNet, for a like-for-like function class)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class BranchQNet(nn.Module):
    """Q_i(s, a_i) for every point i and sub-action a_i in {0, ..., L}.

    Returns (batch, m, L+1).  Dueling with local-mean aggregation: the shared state value
    V(s) carries the level, each branch's centred advantage carries the choice.
    """

    def __init__(self, d, T, hidden=(128, 128), L=2):
        super().__init__()
        self.d, self.T, self.n_sub = d, T, L + 1
        h = hidden[-1]
        self.torso = _mlp([2 * d + T, *hidden])          # theta, error, t one-hot
        self.point_enc = _mlp([d + 1, *hidden])          # phi_i, resid_i
        self.adv_head = _mlp([2 * h, *hidden, self.n_sub])   # per-branch advantages
        self.value_head = _mlp([2 * h, *hidden, 1])          # shared state value

    def forward(self, theta, err, t, phi, resid):
        """theta,err (B,d); t (B,) long; phi (B,m,d); resid (B,m) -> Q (B,m,3)."""
        t_oh = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, t_oh], dim=-1))          # (B, h)
        h_p = self.point_enc(torch.cat([phi, resid[..., None]], dim=-1))  # (B, m, h)

        h_s_exp = h_s[:, None, :].expand(-1, h_p.shape[1], -1)
        adv = self.adv_head(torch.cat([h_s_exp, h_p], dim=-1))            # (B, m, 3)
        adv = adv - adv.mean(dim=-1, keepdim=True)          # reduceLocalMean aggregator

        pooled = h_p.mean(dim=1)                            # permutation-invariant summary
        value = self.value_head(torch.cat([h_s, pooled], dim=-1))         # (B, 1)
        return value[:, None, :] + adv                      # (B, m, 3)


def _features(env, theta, z):
    """State/point features for one (theta, z): err, phi (m,d), resid (m).

    The residual comes from the victim so BDQ sees the task-correct quantity
    (least-squares vs logistic) -- same features either way, no BDQ change needed.
    """
    resid = env.victim.residual(theta, z)
    phi = env.victim.phi_batch(theta, z)
    return theta - env.theta_target, phi, resid


def _q_values(net, env, theta, z, t):
    """Q for a single state -> (m, 3)."""
    err, phi, resid = _features(env, theta, z)
    return net(theta[None], err[None], torch.tensor([t], device=theta.device),
               phi[None], resid[None])[0]


# --------------------------------------------------------------------------- #
# Feasibility projection (mirrors selection._select_greedy, on Q-margins)
# --------------------------------------------------------------------------- #
def project_action(q, balance=True, B=None, L=2):
    """Feasible action from per-branch Q-values q (m, L+1), minimising cost.

    Same Algorithm-2 allocation as `_select_greedy`, with the marginal cost of raising
    point i from level s taken as Q_i(s+1) - Q_i(s) instead of alpha_i +/- rho, so both
    methods face the identical constraint and differ only in scoring.
    """
    # allocated on CPU: the loop below reads an argmin back on every step
    device = q.device
    return _project_cpu(q.detach().cpu(), balance, B, L).to(device)


def _project_cpu(q, balance, B, L):
    m = q.shape[0]

    if not balance:
        a = q.argmin(dim=1)
        if B is not None:
            cost = (a.float() - 1.0).abs()
            gain = q[:, 1] - q.gather(1, a[:, None]).squeeze(1)
            spent, keep = 0.0, torch.ones(m)
            for i in torch.argsort(gain, descending=True):
                if spent + float(cost[i]) <= B:
                    keep[i] = a[i].float()
                    spent += float(cost[i])
            return keep
        return a.float()

    k = m if B is None else B // 2
    a = torch.zeros(m, dtype=torch.long)
    D = 0
    for _ in range(m):                            # m allocations => sum_i a_i = m
        nxt = (a + 1).clamp(max=L)
        marg = q.gather(1, nxt[:, None]).squeeze(1) - q.gather(1, a[:, None]).squeeze(1)
        elig = (a < L) & ((a == 0) | (D < k))
        marg = marg.masked_fill(~elig, float("inf"))
        i = int(torch.argmin(marg))
        if a[i] >= 1:
            D += 1
        a[i] += 1
    return a.float()


class BDQAgent:
    """Greedy (or epsilon-greedy) policy w.r.t. the learned branch Q-values."""

    def __init__(self, net, env, cfg, rng):
        self.net, self.env, self.cfg, self.rng = net, env, cfg, rng

    @torch.no_grad()
    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.B,
                                          self.cfg.L, device=z[0].device)
        q = _q_values(self.net, self.env, theta, z, t)
        return project_action(q, self.cfg.balance, self.cfg.B, self.cfg.L)


class BDQBuffer:
    """FIFO replay of (theta, z, t, a, cost, theta_next, z_next, done)."""

    def __init__(self, capacity):
        self.capacity, self.data = capacity, []

    def push(self, transitions):
        self.data.extend(transitions)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]

    def sample(self, batch_size, rng):
        idx = torch.randint(0, len(self.data), (min(batch_size, len(self.data)),),
                            generator=rng)
        return [self.data[int(i)] for i in idx]

    def __len__(self):
        return len(self.data)


def _stack(env, batch):
    """Collate sampled transitions into tensors for the network."""
    th = torch.stack([b[0] for b in batch])
    dev = th.device
    z = [b[1] for b in batch]
    t = torch.tensor([b[2] for b in batch], device=dev)
    a = torch.stack([b[3] for b in batch]).long()
    c = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=dev)
    th_n = torch.stack([b[5] for b in batch])
    z_n = [b[6] for b in batch]
    done = torch.tensor([b[7] for b in batch], dtype=torch.float32, device=dev)

    def feat(thetas, zs):
        errs, phis, resids = zip(*[_features(env, thetas[i], zs[i]) for i in range(len(zs))])
        return torch.stack(errs), torch.stack(phis), torch.stack(resids)

    return (th, *feat(th, z), t, a, c, th_n, *feat(th_n, z_n), done)


def train_bdq(env, cfg, seed=0, verbose=True, return_history=False,
              best_checkpoint=True, eval_episodes=100, eval_seed=999):
    """Train BDQ. Same outer loop / budget as train_fvi, so the comparison is fair."""
    from fvi import eps_schedule, evaluate

    rng = torch.Generator().manual_seed(seed)
    net = BranchQNet(env.d, env.T, cfg.hidden, cfg.L).to(env.device)
    agent = BDQAgent(net, env, cfg, rng)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = BDQBuffer(cfg.buffer_capacity)
    history, best_cost, best_state = [], float("inf"), None

    for k in range(cfg.K_iters):
        # --- 1. collect episodes under the current epsilon-greedy policy ---
        eps = eps_schedule(k, cfg.K_iters, cfg.eps_start, cfg.eps_end)
        transitions = []
        for _ in range(cfg.N_episodes):
            theta = env.theta_0.clone()
            z = env.stream.sample_batch(env.m, rng)
            for t in range(env.T):
                a = agent.act(theta, z, t, epsilon=eps)
                theta_next = env.victim.update(theta, z, a).detach()
                c = env.cost(theta, z, a, theta_next)
                done = float(t + 1 == env.T)
                z_next = (z if done else env.stream.sample_batch(env.m, rng))
                transitions.append((theta.detach(), z, t, a, c, theta_next, z_next, done))
                theta, z = theta_next, z_next
        buffer.push(transitions)

        # --- 2. fit against a frozen target net ---
        target_net = copy.deepcopy(net)
        for _ in range(cfg.n_grad_steps):
            (th, err, phi, res, t, a, c,
             th_n, err_n, phi_n, res_n, done) = _stack(env, buffer.sample(cfg.batch_size, rng))

            with torch.no_grad():
                q_next_online = net(th_n, err_n, t + 1, phi_n, res_n)      # double-Q
                a_star = q_next_online.argmin(dim=-1, keepdim=True)
                q_next = target_net(th_n, err_n, t + 1, phi_n, res_n)
                boot = q_next.gather(-1, a_star).squeeze(-1)               # (B, m)
                boot = boot.mean(dim=1) * (1.0 - done)     # target_version='mean'; 0 if terminal
                y = c + boot                                              # gamma = 1

            q = net(th, err, phi=phi, resid=res, t=t)
            q_taken = q.gather(-1, a[..., None]).squeeze(-1)               # (B, m)
            loss = ((q_taken - y[:, None]) ** 2).mean()   # losses_version=2 (mean over branches)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose or return_history or best_checkpoint:
            mean_cost = evaluate(env, agent, torch.Generator().manual_seed(eval_seed),
                                 n_episodes=eval_episodes)
            history.append(mean_cost)
            if best_checkpoint and mean_cost < best_cost:
                best_cost, best_state = mean_cost, copy.deepcopy(net.state_dict())
            if verbose:
                print(f"[BDQ] iter {k:2d}  eps={eps:.3f}  loss={loss.item():.4f}  "
                      f"mean_cost={mean_cost:.4f}")

    if best_checkpoint and best_state is not None:
        net.load_state_dict(best_state)
    return (net, history) if return_history else net
