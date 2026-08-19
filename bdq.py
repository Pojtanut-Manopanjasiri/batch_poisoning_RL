"""BDQ baseline: Branching Dueling Q-Network for batch-curation poisoning.

A model-free comparison point for our FVI + per-point decomposition method.

Algorithm ported from Tavakoli et al., "Action Branching Architectures for Deep
Reinforcement Learning" (AAAI 2018) -- github.com/atavakol/action-branching-agents
(MIT License, Copyright (c) 2018 Arash Tavakoli).  That reference implementation is
TensorFlow-1 / OpenAI-baselines; this is a faithful PyTorch re-implementation of the
same algorithm, using the paper's best-performing settings:

  - dueling with `aggregator='reduceLocalMean'`  : Q_i(a) = V(s) + A_i(a) - mean_a A_i(a)
  - `target_version='mean'`                      : one shared target, averaged over branches
  - `losses_version=2`                           : mean over branches of the per-branch TD loss
  - double-Q                                     : argmin from the online net, value from target

Why BDQ fits this problem: the joint action a in {0,1,2}^m has 3^m combinations
(59049 at m=10), which plain DQN cannot enumerate.  Action branching factors it into
m independent branches of 3 sub-actions each -- linear instead of exponential in m.

Four adaptations are forced by our setting (see README for the comparison caveats):

 1. COST, NOT REWARD.  We minimise, so every `max`/`argmax` in the reference becomes
    `min`/`argmin`.  gamma = 1 (finite horizon T, undiscounted), terminal Q = 0 at t = T.

 2. THE BATCH IS PART OF THE STATE.  s_t = (theta_t, z_t), and branch i must know its
    own point.  We use a permutation-equivariant encoder: a shared state torso over
    (theta, theta - theta_target, t), a shared per-point encoder over (phi_i, resid_i),
    and weight-shared branch heads.  Sharing branch weights is what makes the net
    independent of m (the reference uses one head per action dimension, which is fine
    for a fixed robot morphology but not for exchangeable data points).

 3. FEASIBILITY.  Branches choose independently, so the joint action generally violates
    the balance constraint (#deletes == #duplicates).  We project onto the feasible set
    using the SAME greedy-swap rule as `selection._select_sort`, but ranking by learned
    Q-margins instead of alpha -- so the constraint handling is identical across methods
    and the only thing being compared is how the action is *scored*.

 4. c_edit IS LEARNED, NOT GIVEN.  It reaches BDQ only through the observed cost, whereas
    our method gets it analytically in the objective.  That asymmetry is the point of the
    comparison: BDQ has to discover the edit/benefit trade-off from data.
"""
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
    """Q_i(s, a_i) for every point i and sub-action a_i in {delete, keep, duplicate}.

    Returns (batch, m, 3).  Dueling with local-mean aggregation: the shared state value
    V(s) carries the level, each branch's centred advantage carries the choice.
    """

    N_SUB = 3                                    # a_i in {0, 1, 2}

    def __init__(self, d, T, hidden=(128, 128)):
        super().__init__()
        self.d, self.T = d, T
        h = hidden[-1]
        self.torso = _mlp([2 * d + T, *hidden])          # theta, error, t one-hot
        self.point_enc = _mlp([d + 1, *hidden])          # phi_i, resid_i
        self.adv_head = _mlp([2 * h, *hidden, self.N_SUB])   # per-branch advantages
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
    return net(theta[None], err[None], torch.tensor([t]), phi[None], resid[None])[0]


# --------------------------------------------------------------------------- #
# Feasibility projection (mirrors selection._select_sort, on Q-margins)
# --------------------------------------------------------------------------- #
def project_action(q, balance=True, k_max=None):
    """Greedy feasible action from per-branch Q-values q (m, 3), minimising cost.

    Unbalanced: each branch picks its own argmin (pure BDQ), then the delete budget is
    enforced by keeping the most valuable deletes.

    Balanced: edits come in delete/duplicate swaps.  gain_del_i / gain_dup_i are the cost
    reductions from editing point i; ranking by (gain_del - gain_dup) puts delete
    candidates at one end and duplicate candidates at the other, so taking the top-r and
    bottom-r gives r disjoint swaps.  We scan r and keep the best -- the same greedy-swap
    structure as `_select_sort`, so both methods face the identical constraint.
    """
    m = q.shape[0]
    q = q.detach()
    gain_del = q[:, 1] - q[:, 0]                 # >0 : deleting i lowers cost
    gain_dup = q[:, 1] - q[:, 2]                 # >0 : duplicating i lowers cost

    if not balance:
        a = q.argmin(dim=1).float()
        if k_max is not None:
            del_idx = torch.where(a == 0.0)[0]
            if len(del_idx) > k_max:
                keep = del_idx[torch.argsort(gain_del[del_idx], descending=True)[:k_max]]
                a[del_idx] = 1.0
                a[keep] = 0.0
        return a

    max_r = m // 2
    if k_max is not None:
        max_r = min(max_r, k_max)
    a = torch.ones(m)
    if max_r == 0:
        return a

    order = torch.argsort(gain_del - gain_dup, descending=True)   # deletes first
    best_r, best_gain, running = 0, 0.0, 0.0
    for r in range(1, max_r + 1):
        running += float(gain_del[order[r - 1]] + gain_dup[order[m - r]])
        if running > best_gain:                  # scan (not stop): gains aren't monotone
            best_gain, best_r = running, r
    for r in range(best_r):
        a[order[r]] = 0.0                        # delete
        a[order[m - 1 - r]] = 2.0                # duplicate
    return a


class BDQAgent:
    """Greedy (or epsilon-greedy) policy w.r.t. the learned branch Q-values."""

    def __init__(self, net, env, cfg, rng):
        self.net, self.env, self.cfg, self.rng = net, env, cfg, rng

    @torch.no_grad()
    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.k_max)
        q = _q_values(self.net, self.env, theta, z, t)
        return project_action(q, self.cfg.balance, self.cfg.k_max)


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
    z = [b[1] for b in batch]
    t = torch.tensor([b[2] for b in batch])
    a = torch.stack([b[3] for b in batch]).long()
    c = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    th_n = torch.stack([b[5] for b in batch])
    z_n = [b[6] for b in batch]
    done = torch.tensor([b[7] for b in batch], dtype=torch.float32)

    def feat(thetas, zs):
        errs, phis, resids = zip(*[_features(env, thetas[i], zs[i]) for i in range(len(zs))])
        return torch.stack(errs), torch.stack(phis), torch.stack(resids)

    return (th, *feat(th, z), t, a, c, th_n, *feat(th_n, z_n), done)


def train_bdq(env, cfg, seed=0, verbose=True, return_history=False,
              best_checkpoint=True, eval_episodes=100, eval_seed=999):
    """Train BDQ. Same outer loop / budget as train_fvi, so the comparison is fair."""
    from fvi import eps_schedule, evaluate

    rng = torch.Generator().manual_seed(seed)
    net = BranchQNet(env.d, env.T, cfg.hidden)
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
