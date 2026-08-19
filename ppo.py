"""PPO baseline for batch-curation poisoning.

Why PPO needs no max over the action space
------------------------------------------
The joint action is a in {0,1,2}^m -- 3^m combinations. Value-based methods (DQN, and BDQ)
must bootstrap with max_a Q(s', a), which is exactly the intractable part; BDQ makes it
tractable by factoring Q into m branches. PPO sidesteps the max entirely:

  * its critic is a STATE value V(s), so the TD target r + gamma V(s') has no max;
  * its actor is updated by policy gradient on SAMPLED actions, never an argmax.

So the combinatorial difficulty moves from "maximise over 3^m" to "sample from, and score,
a distribution over 3^m".

How we parameterise that distribution: a policy over SWAPS
----------------------------------------------------------
The obvious choice -- an independent 3-way categorical per point, then project the sample
onto the feasible set -- FAILS badly here, and the reason is instructive. With
`balance=True` an action is only feasible when #deletes == #duplicates, so flipping a
single point to "delete" changes nothing until a matching duplicate also appears. The
projection erases single-coordinate changes, the factored policy gradient sits on a
plateau, and PPO reliably collapses to the all-keep (null) policy.

Instead we parameterise the policy over the feasible set directly, in the same
delete/duplicate SWAP coordinates the constraint is written in:

    r          ~ Categorical(r_logits)                    how many swaps, r <= max_r
    del_1..r   ~ sequential softmax over del_scores       r deletes, without replacement
    dup_1..r   ~ sequential softmax over dup_scores       r duplicates, disjoint from those

This is a Plackett-Luce draw. It gives

  * feasibility BY CONSTRUCTION -- balance and k_max hold exactly, no projection;
  * an EXACT log-probability, log pi = log pi(r) + sum_j log pi(del_j) + sum_j log pi(dup_j),
    so the PPO ratio is unbiased;
  * O(m) sampling from a space of size 3^m, and one network forward per decision.

The sequential masking is what makes "sample without replacement" tractable: each draw is
a plain softmax over the still-available points, so the joint log-prob telescopes into a
sum. Only `evaluate`'s deterministic mode uses an argmax, and that is a greedy top-r read
of the same scores (cheap), never a search over 3^m.
"""
import copy

import torch
import torch.nn as nn

from bdq import _features
from env import random_feasible_action
from fvi import evaluate

NEG = -1e9          # mask value; finite so log_softmax never yields nan


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def max_swaps(m, k_max=None):
    r = m // 2
    return r if k_max is None else min(r, k_max)


class ActorCritic(nn.Module):
    """Shared trunk; swap-policy heads (r, delete scores, duplicate scores) + V(s).

    Mirrors BranchQNet's feature scheme (theta, theta-target, t, and a shared per-point
    encoder over (phi_i, resid_i)) so PPO and BDQ see identical information.
    """

    def __init__(self, d, T, max_r, hidden=(128, 128)):
        super().__init__()
        self.T, self.max_r = T, max_r
        h = hidden[-1]
        self.torso = _mlp([2 * d + T, *hidden])
        self.point_enc = _mlp([d + 1, *hidden])
        self.del_head = _mlp([2 * h, *hidden, 1])          # per-point delete score
        self.dup_head = _mlp([2 * h, *hidden, 1])          # per-point duplicate score
        self.r_head = _mlp([2 * h, *hidden, max_r + 1])    # how many swaps
        self.critic = _mlp([2 * h, *hidden, 1])            # state value

    def forward(self, theta, err, t, phi, resid):
        """theta,err (B,d); t (B,); phi (B,m,d); resid (B,m).
        Returns r_logits (B,max_r+1), del_s (B,m), dup_s (B,m), v (B,)."""
        t1h = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, t1h], dim=-1))            # (B, h)
        h_p = self.point_enc(torch.cat([phi, resid[..., None]], dim=-1))  # (B, m, h)
        h_sp = torch.cat([h_s[:, None, :].expand_as(h_p), h_p], dim=-1)   # (B, m, 2h)
        pooled = torch.cat([h_s, h_p.mean(1)], dim=-1)                    # (B, 2h)
        return (self.r_head(pooled), self.del_head(h_sp).squeeze(-1),
                self.dup_head(h_sp).squeeze(-1), self.critic(pooled).squeeze(-1))


# --------------------------------------------------------------------------- #
# swap-policy: sampling, scoring, and the deterministic (greedy) read
# --------------------------------------------------------------------------- #
def _draw(scores, avail, rng):
    """One masked-softmax draw without replacement. Returns (index, logprob)."""
    lp = torch.log_softmax(scores.masked_fill(~avail, NEG), dim=-1)
    i = int(torch.multinomial(lp.exp(), 1, generator=rng))
    return i, lp[i]


def sample_swaps(r_logits, del_s, dup_s, max_r, rng):
    """Plackett-Luce draw. Returns (a, logp, r, del_idx, dup_idx) with a feasible."""
    m = del_s.shape[0]
    lp_r = torch.log_softmax(r_logits, dim=-1)
    r = int(torch.multinomial(lp_r.exp(), 1, generator=rng))
    logp = lp_r[r]

    avail = torch.ones(m, dtype=torch.bool)
    del_idx, dup_idx = [], []
    for _ in range(r):                                  # deletes
        i, lp = _draw(del_s, avail, rng)
        del_idx.append(i); avail[i] = False; logp = logp + lp
    for _ in range(r):                                  # duplicates, disjoint
        i, lp = _draw(dup_s, avail, rng)
        dup_idx.append(i); avail[i] = False; logp = logp + lp

    a = torch.ones(m)
    for i in del_idx:
        a[i] = 0.0
    for i in dup_idx:
        a[i] = 2.0
    return a, logp, r, del_idx, dup_idx


def logp_swaps(r_logits, del_s, dup_s, r, del_idx, dup_idx, max_r):
    """Exact log-prob of stored ordered draws, batched.

    r (B,) long; del_idx/dup_idx (B, max_r) padded with -1. Replays the same sequential
    masking used at sampling time, so the ratio in the PPO surrogate is exact.
    """
    B, m = del_s.shape
    logp = torch.log_softmax(r_logits, -1).gather(1, r[:, None]).squeeze(1)
    avail = torch.ones(B, m, dtype=torch.bool, device=del_s.device)
    for scores, idx in ((del_s, del_idx), (dup_s, dup_idx)):
        for j in range(max_r):
            active = idx[:, j] >= 0                     # rows that made a j-th draw
            if not bool(active.any()):
                continue
            lp = torch.log_softmax(scores.masked_fill(~avail, NEG), dim=-1)
            pick = idx[:, j].clamp(min=0)
            step = lp.gather(1, pick[:, None]).squeeze(1)
            logp = logp + torch.where(active, step, torch.zeros_like(step))
            hit = torch.nn.functional.one_hot(pick, m).bool() & active[:, None]
            avail = avail & ~hit
    return logp


def entropy_swaps(r_logits, del_s, dup_s):
    """Cheap entropy proxy: H(r) + H(first delete draw) + H(first duplicate draw).

    The exact Plackett-Luce entropy has no closed form; these three terms are the ones
    that actually drive exploration (how many swaps, and which points are favoured).
    """
    def H(logits):
        lp = torch.log_softmax(logits, -1)
        return -(lp.exp() * lp).sum(-1)
    return H(r_logits) + H(del_s) + H(dup_s)


def greedy_swaps(r_logits, del_s, dup_s):
    """Deterministic read used at evaluation: take the modal r, then its top-r scores."""
    m = del_s.shape[0]
    r = int(r_logits.argmax())
    a = torch.ones(m)
    if r == 0:
        return a
    order_del = torch.argsort(del_s, descending=True)
    dels = order_del[:r]
    mask = torch.ones(m, dtype=torch.bool)
    mask[dels] = False
    cand = torch.where(mask)[0]
    dups = cand[torch.argsort(dup_s[cand], descending=True)[:r]]
    a[dels] = 0.0
    a[dups] = 2.0
    return a


class PPOAgent:
    """Deterministic (greedy swap read) for evaluation; stochastic during collection."""

    def __init__(self, net, env, cfg, rng, deterministic=True):
        self.net, self.env, self.cfg, self.rng = net, env, cfg, rng
        self.deterministic = deterministic
        self.max_r = max_swaps(env.m, cfg.k_max)

    @torch.no_grad()
    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.k_max)
        err, phi, resid = _features(self.env, theta, z)
        rl, ds, us, _ = self.net(theta[None], err[None], torch.tensor([t]),
                                 phi[None], resid[None])
        if self.deterministic:
            return greedy_swaps(rl[0], ds[0], us[0])
        return sample_swaps(rl[0], ds[0], us[0], self.max_r, self.rng)[0]


# --------------------------------------------------------------------------- #
# rollout + GAE + PPO update
# --------------------------------------------------------------------------- #
def _rollout_ppo(env, net, cfg, rng, max_r):
    recs, total = [], 0.0
    theta = env.theta_0.clone()
    for t in range(env.T):
        z = env.stream.sample_batch(env.m, rng)
        err, phi, resid = _features(env, theta, z)
        with torch.no_grad():
            rl, ds, us, v = net(theta[None], err[None], torch.tensor([t]),
                                phi[None], resid[None])
        a, logp, r, di, ui = sample_swaps(rl[0], ds[0], us[0], max_r, rng)
        theta_next = env.victim.update(theta, z, a)
        c = env.cost(theta, z, a, theta_next)
        pad = lambda idx: idx + [-1] * (max_r - len(idx))
        recs.append({"theta": theta.detach(), "err": err.detach(), "t": t,
                     "phi": phi.detach(), "resid": resid.detach(),
                     "r": r, "del": pad(di), "dup": pad(ui),
                     "logp": float(logp), "v": float(v), "reward": -c})
        total += c
        theta = theta_next.detach()
    return recs, total


def _gae(recs, gamma, lam):
    adv, last = [0.0] * len(recs), 0.0
    for i in reversed(range(len(recs))):
        v_next = recs[i + 1]["v"] if i + 1 < len(recs) else 0.0   # terminal V = 0
        last = recs[i]["reward"] + gamma * v_next - recs[i]["v"] + gamma * lam * last
        adv[i] = last
    for i, rec in enumerate(recs):
        rec["adv"], rec["ret"] = adv[i], adv[i] + rec["v"]
    return recs


def train_ppo(env, cfg, seed=0, verbose=True, return_history=False,
              best_checkpoint=False, eval_episodes=100, eval_seed=999):
    """Train a PPO swap-policy. Signature matches train_fvi / train_bdq.

    NOTE on budget: PPO is ON-policy, so each iteration's data is used for a few epochs
    and discarded -- it cannot consume the same number of gradient steps as the
    replay-based learners without overfitting its on-policy batch. We match ENVIRONMENT
    INTERACTIONS (K_iters * N_episodes), the standard currency for on-policy vs
    off-policy comparisons, and let gradient-step counts differ.
    """
    rng = torch.Generator().manual_seed(seed)
    max_r = max_swaps(env.m, cfg.k_max)
    net = ActorCritic(env.d, env.T, max_r, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.ppo_lr, weight_decay=cfg.weight_decay)
    history, best_cost, best_state = [], float("inf"), None

    for k in range(cfg.K_iters):
        batch = []
        for _ in range(cfg.N_episodes):
            recs, _ = _rollout_ppo(env, net, cfg, rng, max_r)
            batch.extend(_gae(recs, cfg.gamma, cfg.gae_lambda))

        th_b = torch.stack([r["theta"] for r in batch])
        er_b = torch.stack([r["err"] for r in batch])
        t_b = torch.tensor([r["t"] for r in batch])
        ph_b = torch.stack([r["phi"] for r in batch])
        rs_b = torch.stack([r["resid"] for r in batch])
        r_b = torch.tensor([r["r"] for r in batch])
        di_b = torch.tensor([r["del"] for r in batch])
        ui_b = torch.tensor([r["dup"] for r in batch])
        lp_old = torch.tensor([r["logp"] for r in batch])
        adv_b = torch.tensor([r["adv"] for r in batch], dtype=torch.float32)
        ret_b = torch.tensor([r["ret"] for r in batch], dtype=torch.float32)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        n = len(batch)
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, generator=rng)
            for s in range(0, n, cfg.ppo_minibatch):
                i = perm[s:s + cfg.ppo_minibatch]
                rl, ds, us, v = net(th_b[i], er_b[i], t_b[i], ph_b[i], rs_b[i])
                logp = logp_swaps(rl, ds, us, r_b[i], di_b[i], ui_b[i], max_r)
                ratio = torch.exp(logp - lp_old[i])
                a_i = adv_b[i]
                surr = torch.min(ratio * a_i,
                                 torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a_i)
                loss = (-surr.mean()
                        + cfg.value_coef * ((v - ret_b[i]) ** 2).mean()
                        - cfg.entropy_coef * entropy_swaps(rl, ds, us).mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        if verbose or return_history or best_checkpoint:
            agent = PPOAgent(net, env, cfg, rng, deterministic=True)
            mean_cost = evaluate(env, agent, torch.Generator().manual_seed(eval_seed),
                                 n_episodes=eval_episodes)
            history.append(mean_cost)
            if best_checkpoint and mean_cost < best_cost:
                best_cost, best_state = mean_cost, copy.deepcopy(net.state_dict())
            if verbose:
                print(f"[PPO] iter {k:2d}  loss={loss.item():.4f}  mean_cost={mean_cost:.4f}")

    if best_checkpoint and best_state is not None:
        net.load_state_dict(best_state)
    return (net, history) if return_history else net
