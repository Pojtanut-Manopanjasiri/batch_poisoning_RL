import copy
import math
import warnings

import torch
import torch.nn as nn

from alloc_policy import (AllocationPolicy, sample as alloc_sample,
                          log_prob as alloc_log_prob, kl as alloc_kl,
                          mode as alloc_mode)
from bdq import _features, _mean_cost, _myopic_scores
from env import random_feasible_action
from fvi import evaluate

NEG = -1e9         


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def max_swaps(m, B=None, L=2):
    r = m * (L - 1) // L                 # r deletes need r copies on m - r points, <= L - 1 each
    return r if B is None else min(r, B // 2)


def _next_copy(dup_s, cnt):
    # each point's score for its next extra copy: dup_s[..., cnt]
    k = cnt.clamp(max=dup_s.shape[-1] - 1)
    return dup_s.gather(-1, k[..., None]).squeeze(-1)


def _draw(scores, avail, rng):
    """One masked-softmax draw. Returns (index, logprob)."""
    lp = torch.log_softmax(scores.masked_fill(~avail, NEG), dim=-1)
    i = int(torch.multinomial(lp.exp(), 1, generator=rng))
    return i, lp[i]


def sample_swaps(r_logits, del_s, dup_s, max_r, rng):
    """Plackett-Luce draw. Returns (a, logp, r, del_idx, dup_idx) with a feasible."""
   
    device = del_s.device
    r_logits, del_s, dup_s = r_logits.cpu(), del_s.cpu(), dup_s.cpu()
    m = del_s.shape[0]
    lp_r = torch.log_softmax(r_logits, dim=-1)
    r = int(torch.multinomial(lp_r.exp(), 1, generator=rng))
    logp = lp_r[r]

    deleted = torch.zeros(m, dtype=torch.bool)
    cnt = torch.zeros(m, dtype=torch.long)              # extra copies per point
    cap = dup_s.shape[-1]                               # L - 1
    del_idx, dup_idx = [], []
    for _ in range(r):
        i, lp = _draw(del_s, ~deleted, rng)
        del_idx.append(i); deleted[i] = True; logp = logp + lp
    for _ in range(r):                                  # a point can be drawn until a_i = L
        i, lp = _draw(_next_copy(dup_s, cnt), ~deleted & (cnt < cap), rng)
        dup_idx.append(i); cnt[i] += 1; logp = logp + lp

    a = (~deleted).float() + cnt.float()
    return a.to(device), logp, r, del_idx, dup_idx


def _replay(del_idx, dup_idx, max_r, m, cap):
    # walks the stored draws: yields (copy_phase, active, pick, avail, cnt) before each one
    B = del_idx.shape[0]
    deleted = torch.zeros(B, m, dtype=torch.bool, device=del_idx.device)
    cnt = torch.zeros(B, m, dtype=torch.long, device=del_idx.device)
    for copy_phase, idx in ((False, del_idx), (True, dup_idx)):
        for j in range(max_r):
            active = idx[:, j] >= 0
            if not bool(active.any()):
                continue
            pick = idx[:, j].clamp(min=0)
            avail = ~deleted & (cnt < cap) if copy_phase else ~deleted
            yield copy_phase, active, pick, avail, cnt
            hit = torch.nn.functional.one_hot(pick, m).bool() & active[:, None]
            if copy_phase:
                cnt = cnt + hit.long()
            else:
                deleted = deleted | hit


def logp_swaps(r_logits, del_s, dup_s, r, del_idx, dup_idx, max_r):
    """Exact log-prob of stored ordered draws"""
    B, m = del_s.shape
    logp = torch.log_softmax(r_logits, -1).gather(1, r[:, None]).squeeze(1)
    cap = dup_s.shape[-1]
    for copy_phase, active, pick, avail, cnt in _replay(del_idx, dup_idx, max_r, m, cap):
        scores = _next_copy(dup_s, cnt) if copy_phase else del_s
        lp = torch.log_softmax(scores.masked_fill(~avail, NEG), dim=-1)
        step = lp.gather(1, pick[:, None]).squeeze(1)
        logp = logp + torch.where(active, step, torch.zeros_like(step))
    return logp


def greedy_swaps(r_logits, del_s, dup_s):
    device = del_s.device
    r_logits, del_s, dup_s = r_logits.cpu(), del_s.cpu(), dup_s.cpu()
    m = del_s.shape[0]
    r = int(r_logits.argmax())
    deleted = torch.zeros(m, dtype=torch.bool)
    cnt = torch.zeros(m, dtype=torch.long)
    deleted[torch.argsort(del_s, descending=True)[:r]] = True
    for _ in range(r):                                  # best next copy among points below L
        s = _next_copy(dup_s, cnt).masked_fill(deleted | (cnt >= dup_s.shape[-1]), NEG)
        cnt[int(s.argmax())] += 1
    return ((~deleted).float() + cnt.float()).to(device)


def _gae(recs, gamma, lam):
    adv, last = [0.0] * len(recs), 0.0
    for i in reversed(range(len(recs))):
        v_next = recs[i + 1]["v"] if i + 1 < len(recs) else 0.0   # terminal V = 0
        last = recs[i]["reward"] + gamma * v_next - recs[i]["v"] + gamma * lam * last
        adv[i] = last
    for i, rec in enumerate(recs):
        rec["adv"], rec["ret"] = adv[i], adv[i] + rec["v"]
    return recs


class SwapPolicy(nn.Module):
    """Actor only: the swap-policy heads, no value head."""
    def __init__(self, d, T, max_r, hidden=(128, 128), L=2):
        super().__init__()
        self.T, self.max_r = T, max_r
        h = hidden[-1]
        self.torso = _mlp([3 * d + T, *hidden])   # theta, error, sum_j phi_j, t one-hot
        self.point_enc = _mlp([d + 1, *hidden])
        self.del_head = _mlp([2 * h, *hidden, 1])
        self.dup_head = _mlp([2 * h, *hidden, L - 1])    # one score per extra copy
        self.r_head = _mlp([2 * h, *hidden, max_r + 1])

    def forward(self, theta, err, t, phi, resid):
        t1h = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, phi.sum(1), t1h], dim=-1))
        h_p = self.point_enc(torch.cat([phi, resid[..., None]], dim=-1))
        h_sp = torch.cat([h_s[:, None, :].expand_as(h_p), h_p], dim=-1)
        pooled = torch.cat([h_s, h_p.mean(1)], dim=-1)
        return (self.r_head(pooled), self.del_head(h_sp).squeeze(-1),
                self.dup_head(h_sp))


class PriorAllocationPolicy(AllocationPolicy):
    """Allocation policy on the separable one-step scores: psi = -beta * score + learned correction."""
    def __init__(self, d, T, K, hidden=(128, 128), L=2, rho=0.0, beta=1.0):
        super().__init__(d, T, K, hidden, L)
        self.rho = rho
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        for head in (self.level_head, self.card_head):   # correction starts at zero: mode = the separable prior
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, theta, err, t, phi, resid):
        gamma, psi = super().forward(theta, err, t, phi, resid)
        p = _myopic_scores(err, phi, self.rho, self.L)
        return gamma, psi - self.log_beta.exp() * (p - p[..., 1:2])     # keep stays pinned at 0


class StateValue(nn.Module):
    def __init__(self, d, T, hidden=(128, 128)):
        super().__init__()
        self.T = T
        h = hidden[-1]
        self.torso = _mlp([3 * d + T, *hidden])   # theta, error, sum_j phi_j, t one-hot
        self.point_enc = _mlp([d + 1, *hidden])
        self.head = _mlp([2 * h, *hidden, 1])

    def forward(self, theta, err, t, phi, resid):
        t1h = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, phi.sum(1), t1h], dim=-1))
        h_p = self.point_enc(torch.cat([phi, resid[..., None]], dim=-1))
        return self.head(torch.cat([h_s, h_p.mean(1)], dim=-1)).squeeze(-1)

def _cat_kl(old_logits, new_logits, mask=None):
    """KL(old || new) for a batch of categoricals, optionally masked to a subset."""
    if mask is not None:
        old_logits = old_logits.masked_fill(~mask, NEG)
        new_logits = new_logits.masked_fill(~mask, NEG)
    lp_old = torch.log_softmax(old_logits, -1)
    lp_new = torch.log_softmax(new_logits, -1)
    return (lp_old.exp() * (lp_old - lp_new)).sum(-1)


def kl_swaps(old, new, del_idx, dup_idx, max_r):
    """Single-sample KL estimate along the stored path. old/new = (r_logits, del_s, dup_s)."""
    o_rl, o_ds, o_us = old
    n_rl, n_ds, n_us = new
    B, m = o_ds.shape
    kl = _cat_kl(o_rl, n_rl)
    cap = o_us.shape[-1]
    for copy_phase, active, _, avail, cnt in _replay(del_idx, dup_idx, max_r, m, cap):
        o_s, n_s = (_next_copy(o_us, cnt), _next_copy(n_us, cnt)) if copy_phase else (o_ds, n_ds)
        step = _cat_kl(o_s, n_s, mask=avail)
        kl = kl + torch.where(active, step, torch.zeros_like(step))
    return kl


def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _flat_grad(y, params, retain=False, create=False):
    """Flat gradient of y w.r.t. params, treating unused parameters as zero-gradient."""
    g = torch.autograd.grad(y, params, retain_graph=retain, create_graph=create,
                            allow_unused=True)
    return _flat([torch.zeros_like(p) if gi is None else gi
                  for p, gi in zip(params, g)])


def _set_flat(params, flat):
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    """Solve A x = b for x, with A available only as a matrix-vector product."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        if float(rr_new) < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


class TRPOAgent:
    """Deterministic (greedy swap read) for evaluation; stochastic during collection."""

    def __init__(self, policy, env, cfg, rng, deterministic=True):
        self.policy, self.env, self.cfg, self.rng = policy, env, cfg, rng
        self.deterministic = deterministic
        self.actor = "alloc" if isinstance(policy, AllocationPolicy) else "swap"
        self.max_r = max_swaps(env.m, cfg.B, cfg.L)

    @torch.no_grad()
    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.B,
                                          self.cfg.L, device=z[0].device)
        err, phi, resid = _features(self.env, theta, z)
        if self.actor == "alloc":
            gamma, psi = self.policy(theta[None], err[None],
                                     torch.tensor([t], device=theta.device), phi[None], resid[None])
            if self.deterministic:
                a = alloc_mode(psi, gamma)[0]
            else:
                a = alloc_sample(psi, gamma, self.rng)[0]
            return a.float().to(theta.device)
        rl, ds, us = self.policy(theta[None], err[None], torch.tensor([t], device=theta.device),
                                 phi[None], resid[None])
        if self.deterministic:
            return greedy_swaps(rl[0], ds[0], us[0])
        return sample_swaps(rl[0], ds[0], us[0], self.max_r, self.rng)[0]


def _rollout_trpo(env, policy, critic, cfg, rng, max_r, actor="swap"):
    recs, total = [], 0.0
    theta = env.theta_0.clone()
    for t in range(env.T):
        z = env.stream.sample_batch(env.m, rng)
        err, phi, resid = _features(env, theta, z)
        tt = torch.tensor([t], device=theta.device)
        with torch.no_grad():
            out = policy(theta[None], err[None], tt, phi[None], resid[None])
            v = critic(theta[None], err[None], tt, phi[None], resid[None])
        if actor == "alloc":
            gamma, psi = out
            a_long = alloc_sample(psi, gamma, rng)[0]                     # CPU, long
            logp = alloc_log_prob(psi.cpu(), gamma.cpu(), a_long[None])[0]
            a = a_long.float().to(theta.device)
            drawn = {"a": a_long.tolist()}
        else:
            rl, ds, us = out
            a, logp, r, di, ui = sample_swaps(rl[0], ds[0], us[0], max_r, rng)
            pad = lambda idx: idx + [-1] * (max_r - len(idx))
            drawn = {"r": r, "del": pad(di), "dup": pad(ui)}
        theta_next = env.victim.update(theta, z, a)
        c = env.cost(theta, z, a, theta_next)
        recs.append({"theta": theta.detach(), "err": err.detach(), "t": t,
                     "phi": phi.detach(), "resid": resid.detach(), **drawn,
                     "logp": float(logp), "v": float(v), "reward": -c})
        total += c
        theta = theta_next.detach()
    return recs, total


def train_trpo(env, cfg, seed=0, verbose=True, return_history=False,
               best_checkpoint=False, eval_episodes=100, eval_seed=999):
    """Train a TRPO swap-policy. Signature matches train_rfevi / train_bdq."""
    rng = torch.Generator().manual_seed(seed)
    actor = getattr(cfg, "trpo_actor", "alloc")
    max_r = max_swaps(env.m, cfg.B, cfg.L)      # K: cap on deletions, same for both actors
    if cfg.myopic_prior and actor != "alloc":
        raise ValueError("myopic_prior needs trpo_actor='alloc' (the swap actor has no per-level scores)")
    if cfg.myopic_prior:
        policy = PriorAllocationPolicy(env.d, env.T, max_r, cfg.hidden, cfg.L, env.rho,
                                       cfg.trpo_prior_beta).to(env.device)
    elif actor == "alloc":
        policy = AllocationPolicy(env.d, env.T, max_r, cfg.hidden, cfg.L).to(env.device)
    else:
        policy = SwapPolicy(env.d, env.T, max_r, cfg.hidden, cfg.L).to(env.device)
    critic = StateValue(env.d, env.T, cfg.hidden).to(env.device)
    vopt = torch.optim.Adam(critic.parameters(), lr=cfg.trpo_vf_lr,
                            weight_decay=cfg.weight_decay)
    params = list(policy.parameters())
    history, best_cost, best_state = [], float("inf"), None
    if cfg.myopic_prior and best_checkpoint:          # untrained = the separable prior: the first checkpoint
        best_cost = _mean_cost(env, TRPOAgent(policy, env, cfg, rng, deterministic=True),
                               eval_seed, eval_episodes)
        best_state = copy.deepcopy(policy.state_dict())

    for k in range(cfg.K_iters):
        batch = []
        for _ in range(cfg.N_episodes):
            recs, _ = _rollout_trpo(env, policy, critic, cfg, rng, max_r, actor)
            batch.extend(_gae(recs, cfg.gamma, cfg.gae_lambda))

        th_b = torch.stack([r["theta"] for r in batch])
        er_b = torch.stack([r["err"] for r in batch])
        dev = env.device
        t_b = torch.tensor([r["t"] for r in batch], device=dev)
        ph_b = torch.stack([r["phi"] for r in batch])
        rs_b = torch.stack([r["resid"] for r in batch])
        if actor == "alloc":
            a_b = torch.tensor([r["a"] for r in batch], dtype=torch.long, device=dev)
        else:
            r_b = torch.tensor([r["r"] for r in batch], device=dev)
            di_b = torch.tensor([r["del"] for r in batch], device=dev)
            ui_b = torch.tensor([r["dup"] for r in batch], device=dev)
        lp_old = torch.tensor([r["logp"] for r in batch], device=dev)
        adv = torch.tensor([r["adv"] for r in batch], dtype=torch.float32, device=dev)
        ret = torch.tensor([r["ret"] for r in batch], dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        with torch.no_grad():                      # frozen reference distribution
            old = tuple(x.detach() for x in
                        policy(th_b, er_b, t_b, ph_b, rs_b))

        def surrogate():
            new = policy(th_b, er_b, t_b, ph_b, rs_b)
            if actor == "alloc":
                gamma, psi = new
                logp = alloc_log_prob(psi, gamma, a_b)
            else:
                logp = logp_swaps(*new, r_b, di_b, ui_b, max_r)
            return (torch.exp(logp - lp_old) * adv).mean()

        def mean_kl():
            new = policy(th_b, er_b, t_b, ph_b, rs_b)
            if actor == "alloc":
                return alloc_kl(old, new).mean()        # old = (gamma, psi), frozen
            return kl_swaps(old, new, di_b, ui_b, max_r).mean()

        # --- natural gradient direction ---
        loss = surrogate()
        g = _flat_grad(loss, params, retain=True).detach()
        if float(g.norm()) < 1e-8:                  # nothing to do this iteration
            step_ok = False
        else:
            def Fvp(v):
                kl = mean_kl()
                gk = _flat_grad(kl, params, retain=True, create=True)
                return _flat_grad((gk * v).sum(), params, retain=True).detach() \
                    + cfg.trpo_damping * v

            stepdir = conjugate_gradient(Fvp, g, iters=cfg.trpo_cg_iters)
            shs = 0.5 * (stepdir @ Fvp(stepdir))
            step = stepdir * torch.sqrt(cfg.trpo_delta / (shs + 1e-8))

            # --- backtracking line search on (improvement, KL) ---
            old_params = _flat([p.data for p in params]).clone()
            old_surr = float(loss.detach())
            step_ok = False
            for j in range(cfg.trpo_backtracks):
                _set_flat(params, old_params + (0.5 ** j) * step)
                with torch.no_grad():
                    new_surr, kl_now = float(surrogate()), float(mean_kl())
                if kl_now <= cfg.trpo_delta * 1.5 and new_surr > old_surr:
                    step_ok = True
                    break
            if not step_ok:
                _set_flat(params, old_params)       # reject: stay inside the region

        # --- fit the critic (plain regression, outside the trust region) ---
        for _ in range(cfg.trpo_vf_iters):
            perm = torch.randperm(len(batch), generator=rng)[:cfg.trpo_minibatch].to(dev)
            v = critic(th_b[perm], er_b[perm], t_b[perm], ph_b[perm], rs_b[perm])
            vloss = ((v - ret[perm]) ** 2).mean()
            vopt.zero_grad()
            vloss.backward()
            vopt.step()

        if verbose or return_history or best_checkpoint:
            agent = TRPOAgent(policy, env, cfg, rng, deterministic=True)
            mean_cost = evaluate(env, agent, torch.Generator().manual_seed(eval_seed),
                                 n_episodes=eval_episodes)
            history.append(mean_cost)
            if best_checkpoint and mean_cost < best_cost:
                best_cost, best_state = mean_cost, copy.deepcopy(policy.state_dict())
            if verbose:
                print(f"[TRPO-{actor}] iter {k:2d}  step={'ok ' if step_ok else 'rej'}  "
                      f"mean_cost={mean_cost:.4f}")

    if best_checkpoint and best_state is not None:
        policy.load_state_dict(best_state)
    return (policy, history) if return_history else policy
