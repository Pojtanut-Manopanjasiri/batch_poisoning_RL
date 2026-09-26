import copy
import warnings

import torch
import torch.nn as nn

from bdq import _features
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


def max_swaps(m, B=None):
    r = m // 2
    return r if B is None else min(r, B // 2)


def _draw(scores, avail, rng):
    """One masked-softmax draw without replacement. Returns (index, logprob)."""
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

    avail = torch.ones(m, dtype=torch.bool)
    del_idx, dup_idx = [], []
    for _ in range(r):                                  
        i, lp = _draw(del_s, avail, rng)
        del_idx.append(i); avail[i] = False; logp = logp + lp
    for _ in range(r):                                
        i, lp = _draw(dup_s, avail, rng)
        dup_idx.append(i); avail[i] = False; logp = logp + lp

    a = torch.ones(m)
    for i in del_idx:
        a[i] = 0.0
    for i in dup_idx:
        a[i] = 2.0
    return a.to(device), logp, r, del_idx, dup_idx


def logp_swaps(r_logits, del_s, dup_s, r, del_idx, dup_idx, max_r):
    """Exact log-prob of stored ordered draws"""
    B, m = del_s.shape
    logp = torch.log_softmax(r_logits, -1).gather(1, r[:, None]).squeeze(1)
    avail = torch.ones(B, m, dtype=torch.bool, device=del_s.device)
    for scores, idx in ((del_s, del_idx), (dup_s, dup_idx)):
        for j in range(max_r):
            active = idx[:, j] >= 0                    
            if not bool(active.any()):
                continue
            lp = torch.log_softmax(scores.masked_fill(~avail, NEG), dim=-1)
            pick = idx[:, j].clamp(min=0)
            step = lp.gather(1, pick[:, None]).squeeze(1)
            logp = logp + torch.where(active, step, torch.zeros_like(step))
            hit = torch.nn.functional.one_hot(pick, m).bool() & active[:, None]
            avail = avail & ~hit
    return logp


def greedy_swaps(r_logits, del_s, dup_s):
    device = del_s.device
    r_logits, del_s, dup_s = r_logits.cpu(), del_s.cpu(), dup_s.cpu()
    m = del_s.shape[0]
    r = int(r_logits.argmax())
    a = torch.ones(m)
    if r == 0:
        return a.to(device)
    order_del = torch.argsort(del_s, descending=True)
    dels = order_del[:r]
    mask = torch.ones(m, dtype=torch.bool)
    mask[dels] = False
    cand = torch.where(mask)[0]
    dups = cand[torch.argsort(dup_s[cand], descending=True)[:r]]
    a[dels] = 0.0
    a[dups] = 2.0
    return a.to(device)

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
    def __init__(self, d, T, max_r, hidden=(128, 128)):
        super().__init__()
        self.T, self.max_r = T, max_r
        h = hidden[-1]
        self.torso = _mlp([2 * d + T, *hidden])
        self.point_enc = _mlp([d + 1, *hidden])
        self.del_head = _mlp([2 * h, *hidden, 1])
        self.dup_head = _mlp([2 * h, *hidden, 1])
        self.r_head = _mlp([2 * h, *hidden, max_r + 1])

    def forward(self, theta, err, t, phi, resid):
        t1h = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, t1h], dim=-1))
        h_p = self.point_enc(torch.cat([phi, resid[..., None]], dim=-1))
        h_sp = torch.cat([h_s[:, None, :].expand_as(h_p), h_p], dim=-1)
        pooled = torch.cat([h_s, h_p.mean(1)], dim=-1)
        return (self.r_head(pooled), self.del_head(h_sp).squeeze(-1),
                self.dup_head(h_sp).squeeze(-1))


class StateValue(nn.Module):
    def __init__(self, d, T, hidden=(128, 128)):
        super().__init__()
        self.T = T
        h = hidden[-1]
        self.torso = _mlp([2 * d + T, *hidden])
        self.point_enc = _mlp([d + 1, *hidden])
        self.head = _mlp([2 * h, *hidden, 1])

    def forward(self, theta, err, t, phi, resid):
        t1h = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, t1h], dim=-1))
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
    avail = torch.ones(B, m, dtype=torch.bool, device=o_ds.device)
    for o_s, n_s, idx in ((o_ds, n_ds, del_idx), (o_us, n_us, dup_idx)):
        for j in range(max_r):
            active = idx[:, j] >= 0
            if not bool(active.any()):
                continue
            step = _cat_kl(o_s, n_s, mask=avail)
            kl = kl + torch.where(active, step, torch.zeros_like(step))
            pick = idx[:, j].clamp(min=0)
            hit = torch.nn.functional.one_hot(pick, m).bool() & active[:, None]
            avail = avail & ~hit
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
        self.max_r = max_swaps(env.m, cfg.B)

    @torch.no_grad()
    def act(self, theta, z, t, epsilon=0.0):
        m = z[0].shape[0]
        if epsilon > 0 and float(torch.rand(1, generator=self.rng)) < epsilon:
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.B,
                                          self.cfg.L, device=z[0].device)
        err, phi, resid = _features(self.env, theta, z)
        rl, ds, us = self.policy(theta[None], err[None], torch.tensor([t], device=theta.device),
                                 phi[None], resid[None])
        if self.deterministic:
            return greedy_swaps(rl[0], ds[0], us[0])
        return sample_swaps(rl[0], ds[0], us[0], self.max_r, self.rng)[0]


def _rollout_trpo(env, policy, critic, cfg, rng, max_r):
    recs, total = [], 0.0
    theta = env.theta_0.clone()
    for t in range(env.T):
        z = env.stream.sample_batch(env.m, rng)
        err, phi, resid = _features(env, theta, z)
        tt = torch.tensor([t], device=theta.device)
        with torch.no_grad():
            rl, ds, us = policy(theta[None], err[None], tt, phi[None], resid[None])
            v = critic(theta[None], err[None], tt, phi[None], resid[None])
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


def train_trpo(env, cfg, seed=0, verbose=True, return_history=False,
               best_checkpoint=False, eval_episodes=100, eval_seed=999):
    """Train a TRPO swap-policy. Signature matches train_rfevi / train_bdq."""
    rng = torch.Generator().manual_seed(seed)
    max_r = max_swaps(env.m, cfg.B)
    policy = SwapPolicy(env.d, env.T, max_r, cfg.hidden).to(env.device)
    critic = StateValue(env.d, env.T, cfg.hidden).to(env.device)
    vopt = torch.optim.Adam(critic.parameters(), lr=cfg.trpo_vf_lr,
                            weight_decay=cfg.weight_decay)
    params = list(policy.parameters())
    history, best_cost, best_state = [], float("inf"), None

    for k in range(cfg.K_iters):
        batch = []
        for _ in range(cfg.N_episodes):
            recs, _ = _rollout_trpo(env, policy, critic, cfg, rng, max_r)
            batch.extend(_gae(recs, cfg.gamma, cfg.gae_lambda))

        th_b = torch.stack([r["theta"] for r in batch])
        er_b = torch.stack([r["err"] for r in batch])
        dev = env.device
        t_b = torch.tensor([r["t"] for r in batch], device=dev)
        ph_b = torch.stack([r["phi"] for r in batch])
        rs_b = torch.stack([r["resid"] for r in batch])
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
            logp = logp_swaps(*new, r_b, di_b, ui_b, max_r)
            return (torch.exp(logp - lp_old) * adv).mean()

        def mean_kl():
            new = policy(th_b, er_b, t_b, ph_b, rs_b)
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
                print(f"[TRPO] iter {k:2d}  step={'ok ' if step_ok else 'rej'}  "
                      f"mean_cost={mean_cost:.4f}")

    if best_checkpoint and best_state is not None:
        policy.load_state_dict(best_state)
    return (policy, history) if return_history else policy
