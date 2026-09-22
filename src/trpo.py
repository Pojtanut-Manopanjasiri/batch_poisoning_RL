"""TRPO baseline for batch-curation poisoning.

A policy over delete/duplicate SWAPS, in the coordinates the batch-size constraint is
written in.  This covers a_i in {0,1,2} only, i.e. L = 2:

    r          ~ Categorical(r_logits)                how many swaps, r <= max_r
    del_1..r   ~ sequential softmax over del_scores   r deletes, without replacement
    dup_1..r   ~ sequential softmax over dup_scores   r duplicates, disjoint from those

This Plackett-Luce draw gives feasibility BY CONSTRUCTION (no projection), an EXACT
log-prob that telescopes into a sum, and O(m) sampling from a space of size 3^m.

The obvious alternative -- a factored per-point categorical projected onto the feasible
set -- fails: with `balance`, flipping one point to "delete" does nothing until a matching
duplicate appears, so the projection erases single-coordinate changes and the policy
gradient sits on a plateau.

The update maximises an importance-weighted surrogate under a hard KL trust region:

    maximise  E[ pi_new(u|s)/pi_old(u|s) * A(s,u) ]   s.t.  KL(pi_old || pi_new) <= delta

solved by a natural-gradient step: conjugate gradient against the Fisher-vector product,
then a backtracking line search that enforces both the KL bound and actual improvement.
It never needs max_a Q(s,a) -- the critic is a state value V(s).

The KL for a Plackett-Luce policy
---------------------------------
The exact KL between two swap policies has no closed form: it is an expectation over
sequences, and each draw's categorical depends on which points earlier draws consumed.
We use the standard single-sample estimator -- replay the SAMPLED trajectory's masks and
sum the exact per-step categorical KLs along it:

    KL ~= KL(r_old || r_new)
          + sum_j KL(delete draw j | mask from the sampled path)
          + sum_j KL(duplicate draw j | mask from the sampled path)

Each term is exact given the path, so this is an unbiased estimate of the conditional KL,
and it is twice-differentiable -- which is what the Fisher-vector product needs.

Masked entries use a finite NEG (not -inf) so that p_old * (logp_old - logp_new) evaluates
to ~0 instead of nan.
"""
import copy
import warnings

import torch
import torch.nn as nn

from bdq import _features
from env import random_feasible_action
from fvi import evaluate


# --------------------------------------------------------------------------- #
# swap policy: feasible-by-construction Plackett-Luce over delete/duplicate pairs
# --------------------------------------------------------------------------- #
NEG = -1e9          # mask value; finite so log_softmax never yields nan


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
    masking used at sampling time, so the importance ratio is exact.
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


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #
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
    """Critic V(s); fitted by plain regression, outside the trust region."""

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


# --------------------------------------------------------------------------- #
# KL along the sampled path
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# flat-parameter helpers + conjugate gradient
# --------------------------------------------------------------------------- #
def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _flat_grad(y, params, retain=False, create=False):
    """Flat gradient of y w.r.t. params, treating unused parameters as zero-gradient.

    allow_unused is REQUIRED, not defensive: both logp_swaps and kl_swaps skip a draw
    index entirely when no row of the batch made that draw (`if not active.any()`).  If
    the policy collapses to all-keep -- every sampled r == 0, which is exactly what
    happens at large c_edit -- the delete/duplicate heads never enter the graph at all.
    Their gradient is then genuinely zero (nothing in the objective depends on them for
    this batch), so autograd's None must be read as 0 rather than raised as an error.
    """
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
            return random_feasible_action(m, self.rng, self.cfg.balance, self.cfg.B, self.cfg.L)
        err, phi, resid = _features(self.env, theta, z)
        rl, ds, us = self.policy(theta[None], err[None], torch.tensor([t]),
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
        tt = torch.tensor([t])
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
    """Train a TRPO swap-policy. Signature matches train_fvi / train_bdq.

    This is ON-policy: we match ENVIRONMENT INTERACTIONS (K_iters * N_episodes)
    with the replay-based learners and let gradient-step counts differ.
    """
    rng = torch.Generator().manual_seed(seed)
    if getattr(cfg, "L", 2) > 2:
        warnings.warn(f"TRPO's swap policy only reaches a_i <= 2; at L={cfg.L} it is a "
                      "restricted baseline and will look artificially weak.")
    max_r = max_swaps(env.m, cfg.B)
    policy = SwapPolicy(env.d, env.T, max_r, cfg.hidden)
    critic = StateValue(env.d, env.T, cfg.hidden)
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
        t_b = torch.tensor([r["t"] for r in batch])
        ph_b = torch.stack([r["phi"] for r in batch])
        rs_b = torch.stack([r["resid"] for r in batch])
        r_b = torch.tensor([r["r"] for r in batch])
        di_b = torch.tensor([r["del"] for r in batch])
        ui_b = torch.tensor([r["dup"] for r in batch])
        lp_old = torch.tensor([r["logp"] for r in batch])
        adv = torch.tensor([r["adv"] for r in batch], dtype=torch.float32)
        ret = torch.tensor([r["ret"] for r in batch], dtype=torch.float32)
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
            perm = torch.randperm(len(batch), generator=rng)[:cfg.trpo_minibatch]
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
