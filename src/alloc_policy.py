"""Exact constrained-allocation policy for TRPO (replacement for the swap actor in trpo.py).

Policy (a Gibbs / conditional-Poisson-type distribution restricted to the feasible set):

    pi(a | s) = exp( sum_i psi_i(a_i) + gamma(z(a)) ) * 1{a in A} / Z(s),

    A = { a in {0,...,L}^m : sum_i a_i = m,  z(a) <= K },   z(a) := #{i : a_i = 0},  K := floor(B/2).

(On {sum a = m}, ||a - 1||_1 = 2 z(a), so z(a) <= K is exactly the paper's budget constraint.)

psi : (N, m, L+1)  per-sample, per-level scores   (psi_i(1) = 0 for identifiability)
gamma: (N, K+1)    cardinality potential on the number of deletions (plays the role of r_head)

Z, exact sampling, exact log-prob, exact marginals, exact KL and the exact mode are all
computed by a dynamic program over (prefix sum n, prefix #zeros z):  O(m * m * K * (L+1)).
"""
import torch
import torch.nn as nn

NEG = -1e9


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- DP core
def _forward(psi, K, op="logsumexp"):
    """alphas[i][:, n, z] = op over prefixes a_{1:i} with sum = n and #zeros = z of sum_j psi_j(a_j)."""
    N, m, L1 = psi.shape
    a0 = psi.new_full((N, m + 1, K + 1), NEG)
    a0[:, 0, 0] = 0.0
    alphas = [a0]
    for i in range(m):
        prev, terms = alphas[-1], []
        for l in range(L1):
            dz = 1 if l == 0 else 0
            sh = psi.new_full((N, m + 1, K + 1), NEG)
            if l <= m:
                sh[:, l:, dz:] = prev[:, : m + 1 - l, : K + 1 - dz]
            terms.append(sh + psi[:, i, l][:, None, None])
        t = torch.stack(terms, 0)
        alphas.append(torch.logsumexp(t, 0) if op == "logsumexp" else t.max(0).values)
    return alphas


def log_partition(psi, gamma):
    K = gamma.shape[-1] - 1
    m = psi.shape[1]
    return torch.logsumexp(_forward(psi, K)[-1][:, m, :] + gamma, -1)


def _zeros(a):
    return (a == 0).sum(-1)


def log_prob(psi, gamma, a):
    """Exact log pi(a|s) for a batch of feasible integer actions a: (N, m) long."""
    s = psi.gather(-1, a[..., None]).squeeze(-1).sum(-1)
    s = s + gamma.gather(-1, _zeros(a)[:, None]).squeeze(-1)
    return s - log_partition(psi, gamma)


@torch.no_grad()
def _backtrack(psi, gamma, alphas, rng=None):
    """Backward pass: sample (rng given) or argmax (rng None) one action per row."""
    N, m, L1 = psi.shape
    K = gamma.shape[-1] - 1
    rows = torch.arange(N, device=psi.device)
    final = alphas[m][:, m, :] + gamma
    z = (torch.multinomial(torch.softmax(final, -1), 1, generator=rng).squeeze(1)
         if rng is not None else final.argmax(-1))
    n = torch.full((N,), m, dtype=torch.long, device=psi.device)
    a = torch.zeros(N, m, dtype=torch.long, device=psi.device)
    for i in reversed(range(m)):
        prev, w = alphas[i], []
        for l in range(L1):
            nl, zl = n - l, z - (1 if l == 0 else 0)
            ok = (nl >= 0) & (zl >= 0)
            v = prev[rows, nl.clamp(min=0), zl.clamp(min=0)] + psi[:, i, l]
            w.append(torch.where(ok, v, torch.full_like(v, NEG)))
        w = torch.stack(w, 1)
        l = (torch.multinomial(torch.softmax(w, -1), 1, generator=rng).squeeze(1)
             if rng is not None else w.argmax(-1))
        a[:, i] = l
        n = n - l
        z = z - (l == 0).long()
    assert bool((n == 0).all()) and bool((z == 0).all())
    return a


@torch.no_grad()
def sample(psi, gamma, rng):
    """Exact sample from pi(.|s) (forward filtering, backward sampling). rng: CPU generator."""
    psi, gamma = psi.cpu(), gamma.cpu()
    K = gamma.shape[-1] - 1
    return _backtrack(psi, gamma, _forward(psi, K), rng)


@torch.no_grad()
def mode(psi, gamma):
    """Exact argmax_{a in A} of sum psi_i(a_i) + gamma(z(a)) (max-plus DP; no convexity needed)."""
    K = gamma.shape[-1] - 1
    return _backtrack(psi, gamma, _forward(psi, K, op="max"), rng=None)


def marginals(psi, gamma):
    """mu[:, i, l] = P(a_i = l), nu[:, z] = P(z(a) = z), logZ. Via d logZ / d(psi, gamma)."""
    with torch.enable_grad():
        p = psi.detach().clone().requires_grad_(True)
        g = gamma.detach().clone().requires_grad_(True)
        logZ = log_partition(p, g)
        mu, nu = torch.autograd.grad(logZ.sum(), [p, g])
    return mu, nu, logZ.detach()


def kl(old, new):
    """Exact KL(pi_old || pi_new) per state; differentiable in `new` = (gamma, psi)."""
    g_o, p_o = old
    g_n, p_n = new
    mu, nu, logZ_o = marginals(p_o, g_o)
    lin = (mu * (p_o.detach() - p_n)).sum((1, 2)) + (nu * (g_o.detach() - g_n)).sum(-1)
    return lin - logZ_o + log_partition(p_n, g_n)


def entropy(psi, gamma):
    mu, nu, logZ = marginals(psi, gamma)
    return logZ - (mu * psi).sum((1, 2)) - (nu * gamma).sum(-1)


# ----------------------------------------------------------------------------- actor
class AllocationPolicy(nn.Module):
    """Same encoder as SwapPolicy; heads output (gamma, psi) instead of (r, delete, duplicate)."""

    def __init__(self, d, T, K, hidden=(128, 128), L=2):
        super().__init__()
        self.T, self.K, self.L = T, K, L
        h = hidden[-1]
        self.torso = _mlp([3 * d + T, *hidden])            # theta, error, clean step sum_j phi_j, t one-hot
        self.point_enc = _mlp([d + 1, *hidden])
        self.level_head = _mlp([2 * h, *hidden, L])        # levels {0, 2, ..., L}; level 1 pinned to 0
        self.card_head = _mlp([2 * h, *hidden, K + 1])     # gamma(z), z = 0..K

    def forward(self, theta, err, t, phi, resid):
        t1h = torch.nn.functional.one_hot(t.clamp(max=self.T - 1), self.T).float()
        h_s = self.torso(torch.cat([theta, err, phi.sum(1), t1h], dim=-1))
        h_p = self.point_enc(torch.cat([phi, resid[..., None]], dim=-1))
        h_sp = torch.cat([h_s[:, None, :].expand_as(h_p), h_p], dim=-1)
        out = self.level_head(h_sp)                         # (N, m, L)
        psi = torch.cat([out[..., :1], torch.zeros_like(out[..., :1]), out[..., 1:]], -1)
        gamma = self.card_head(torch.cat([h_s, h_p.mean(1)], dim=-1))
        return gamma, psi
