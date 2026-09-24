"""Environment and victim (reconstructed minimal Brief I).

State  s_t = (theta_t, z_t),  theta in R^d,  batch z of m i.i.d. points.
Action a in {0,...,L}^m  (0 delete, 1 keep, l>1 duplicate l-1 times).

Transition is action-linear (the whole decomposition rests on this):

    theta_{t+1} = theta_t + sum_i a_i * phi_i ,   phi_i = -(eta/m) grad_theta l_i .

Cost (minimisation, ZZL convention):

    g = c_edit * sum_i |a_i - 1|            (perturbation / total-variation)
      + (lam/2) * ||theta_{t+1} - theta_target||^2   (nefarious)
"""
import functools

import torch


class Stream:
    """I.i.d. data stream P_Z: z = (x, y), x ~ N(0, I_d).

    task="regression": y = <x, theta_star> + noise*N(0,1)      (real-valued)
    task="logistic"  : y ~ Bernoulli(sigmoid(<x, theta_star>)) (binary), then each
                       label is flipped with probability `noise`.
    task="mlp"       : y = sin(<x, theta_star>) + noise*N(0,1) (real-valued, non-linear
                       target -- gives the MLP victim something a linear model can't fit).
    """

    def __init__(self, cfg):
        self.d = cfg.d
        self.noise = cfg.noise
        self.task = cfg.task
        self.device = torch.device(cfg.device)
        g = torch.Generator().manual_seed(cfg.seed)
        self.theta_star = torch.randn(cfg.d, generator=g)

    def sample_batch(self, m, rng):
        x, y = self._sample_cpu(m, rng)
        return (x.to(self.device), y.to(self.device))

    def _sample_cpu(self, m, rng):
        x = torch.randn(m, self.d, generator=rng)
        if self.task == "logistic":
            p = torch.sigmoid(x @ self.theta_star)
            y = (torch.rand(m, generator=rng) < p).float()
            if self.noise > 0:                                   # symmetric label flips
                flip = torch.rand(m, generator=rng) < self.noise
                y = torch.where(flip, 1.0 - y, y)
        elif self.task == "mlp":
            y = torch.sin(x @ self.theta_star) + self.noise * torch.randn(m, generator=rng)
        else:
            y = x @ self.theta_star + self.noise * torch.randn(m, generator=rng)
        return (x, y)


class Victim:
    """Online mini-batch GD victim.  Per-point loss:

        regression: l_i = 1/2 (<theta,x_i> - y_i)^2          -> grad = (<theta,x_i> - y_i) x_i
        logistic  : l_i = BCE(sigmoid(<theta,x_i>), y_i)     -> grad = (sigmoid(<theta,x_i>) - y_i) x_i

    Both have gradient `resid_i * x_i` -- only the definition of the residual changes,
    so action-linearity (and hence the whole decomposition) is untouched.

    phi_batch returns Phi in R^{m x d} whose rows are phi_i (the per-point
    contribution to the update).  No clipping / normalisation: linearity must hold.
    """

    def __init__(self, cfg):
        self.eta = cfg.eta
        self.task = cfg.task

    def residual(self, theta, z):
        """(m,) per-point residual r_i; grad of l_i is r_i * x_i for both tasks."""
        x, y = z
        pred = x @ theta
        if self.task == "logistic":
            pred = torch.sigmoid(pred)
        return pred - y

    def phi_batch(self, theta, z):
        x, _ = z
        m = x.shape[0]
        resid = self.residual(theta, z)  # (m,)  residuals r_i
        grad = resid[:, None] * x        # (m, d) per-point gradient r_i * x_i
        return -(self.eta / m) * grad    # (m, d) = Phi, rows phi_i

    def update(self, theta, z, a):
        Phi = self.phi_batch(theta, z)
        return theta + a @ Phi           # theta + sum_i a_i phi_i


class MLPVictim:
    """Small neural-network victim: theta is the FLATTENED parameter vector.

        f(x; theta) = W2 tanh(W1 x + b1) + b2 ,   l_i = 1/2 (f(x_i) - y_i)^2

    Action-linearity is unchanged -- one SGD step on the reweighted batch loss is still
    theta + sum_i a_i phi_i with phi_i = -(eta/m) grad_theta l_i -- because the batch
    gradient is a sum over points whatever the model is. Only phi_batch gets harder: we
    need PER-SAMPLE gradients, which torch.func.vmap(grad(.)) gives in one pass.

    Note the parameter dimension D (not the data dimension d) is what theta lives in, so
    env.d == D for this task. Keep victim_hidden small: D enters the value network's input.
    """

    def __init__(self, cfg):
        self.eta, self.d_in, self.h = cfg.eta, cfg.d, cfg.victim_hidden
        self.shapes = [(self.h, self.d_in), (self.h,), (self.h,), ()]   # W1, b1, W2, b2
        self.sizes = [self.h * self.d_in, self.h, self.h, 1]
        self.D = sum(self.sizes)
        self.task = "mlp"

    def unflatten(self, theta):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[i:i + n].reshape(s) if s else theta[i])
            i += n
        return out                                   # W1 (h,d), b1 (h), W2 (h,), b2 ()

    def forward(self, theta, x):
        W1, b1, W2, b2 = self.unflatten(theta)
        return torch.tanh(x @ W1.T + b1) @ W2 + b2   # (m,)

    def residual(self, theta, z):
        x, y = z
        return self.forward(theta, x) - y

    def phi_batch(self, theta, z):
        x, y = z
        m = x.shape[0]

        def loss_one(th, xi, yi):
            return 0.5 * (self.forward(th, xi[None])[0] - yi) ** 2

        g = torch.func.vmap(torch.func.grad(loss_one), in_dims=(None, 0, 0))(
            theta.detach(), x, y)                    # (m, D) per-sample gradients
        return -(self.eta / m) * g

    def update(self, theta, z, a):
        return theta + a @ self.phi_batch(theta, z)


@functools.cache
def _use_stock_cuda_kernels():
    # torch sends the bmm inside vmap(grad)'s backward to a Triton kernel, which compiles
    # a C launcher on first use and fails on machines without a C compiler
    try:
        from torch._native.registry import deregister_op_overrides
    except ImportError:                      # torch without the Triton override layer
        return
    deregister_op_overrides(disable_dsl_names="triton")


class Environment:
    """Bundles stream + victim + attacker target and exposes cost / reset."""

    def __init__(self, env_cfg, method_cfg):
        self._env_cfg = env_cfg          # kept so diagnostics can rebuild with new eta
        self.stream = Stream(env_cfg)
        g = torch.Generator().manual_seed(env_cfg.seed + 1)
        if env_cfg.task == "mlp":
            self.victim = MLPVictim(env_cfg)
            self.d = self.victim.D                       # theta lives in parameter space
            # a zero start is degenerate for an MLP (symmetric, zero gradient), so the
            # episode starts from a small random init; the target uses a larger scale
            # so f(.;theta_target) is visibly non-linear (small weights keep tanh in
            # its near-linear regime).
            self.theta_0 = env_cfg.init_scale * torch.randn(self.d, generator=g)
            self.theta_target = env_cfg.target_scale * torch.randn(self.d, generator=g)
        else:
            self.victim = Victim(env_cfg)
            self.d = env_cfg.d
            self.theta_target = torch.randn(env_cfg.d, generator=g)
            self.theta_0 = torch.zeros(env_cfg.d)
        self.device = torch.device(env_cfg.device)
        if self.device.type == "cuda":
            _use_stock_cuda_kernels()
        self.theta_0 = self.theta_0.to(self.device)
        self.theta_target = self.theta_target.to(self.device)
        self.m, self.T = env_cfg.m, env_cfg.T
        self.lam = method_cfg.lam
        self.c_edit = method_cfg.c_edit

    def cost(self, theta, z, a, theta_next):
        edits = torch.abs(a - 1.0).sum()
        nefarious = 0.5 * self.lam * ((theta_next - self.theta_target) ** 2).sum()
        return float(self.c_edit * edits + nefarious)


# --------------------------------------------------------------------------- #
# Feasibility helpers (balance / budget)
# --------------------------------------------------------------------------- #
def is_feasible(a, balance=True, B=None, L=2):
    """a_i in {0,...,L};  sum_i a_i = m (balance);  ||a - 1||_1 <= B."""
    if bool((a < 0).any()) or bool((a > L).any()):
        return False
    if balance and int(a.sum()) != len(a):
        return False
    if B is not None and float((a - 1.0).abs().sum()) > B:
        return False
    return True


def random_feasible_action(m, rng, balance=True, B=None, L=2, device="cpu"):
    """Random feasible action via the Algorithm-2 allocation with random ties."""
    return _random_feasible_cpu(m, rng, balance, B, L).to(device)


def _random_feasible_cpu(m, rng, balance, B, L):
    if not balance:
        a = torch.ones(m)
        budget = m if B is None else B
        order = torch.randperm(m, generator=rng)
        for i in order:
            if budget <= 0:
                break
            lvl = int(torch.randint(0, L + 1, (1,), generator=rng))
            if abs(lvl - 1) <= budget:
                a[i] = float(lvl)
                budget -= abs(lvl - 1)
        return a

    k = m if B is None else B // 2       # ||a-1||_1 = 2D under sum a = m
    a = torch.zeros(m)
    D = 0
    for _ in range(m):
        elig = (a < L) & ((a == 0) | (D < k))
        idx = torch.where(elig)[0]
        i = int(idx[torch.randint(0, len(idx), (1,), generator=rng)])
        if a[i] >= 1:
            D += 1
        a[i] += 1
    return a
