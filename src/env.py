import torch

class Stream:
    """iid. data stream P_Z: z = (x, y), x ~ N(0, I_d).

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
        g = torch.Generator().manual_seed(cfg.seed)
        self.theta_star = torch.randn(cfg.d, generator=g)

    def sample_batch(self, m, rng):
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
            # episode starts from a small random init.
            self.theta_0 = env_cfg.init_scale * torch.randn(self.d, generator=g)
           
            probe = torch.cat([self.victim.phi_batch(
                self.theta_0, self.stream.sample_batch(env_cfg.m, g)) for _ in range(2)])
            direction = torch.randn(probe.shape[0], generator=g) @ probe
            self.theta_target = self.theta_0 + env_cfg.target_scale * direction / direction.norm()
        else:
            self.victim = Victim(env_cfg)
            self.d = env_cfg.d
            self.theta_target = torch.randn(env_cfg.d, generator=g)
            self.theta_0 = torch.zeros(env_cfg.d)
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
def is_feasible(a, balance=True, k_max=None):
    n_del = int((a == 0).sum())
    n_dup = int((a == 2).sum())
    if balance and n_del != n_dup:
        return False
    if k_max is not None and n_del > k_max:
        return False
    return True


def random_feasible_action(m, rng, balance=True, k_max=None):
    """Draw a uniform-ish feasible action (used for exploration / the random agent)."""
    a = torch.ones(m)
    max_r = m // 2
    if k_max is not None:
        max_r = min(max_r, k_max)
    if max_r == 0:
        return a
    r = int(torch.randint(0, max_r + 1, (1,), generator=rng))
    if r == 0:
        return a
    perm = torch.randperm(m, generator=rng)
    a[perm[:r]] = 0.0        # delete r indices
    a[perm[r:2 * r]] = 2.0   # duplicate r other indices
    return a
