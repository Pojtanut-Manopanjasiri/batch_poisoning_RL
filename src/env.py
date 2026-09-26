import dataclasses
import functools

import torch


class Stream:
    def __init__(self, cfg):
        self.d = cfg.d
        self.noise = cfg.noise
        self.task = cfg.task
        self.device = torch.device(cfg.device)
        g = torch.Generator().manual_seed(cfg.seed)
        self.theta_star = torch.randn(cfg.d, generator=g)
        if cfg.theta_star_norm is not None:                    
            self.theta_star *= cfg.theta_star_norm / self.theta_star.norm()

    def sample_batch(self, m, rng):
        x, y = self._sample_cpu(m, rng)
        return (x.to(self.device), y.to(self.device))

    def _sample_cpu(self, m, rng):
        x = torch.randn(m, self.d, generator=rng)
        if self.task == "logistic":
            p = torch.sigmoid(x @ self.theta_star)
            y = (torch.rand(m, generator=rng) < p).float()
            if self.noise > 0:                                   
                flip = torch.rand(m, generator=rng) < self.noise
                y = torch.where(flip, 1.0 - y, y)
        elif self.task == "mlp":
            y = torch.sin(x @ self.theta_star) + self.noise * torch.randn(m, generator=rng)
        else:
            y = x @ self.theta_star + self.noise * torch.randn(m, generator=rng)
        return (x, y)


class DatasetStream:
    """Batches drawn uniformly, with replacement, from a fixed standardised dataset."""

    def __init__(self, cfg):
        from datasets import load_dataset
        X, y = load_dataset(cfg.dataset, cfg.dataset_n, cfg.dataset_noise, seed=cfg.seed,
                            pca=cfg.dataset_pca)
        self.X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.y = (y > 0).float() if cfg.task == "logistic" else y  
        self.d, self.noise, self.task = self.X.shape[1], cfg.noise, cfg.task
        self.device = torch.device(cfg.device)
        self.theta_star = torch.linalg.lstsq(self.X, y[:, None]).solution[:, 0]  

    def sample_batch(self, m, rng):
        i = torch.randint(0, len(self.y), (m,), generator=rng)
        return (self.X[i].to(self.device), self.y[i].to(self.device))


class Victim:
    def __init__(self, cfg):
        self.eta = cfg.eta
        self.task = cfg.task

    def residual(self, theta, z):
        x, y = z
        pred = x @ theta
        if self.task == "logistic":
            pred = torch.sigmoid(pred)
        return pred - y

    def phi_batch(self, theta, z):
        x, _ = z
        m = x.shape[0]
        resid = self.residual(theta, z) 
        grad = resid[:, None] * x        
        return -(self.eta / m) * grad  

    def update(self, theta, z, a):
        Phi = self.phi_batch(theta, z)
        return theta + a @ Phi         


class MLPVictim:
    def __init__(self, cfg):
        self.eta, self.d_in, self.h = cfg.eta, cfg.d, cfg.victim_hidden
        self.shapes = [(self.h, self.d_in), (self.h,), (self.h,), ()]   
        self.sizes = [self.h * self.d_in, self.h, self.h, 1]
        self.D = sum(self.sizes)
        self.task = "mlp"

    def unflatten(self, theta):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[i:i + n].reshape(s) if s else theta[i])
            i += n
        return out                                 

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
            theta.detach(), x, y)                   
        return -(self.eta / m) * g

    def update(self, theta, z, a):
        return theta + a @ self.phi_batch(theta, z)


@functools.cache
def _use_stock_cuda_kernels():
    try:
        from torch._native.registry import deregister_op_overrides
    except ImportError:                      
        return
    deregister_op_overrides(disable_dsl_names="triton")


class Environment:
    """Bundles stream + victim + attacker target and exposes cost / reset."""

    def __init__(self, env_cfg, method_cfg):
        if env_cfg.dataset is not None:  
            from datasets import n_features
            env_cfg = dataclasses.replace(env_cfg, d=n_features(env_cfg.dataset, env_cfg.dataset_pca))
        self._env_cfg = env_cfg         
        self.stream = DatasetStream(env_cfg) if env_cfg.dataset is not None else Stream(env_cfg)
        g = torch.Generator().manual_seed(env_cfg.seed + 1)
        if env_cfg.task == "mlp":
            self.victim = MLPVictim(env_cfg)
            self.d = self.victim.D                     
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
        self.rho = method_cfg.lam / (2 * env_cfg.m)     

    def cost(self, theta, z, a, theta_next):
        edits = torch.abs(a - 1.0).sum()
        nefarious = 0.5 * ((theta_next - self.theta_target) ** 2).sum()
        return float(self.rho * edits + nefarious)

    def cost_batch(self, a, theta_next):
        """cost for S transitions: a (S, m), theta_next (S, d) -> (S,)."""
        edits = torch.abs(a - 1.0).sum(1)
        nefarious = 0.5 * ((theta_next - self.theta_target) ** 2).sum(1)
        return self.rho * edits + nefarious


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

    k = m if B is None else B // 2      
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
