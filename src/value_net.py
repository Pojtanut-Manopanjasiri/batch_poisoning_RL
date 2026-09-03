"""The learned value function V_hat(theta, t)  (Brief II, sec. 4.1).

V_hat is a function of theta only (the batch z is exogenous and integrated out).
Time enters as an input feature -- a single network, not T separate ones.
Terminal condition V_hat(., T) = 0 is enforced by masking the output.
"""
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, d, T, hidden=(128, 128)):
        super().__init__()
        self.T = T
        # Smooth activation (Tanh, not ReLU): the decomposition's 2nd-order term
        # needs a meaningful Hessian.  A piecewise-linear net has H = 0 a.e., which
        # kills the Hessian part of B and breaks the O(eta^3) remainder.
        layers, in_dim = [], d + 1          # +1 for the (normalised) time feature
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, theta, t):
        """theta: (d,) or (N, d);  t: python int or (N,) tensor. Returns scalar or (N,)."""
        single = theta.dim() == 1
        if single:
            theta = theta.unsqueeze(0)
        N = theta.shape[0]

        if not torch.is_tensor(t):
            t = torch.full((N,), float(t), device=theta.device)
        else:
            t = t.float().reshape(-1)
            if t.numel() == 1:
                t = t.expand(N)

        x = torch.cat([theta, (t / self.T).unsqueeze(-1)], dim=-1)
        v = self.net(x).squeeze(-1)          # (N,)
        v = v * (t < self.T).float()         # terminal V(., T) = 0
        return v.squeeze(0) if single else v
