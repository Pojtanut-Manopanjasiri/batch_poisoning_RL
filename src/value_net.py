import torch
import torch.nn as nn


class ValueNet(nn.Module):
    # time_encoding: "elapsed" = t/T in [0,1] | "remaining" = (T-t) in [-1,1] (Pardo 2018)
    def __init__(self, d, T, hidden=(128, 128), time_encoding="elapsed"):
        super().__init__()
        self.T = T
        self.time_encoding = time_encoding
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

        if self.time_encoding == "remaining":
            feat = 2.0 * (self.T - t) / self.T - 1.0     # +1 at t=0 -> -1 at t=T
        else:
            feat = t / self.T
        x = torch.cat([theta, feat.unsqueeze(-1)], dim=-1)
        v = self.net(x).squeeze(-1)          # (N,)
        v = v * (t < self.T).float()         # terminal V(., T) = 0
        return v.squeeze(0) if single else v
