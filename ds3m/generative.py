# Deviations from the paper (arXiv v3), following the official code (src/DSSSMCode.py):
# 1. Gamma = 0.5 * softmax(Linear(T0)) + 0.5 * I, so every diagonal entry is >= 0.5.
# 2. Networks output a standard deviation via Softplus, not log Sigma (eqs. 5, 7).
# 3. Each network is a shared 2-layer ReLU trunk followed by separate mean and std heads.

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Eq. (3): h_t = f_h(h_{t-1}, x_t), with f_h a 1-layer GRU."""

    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=x_dim, hidden_size=h_dim, batch_first=True)

    def forward(self, x):
        h, _ = self.gru(x)
        return h

class GaussianNet(nn.Module):
    """Shared 2-layer ReLU trunk, then a mean head and a Softplus std head."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(out_dim, out_dim)
        self.std = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())

    def forward(self, inp):
        hidden = self.trunk(inp)
        return self.mean(hidden), self.std(hidden)

class Transition(nn.Module):
    """Official-code Gamma: 0.5 * softmax(Linear(T0)) + 0.5 * I."""

    def __init__(self, K):
        super().__init__()
        T0 = torch.eye(K) * (1 - 0.05 * K) + 0.05 * torch.ones(K, K)
        self.register_buffer("T0", T0)
        self.register_buffer("I", torch.eye(K))
        self.dprior = nn.Linear(K, K)

    def forward(self):
        return 0.5 * torch.softmax(self.dprior(self.T0), dim=1) + 0.5 * self.I


class Generative(nn.Module):
    """Eqs. (4)-(7) with official-code networks: one GaussianNet per regime."""

    def __init__(self, z_dim, h_dim, y_dim, K):
        super().__init__()
        self.K = K
        self.f_z = nn.ModuleList([GaussianNet(z_dim + h_dim, z_dim) for _ in range(K)])
        self.f_y = nn.ModuleList([GaussianNet(z_dim + h_dim, y_dim) for _ in range(K)])

    def transition(self, z_prev, h_t):
        inp = torch.cat([z_prev, h_t], dim=-1)
        outs = [self.f_z[k](inp) for k in range(self.K)]
        mu = torch.stack([o[0] for o in outs], dim=1)
        std = torch.stack([o[1] for o in outs], dim=1)
        return mu, std

    def emission(self, z_t, h_t):
        outs = [self.f_y[k](torch.cat([z_t[:, k], h_t], dim=-1)) for k in range(self.K)]
        mu = torch.stack([o[0] for o in outs], dim=1)
        std = torch.stack([o[1] for o in outs], dim=1)
        return mu, std


if __name__ == "__main__":
    x = torch.randn(64, 20, 1)
    encoder = Encoder(x_dim=1, h_dim=10)
    print("h shape:", encoder(x).shape)

    transition = Transition(K=2)
    print("Gamma:\n", transition())

    gen = Generative(z_dim=2, h_dim=10, y_dim=1, K=2)
    h_t = torch.randn(64, 10)
    mu_z, std_z = gen.transition(torch.randn(64, 2), h_t)
    print("Transition mean/std:", mu_z.shape, std_z.shape)
    mu_y, std_y = gen.emission(torch.randn(64, 2, 2), h_t)
    print("Emission mean/std:", mu_y.shape, std_y.shape)
    print("All stds positive:", bool((std_z > 0).all() and (std_y > 0).all()))