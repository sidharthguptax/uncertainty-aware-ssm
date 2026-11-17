# Deviations from the paper, following the official code:
# 4. A single z_t sample (drawn under the sampled d_t) feeds every regime's emission;
#    eq. (21) has a separate z_t per regime.
# 5. Initial conditions: z_0 = 0 and d_0 ~ Uniform, with steps starting at t = 1.

import math
import torch
import torch.nn as nn

from ds3m.generative import Encoder, Transition, Generative
from ds3m.inference import BackwardRNN, Posterior


def nll_gauss(mean, std, y):
    return 0.5 * math.log(2 * math.pi) + torch.log(std) + (y - mean) ** 2 / (2 * std ** 2)


def kl_gauss(mu_q, std_q, mu_p, std_p):
    return torch.log(std_p) - torch.log(std_q) + (std_q ** 2 + (mu_q - mu_p) ** 2) / (2 * std_p ** 2) - 0.5


def kl_cat(q, p):
    return torch.sum(q * torch.log(q / p), dim=-1)

class DS3M(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, z_dim, K, hidden_dim=None):
        super().__init__()
        self.K = K
        self.z_dim = z_dim
        self.encoder = Encoder(x_dim, h_dim)
        self.transition = Transition(K)
        self.generative = Generative(z_dim, h_dim, y_dim, K, hidden_dim)
        self.backward_rnn = BackwardRNN(y_dim, h_dim, h_dim)
        self.posterior = Posterior(z_dim, h_dim, K, hidden_dim)

    def forward(self, x, y):
        B, T, _ = x.shape
        idx = torch.arange(B)
        Gamma = self.transition()
        h = self.encoder(x)
        A = self.backward_rnn(y, h)

        d_prev = torch.randint(0, self.K, (B,))
        q_d_prev = torch.full((B, self.K), 1.0 / self.K)
        z_prev = torch.zeros(B, self.z_dim)

        nll, kl_z, kl_d = 0.0, 0.0, 0.0
        for t in range(T):
            q_d_all = self.posterior.d_posterior(A[:, t])
            q_d = q_d_all[idx, d_prev]

            mu_q, std_q = self.posterior.z_posterior(z_prev, A[:, t])
            mu_p, std_p = self.generative.transition(z_prev, h[:, t])

            d_t = torch.multinomial(q_d, 1).squeeze(1)
            z_t = mu_q[idx, d_t] + std_q[idx, d_t] * torch.randn(B, self.z_dim)

            z_rep = z_t.unsqueeze(1).expand(-1, self.K, -1)
            mu_y, std_y = self.generative.emission(z_rep, h[:, t])

            y_t = y[:, t].unsqueeze(1)
            nll = nll + (nll_gauss(mu_y, std_y, y_t).sum(-1) * q_d).sum()
            kl_z = kl_z + (kl_gauss(mu_q, std_q, mu_p, std_p).sum(-1) * q_d).sum()
            kl_d = kl_d + (kl_cat(q_d_all, Gamma.unsqueeze(0)) * q_d_prev).sum()

            d_prev, z_prev, q_d_prev = d_t, z_t, q_d
        self.d_last, self.z_last = d_t, z_t
        return nll, kl_z, kl_d

    @torch.no_grad()
    def forecast(self, x, y, S=200):
        B = x.shape[0]
        idx = torch.arange(B)
        Gamma = self.transition()
        h_next = self.encoder(torch.cat([x, y[:, -1:]], dim=1))[:, -1]

        y_samples, d_samples = [], []
        for _ in range(S):
            self(x, y)
            d_next = torch.multinomial(Gamma[self.d_last], 1).squeeze(1)
            mu_z, std_z = self.generative.transition(self.z_last, h_next)
            z_next = mu_z[idx, d_next] + std_z[idx, d_next] * torch.randn(B, self.z_dim)
            z_rep = z_next.unsqueeze(1).expand(-1, self.K, -1)
            mu_y, std_y = self.generative.emission(z_rep, h_next)
            y_next = mu_y[idx, d_next] + std_y[idx, d_next] * torch.randn_like(mu_y[idx, d_next])
            y_samples.append(y_next)
            d_samples.append(d_next)

        return torch.stack(y_samples, dim=1), torch.stack(d_samples, dim=1)

if __name__ == "__main__":
    torch.manual_seed(0)
    model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2)
    x = torch.randn(64, 20, 1)
    y = torch.randn(64, 20, 1)

    nll, kl_z, kl_d = model(x, y)
    print("NLL:", nll.item(), "| KL_z:", kl_z.item(), "| KL_d:", kl_d.item())

    loss = (nll + kl_z + kl_d) / (64 * 20)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    print("Parameters without gradient:", missing)