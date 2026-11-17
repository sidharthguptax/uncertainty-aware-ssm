import torch
import torch.nn as nn
from ds3m.generative import GaussianNet


class BackwardRNN(nn.Module):
    """Eq. (15): A_t = g(A_{t+1}, [y_t, h_t]), a GRU run backwards in time."""

    def __init__(self, y_dim, h_dim, a_dim):
        super().__init__()
        self.gru = nn.GRU(input_size = y_dim+h_dim, hidden_size = a_dim, batch_first=True)

    def forward(self, y, h):
        inp = torch.cat([y,h], dim=-1)
        inp_rev = torch.flip(inp, dims=[1])
        a_rev, _ = self.gru(inp_rev)
        return torch.flip(a_rev, dims=[1])

class Posterior(nn.Module):
    """Eqs. (16)-(18): q(z_t | z_{t-1}, d_t, A_t) and q(d_t | d_{t-1}, A_t)."""

    def __init__(self, z_dim, a_dim, K, hidden_dim=None):
        super().__init__()
        self.K = K
        self.g_z = nn.ModuleList([GaussianNet(z_dim + a_dim, z_dim, hidden_dim) for _ in range(K)])
        self.g_d = nn.ModuleList([nn.Linear(a_dim, K) for _ in range(K)])

    def z_posterior(self, z_prev, A_t):
        inp = torch.cat([z_prev, A_t], dim=-1)
        outs = [self.g_z[k](inp) for k in range(self.K)]
        mu = torch.stack([o[0] for o in outs], dim=1)
        std = torch.stack([o[1] for o in outs], dim=1)
        return mu, std

    def d_posterior(self, A_t):
        logits = torch.stack([self.g_d[j](A_t) for j in range(self.K)], dim=1)
        return torch.softmax(logits, dim=-1)

if __name__ == "__main__":
    y = torch.randn(64, 20,1)
    h = torch.randn(64, 20, 10)
    backward_rnn = BackwardRNN(y_dim=1, h_dim=10, a_dim=10)
    A = backward_rnn(y, h)
    print("A shape:", A.shape)

    y2 = y.clone()
    y2[:, 0] += 1.0
    A2 = backward_rnn(y2, h)
    print("A_1 changed:", not torch.allclose(A[:, 0], A2[:, 0]))
    print("A_2..A_T unchanged:", torch.allclose(A[:, 1:], A2[:, 1:]))

    posterior = Posterior(z_dim=2, a_dim=10, K=2)
    A_t = A[:, 0]
    mu_q, std_q = posterior.z_posterior(torch.randn(64, 2), A_t)
    print("z posterior mean/std:", mu_q.shape, std_q.shape)
    q_d = posterior.d_posterior(A_t)
    print("d posterior shape:", q_d.shape)
    print("Rows sum to 1:", torch.allclose(q_d.sum(dim=-1), torch.ones(64, 2)))