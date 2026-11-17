import torch

from ds3m.model import DS3M
from ds3m.data import load_spy

VARIANT, HIDDEN = "wide_nokl", 16

splits, _, _, _ = load_spy()
X_valid, _ = splits["valid"]

for seed in [1, 2, 3, 4, 5]:
    model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2, hidden_dim=HIDDEN)
    model.load_state_dict(torch.load(f"experiments/spy/ds3m_spy_{VARIANT}_seed{seed}.pt"))
    model.eval()
    with torch.no_grad():
        h = model.encoder(X_valid)[:, -1]
        z = torch.randn(len(h), 2)
        for k in range(2):
            out = model.generative.f_y[k].trunk(torch.cat([z, h], dim=-1))
            _, std = model.generative.f_y[k](torch.cat([z, h], dim=-1))
            print(f"seed {seed}, regime {k}: trunk output zero for {(out == 0).float().mean():.0%} of inputs "
                  f"| emission std varies by {std.std():.4f}")