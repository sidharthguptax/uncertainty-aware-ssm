import torch

from ds3m.model import DS3M
from ds3m.data import load_spy

splits, _, _, _ = load_spy()
X, Y = splits["train"]

print("=== at initialisation (untrained, official width) ===")
for seed in [1, 2, 3, 4, 5]:
    torch.manual_seed(seed)
    model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2)
    model.eval()
    with torch.no_grad():
        model(X, Y)
        inp = torch.cat([model.z_last, model.encoder(X)[:, -1]], dim=-1)
        zeros = [f"{(model.generative.f_y[k].trunk(inp) == 0).float().mean():.0%}" for k in range(2)]
    print(f"seed {seed} | regime 0: {zeros[0]} zero | regime 1: {zeros[1]} zero")

print("\n=== stable_nokl: emission per regime on training data ===")
for seed in [1, 2, 3, 4, 5]:
    torch.manual_seed(seed)
    model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2)
    model.load_state_dict(torch.load(f"experiments/spy/ds3m_spy_stable_nokl_seed{seed}.pt"))
    model.eval()
    with torch.no_grad():
        model(X, Y)
        inp = torch.cat([model.z_last, model.encoder(X)[:, -1]], dim=-1)
        parts = []
        for k in range(2):
            mu, std = model.generative.f_y[k](inp)
            parts.append(f"regime {k}: mean {mu.mean():.3f}, std {std.mean():.3f}")
    print(f"seed {seed} | " + " | ".join(parts))