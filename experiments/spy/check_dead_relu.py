import torch

from ds3m.model import DS3M
from ds3m.data import load_spy

CONFIGS = {
    "official": ("experiments/spy/ds3m_spy_seed{}.pt", None),
    "stable": ("experiments/spy/ds3m_spy_stable_seed{}.pt", None),
    "wide_nokl": ("experiments/spy/ds3m_spy_wide_nokl_seed{}.pt", 16),
    "stable_nokl": ("experiments/spy/ds3m_spy_stable_nokl_seed{}.pt", None),
}

splits, _, _, _ = load_spy()
DATA = {"train": splits["train"], "valid": splits["valid"], "test": splits["test"]}

for name, (pattern, hidden) in CONFIGS.items():
    print(f"=== {name} ===")
    for seed in [1, 2, 3, 4, 5]:
        torch.manual_seed(seed)
        model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2, hidden_dim=hidden)
        model.load_state_dict(torch.load(pattern.format(seed)))
        model.eval()
        zero = {0: [], 1: []}
        with torch.no_grad():
            for data_name, (X, Y) in DATA.items():
                model(X, Y)
                inp = torch.cat([model.z_last, model.encoder(X)[:, -1]], dim=-1)
                for k in range(2):
                    out = model.generative.f_y[k].trunk(inp)
                    zero[k].append(f"{(out == 0).float().mean():.0%}")
        print(f"seed {seed} | regime 0 zero (train/valid/test): {'/'.join(zero[0])}"
              f" | regime 1: {'/'.join(zero[1])}")