import json
import numpy as np

from ds3m.data import load_spy
from ds3m.train import fit
from experiments.spy.evaluate_spy import evaluate_spy, PERIODS

VARIANTS = {
    "official": dict(shuffle=False, clip=None, hidden_dim=None),
    "stable": dict(shuffle=True, clip=10, hidden_dim=None),
    "wide": dict(shuffle=True, clip=10, hidden_dim=16),
    "wide_paperkl": dict(shuffle=True, clip=10, hidden_dim=16, anneal_schedule="paper"),
    "wide_nokl": dict(shuffle=True, clip=10, hidden_dim=16, anneal_schedule="none"),
    "stable_nokl": dict(shuffle=True, clip=10, hidden_dim=None, anneal_schedule="none"),
}
VARIANT = "stable_nokl"

SEEDS = [1, 2, 3, 4, 5]

splits, _, _, _ = load_spy()
summary = []

for seed in SEEDS:
    path = f"experiments/spy/ds3m_spy_{VARIANT}_seed{seed}.pt"
    fit(splits, seed, path, tag=f"SPY {VARIANT} seed {seed}", **VARIANTS[VARIANT])
    df = evaluate_spy(path, seed=seed, hidden_dim=VARIANTS[VARIANT]["hidden_dim"])
    df.to_csv(f"experiments/spy/forecasts_{VARIANT}_seed{seed}.csv")

    row = {"seed": seed,
           "regimes_used": int(df["regime"].nunique()),
           "overall": float(df["covered"].mean())}
    for name, (start, end) in PERIODS.items():
        row[name] = float(df.loc[start:end, "covered"].mean())
    summary.append(row)
    print(row)

with open(f"experiments/spy/summary_{VARIANT}_5seeds.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== {VARIANT}: coverage over seeds (mean ± std) ===")
for key in ["overall"] + list(PERIODS):
    values = np.array([r[key] for r in summary])
    print(f"{key:12s}: {values.mean():.3f} ± {values.std(ddof=1):.3f}")
print("Collapsed runs (1 regime):", sum(r["regimes_used"] == 1 for r in summary), "of", len(SEEDS))