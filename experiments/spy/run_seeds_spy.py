import json
import numpy as np

from ds3m.data import load_spy
from ds3m.train import fit
from experiments.spy.evaluate_spy import evaluate_spy, PERIODS

SEEDS = [1, 2, 3, 4, 5]
splits, _, _, _ = load_spy()
summary = []

for seed in SEEDS:
    path = f"experiments/spy/ds3m_spy_seed{seed}.pt"
    fit(splits, seed, path, tag=f"SPY seed {seed}")
    df = evaluate_spy(path, seed=seed)
    df.to_csv(f"experiments/spy/forecasts_seed{seed}.csv")

    row = {"seed": seed,
           "regimes_used": int(df["regime"].nunique()),
           "overall": float(df["covered"].mean())}
    for name, (start, end) in PERIODS.items():
        row[name] = float(df.loc[start:end, "covered"].mean())
    summary.append(row)
    print(row)

with open("experiments/spy/summary_5seeds.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Coverage over seeds (mean ± std) ===")
for key in ["overall"] + list(PERIODS):
    values = np.array([r[key] for r in summary])
    print(f"{key:12s}: {values.mean():.3f} ± {values.std(ddof=1):.3f}")
print("Collapsed runs (1 regime):", sum(r["regimes_used"] == 1 for r in summary), "of", len(SEEDS))