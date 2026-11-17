import json
import numpy as np

from experiments.toy.train_toy import train
from experiments.toy.evaluate_toy import evaluate

SEEDS = [1, 2, 3, 4, 5]
results = []

for seed in SEEDS:
    print(f"=== Seed {seed} ===")
    path, _ = train(seed)
    metrics = evaluate(seed, path)
    print(metrics)
    results.append(metrics)

with open("experiments/toy/results_5seeds.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== Summary over {len(SEEDS)} seeds (mean ± std) ===")
for key in results[0]:
    values = np.array([r[key] for r in results])
    print(f"{key}: {values.mean():.3f} ± {values.std(ddof=1):.3f}")