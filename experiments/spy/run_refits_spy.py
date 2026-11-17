import pandas as pd

from ds3m.data import load_spy
from ds3m.train import fit
from experiments.spy.evaluate_spy import evaluate_spy

SEEDS = [1, 2, 3, 4, 5]
YEARS = range(2019, 2025)
C5 = dict(shuffle=True, clip=10, hidden_dim=16, anneal_schedule="none")

for seed in SEEDS:
    frames = []
    for year in YEARS:
        dates_kw = dict(train_end=f"{year - 4}-12-31",
                        valid_end=f"{year - 1}-12-31",
                        test_end=f"{year}-12-31")
        splits, _, _, _ = load_spy(**dates_kw)
        path = f"experiments/spy/ds3m_spy_refit{year}_seed{seed}.pt"
        fit(splits, seed, path, tag=f"refit {year} seed {seed}", **C5)
        frames.append(evaluate_spy(path, seed=seed, hidden_dim=16, **dates_kw))
    df = pd.concat(frames)
    df.to_csv(f"experiments/spy/forecasts_refit_wide_nokl_seed{seed}.csv")
    print(f"seed {seed}: {len(df)} days, overall coverage {df['covered'].mean():.3f}")