from experiments.spy.evaluate_spy import evaluate_spy
from experiments.spy.garch_spy import garch_forecasts

for dist in ["normal", "t"]:
    df = garch_forecasts(dist, start="2016-01-01", end="2018-12-31")
    df.to_csv(f"experiments/spy/calib_garch_{dist}.csv")
    print(f"GARCH-{dist} calibration period: {len(df)} days, coverage {df['covered'].mean():.3f}")

for seed in range(1, 6):
    path = f"experiments/spy/ds3m_spy_wide_nokl_seed{seed}.pt"
    df = evaluate_spy(path, seed=seed, hidden_dim=16, target="calib")
    df.to_csv(f"experiments/spy/calib_wide_nokl_seed{seed}.csv")
    print(f"C5 seed {seed} calibration period: {len(df)} days, coverage {df['covered'].mean():.3f}")