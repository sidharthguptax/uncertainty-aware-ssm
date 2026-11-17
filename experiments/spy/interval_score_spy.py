import numpy as np
import pandas as pd

from ds3m.evaluation import interval_score, diebold_mariano

PERIODS = {
    "overall": (None, None),
    "2019": ("2019-01-01", "2019-12-31"),
    "COVID crash": ("2020-02-20", "2020-04-30"),
    "2022": ("2022-01-01", "2022-12-31"),
}
SEEDS = [1, 2, 3, 4, 5]


def scores(name):
    df = pd.read_csv(f"experiments/spy/{name}.csv", index_col=0, parse_dates=True)
    return pd.Series(100 * interval_score(df), index=df.index)


for method, prefix in [("raw", ""), ("ACI", "conformal_aci_")]:
    c5 = {s: scores(f"{prefix}forecasts_wide_nokl_seed{s}") for s in SEEDS}
    for garch_name, garch_file in [("GARCH-N", "forecasts_garch_normal"), ("GARCH-t", "forecasts_garch_t")]:
        g = scores(prefix + garch_file)
        print(f"\n=== {method}: C5 vs {garch_name} (mean interval score x100, lower is better) ===")
        for period, (start, end) in PERIODS.items():
            g_p = g.loc[start:end]
            wins = better = worse = 0
            for s in SEEDS:
                diff, stat, p = diebold_mariano(c5[s].loc[start:end].values, g_p.values)
                wins += diff < 0
                better += (diff < 0) and (p < 0.05)
                worse += (diff > 0) and (p < 0.05)
            c_mean = np.mean([c5[s].loc[start:end].mean() for s in SEEDS])
            print(f"{period:12s} | {garch_name} {g_p.mean():.3f} | C5 {c_mean:.3f} | "
                  f"C5 lower in {wins}/5 seeds | significantly better {better}/5, worse {worse}/5")

print("\n=== Lag sensitivity, ACI, COVID crash: DM p-values for seeds 1-5 ===")
start, end = PERIODS["COVID crash"]
for garch_name, garch_file in [("GARCH-N", "forecasts_garch_normal"), ("GARCH-t", "forecasts_garch_t")]:
    g_p = scores("conformal_aci_" + garch_file).loc[start:end]
    for lags in [0, 3, 6, 10]:
        ps = [diebold_mariano(scores(f"conformal_aci_forecasts_wide_nokl_seed{s}").loc[start:end].values,
                              g_p.values, lags=lags)[2] for s in SEEDS]
        print(f"{garch_name} | lags {lags:2d} | " + "  ".join(f"{p:.3f}" for p in ps))