import pandas as pd

from ds3m.evaluation import kupiec, christoffersen

MODELS = {
    "DS3M seed 3": "experiments/spy/forecasts_seed3.csv",
    "GARCH-N": "experiments/spy/forecasts_garch_normal.csv",
    "GARCH-t": "experiments/spy/forecasts_garch_t.csv",
}

print("Model        | miss rate | Kupiec p | Christoffersen p | crash Kupiec p | 2022 Kupiec p")
for name, path in MODELS.items():
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    miss = ~df["covered"]
    _, p_uc = kupiec(miss)
    _, p_ind = christoffersen(miss)
    _, p_crash = kupiec(miss.loc["2020-02-20":"2020-04-30"])
    _, p_2022 = kupiec(miss.loc["2022-01-01":"2022-12-31"])
    print(f"{name:12s} | {miss.mean():.3f}     | {p_uc:.3f}    | {p_ind:.3f}            | {p_crash:.3f}          | {p_2022:.3f}")