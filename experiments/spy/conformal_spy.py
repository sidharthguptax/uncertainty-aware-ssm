import numpy as np
import pandas as pd

from ds3m.conformal import split_conformal, aci
from ds3m.evaluation import kupiec, christoffersen

MODELS = {
    "GARCH-N": ("calib_garch_normal", "forecasts_garch_normal"),
    "GARCH-t": ("calib_garch_t", "forecasts_garch_t"),
}
for seed in range(1, 6):
    MODELS[f"C5 s{seed}"] = (f"calib_wide_nokl_seed{seed}", f"forecasts_wide_nokl_seed{seed}")


def load(name):
    return pd.read_csv(f"experiments/spy/{name}.csv", index_col=0, parse_dates=True)


def summarize(df):
    miss = ~df["covered"]
    return {
        "overall": df["covered"].mean(),
        "2019": df.loc["2019", "covered"].mean(),
        "crash": df.loc["2020-02-20":"2020-04-30", "covered"].mean(),
        "2022": df.loc["2022", "covered"].mean(),
        "width": df["width"].mean(),
        "kupiec_p": kupiec(miss)[1],
        "christ_p": christoffersen(miss)[1],
    }


rows = []
for model, (calib_name, test_name) in MODELS.items():
    calib, test = load(calib_name), load(test_name)
    split_df, q_split = split_conformal(calib, test)
    aci_df, q_aci = aci(calib, test)
    split_df.to_csv(f"experiments/spy/conformal_split_{test_name}.csv")
    aci_df.to_csv(f"experiments/spy/conformal_aci_{test_name}.csv")
    for method, df in [("raw", test), ("split", split_df), ("ACI", aci_df)]:
        rows.append({"model": model, "method": method, **summarize(df)})
    print(f"{model}: split-conformal q = {q_split:.3f} | ACI q range {q_aci.min():.2f}-{q_aci.max():.2f}")

table = pd.DataFrame(rows)
table.to_csv("experiments/spy/conformal_results.csv", index=False)
print()
print(table.round(3).to_string(index=False))

print("\n=== C5 over 5 seeds (mean ± std) ===")
c5 = table[table["model"].str.startswith("C5")]
for method in ["raw", "split", "ACI"]:
    m = c5[c5["method"] == method]
    print(f"{method:5s} | overall {m['overall'].mean():.3f} ± {m['overall'].std():.3f}"
          f" | crash {m['crash'].mean():.3f} ± {m['crash'].std():.3f}"
          f" | 2022 {m['2022'].mean():.3f} ± {m['2022'].std():.3f}"
          f" | width {m['width'].mean():.4f}")