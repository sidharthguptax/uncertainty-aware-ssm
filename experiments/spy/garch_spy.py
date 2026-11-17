import numpy as np
import pandas as pd
from arch import arch_model

from experiments.spy.evaluate_spy import PERIODS

close = pd.read_csv("data/spy_close.csv", index_col="Date", parse_dates=True)["SPY"]
r = 100 * np.log(close).diff().dropna()


def garch_forecasts(dist, start="2019-01-01", end=None):
    target = r.loc[start:end]
    origin = r.index[r.index.get_loc(target.index[0]) - 1]

    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    res = am.fit(last_obs="2016-01-01", disp="off")
    fc = res.forecast(horizon=1, start=origin, reindex=False)

    mu = fc.mean["h.1"].values[:len(target)]
    sigma = np.sqrt(fc.variance["h.1"].values[:len(target)])
    assert len(mu) == len(target)

    if dist == "normal":
        q_lo, q_hi = -1.6449, 1.6449
    else:
        q_lo, q_hi = am.distribution.ppf([0.05, 0.95], [res.params["nu"]])

    print(f"GARCH-{dist} parameters:\n{res.params.round(4)}\n")
    y = target.values / 100
    lo = (mu + q_lo * sigma) / 100
    hi = (mu + q_hi * sigma) / 100
    return pd.DataFrame({"y": y, "mean": mu / 100, "lo": lo, "hi": hi,
                         "covered": (y >= lo) & (y <= hi), "width": hi - lo},
                        index=target.index)


if __name__ == "__main__":
    for dist in ["normal", "t"]:
        df = garch_forecasts(dist)
        df.to_csv(f"experiments/spy/forecasts_garch_{dist}.csv")
        print(f"GARCH-{dist}: overall 90% coverage {df['covered'].mean():.3f}")
        print("Period       | days | coverage | mean width")
        for name, (start, end) in PERIODS.items():
            p = df.loc[start:end]
            print(f"{name:12s} | {len(p):4d} | {p['covered'].mean():.3f}    | {p['width'].mean():.4f}")
        print()