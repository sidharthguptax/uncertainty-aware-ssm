import numpy as np
import pandas as pd
import torch

from ds3m.model import DS3M
from ds3m.data import load_spy

PERIODS = {
    "2019": ("2019-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "COVID crash": ("2020-02-20", "2020-04-30"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023-2024": ("2023-01-01", "2024-12-31"),
}


def evaluate_spy(path, seed=3, S=200, hidden_dim=None):
    torch.manual_seed(seed)
    splits, (mean, std), y_test, dates = load_spy()
    X_test, Y_test = splits["test"]

    model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(path))
    model.eval()

    y_samples, d_samples = model.forecast(X_test, Y_test, S=S)
    y_samples = y_samples[..., 0].numpy() * std + mean

    y_true = y_test[:, 0]
    lo, hi = np.quantile(y_samples, [0.05, 0.95], axis=1)
    d_pred = (d_samples.float().mean(dim=1) > 0.5).long().numpy()

    return pd.DataFrame({
        "y": y_true,
        "mean": y_samples.mean(axis=1),
        "lo": lo,
        "hi": hi,
        "covered": (y_true >= lo) & (y_true <= hi),
        "width": hi - lo,
        "regime": d_pred,
    }, index=dates)


if __name__ == "__main__":
    df = evaluate_spy("experiments/spy/ds3m_spy_seed3.pt")
    df.to_csv("experiments/spy/forecasts_seed3.csv")

    rmse = np.sqrt(((df["mean"] - df["y"]) ** 2).mean())
    rmse_zero = np.sqrt((df["y"] ** 2).mean())
    print(f"RMSE: {rmse:.5f} | RMSE of always predicting 0: {rmse_zero:.5f}")
    print("Regimes used:", df["regime"].nunique(), "| share of days in regime 1:", round(df["regime"].mean(), 3))
    print(f"Overall 90% coverage: {df['covered'].mean():.3f}")

    print("\nPeriod       | days | coverage | mean width | share regime 1")
    for name, (start, end) in PERIODS.items():
        p = df.loc[start:end]
        print(f"{name:12s} | {len(p):4d} | {p['covered'].mean():.3f}    | {p['width'].mean():.4f}     | {p['regime'].mean():.2f}")