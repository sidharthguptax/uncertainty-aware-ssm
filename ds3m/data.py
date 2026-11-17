# Split, normalization and windowing follow the official main.py and utils.py.
# Regime labels: y[i] pairs with d[i] (verified in experiments/toy/check_official.py),
# so the extra last value of d is dropped. Their main.py evaluates against d[-500:]
# of the full 2001-value file, which shifts their labels one step later.
# Requires the official repo cloned next to this one (see OFFICIAL_TOY).

import numpy as np
import torch
import pandas as pd

OFFICIAL_TOY = "../Deep-Switching-State-Space-Model/data/Toy/"


def make_windows(series, window):
    X, Y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        Y.append(series[i + 1:i + window + 1])
    return np.array(X), np.array(Y)


def load_toy(window=20, test_len=500):
    y = np.loadtxt(OFFICIAL_TOY + "simulation_data_nonlinear_y.csv", delimiter=",").reshape(-1, 1)
    d = np.loadtxt(OFFICIAL_TOY + "simulation_data_nonlinear_d.csv", delimiter=",")[:-1]

    n_trainval = len(y) - test_len
    n_train = int(n_trainval * 0.75)
    train = y[:n_train]
    valid = y[n_train:n_trainval]
    test = y[-test_len - window - 1:-1]

    mean, std = train.mean(), train.std()

    splits = {}
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        X, Y = make_windows((part - mean) / std, window)
        splits[name] = (torch.tensor(X, dtype=torch.float32),
                        torch.tensor(Y, dtype=torch.float32))

    return splits, (mean, std), y[-test_len:], d[-test_len:]

def load_spy(window=20, train_end="2015-12-31", valid_end="2018-12-31",
             path="data/spy_close.csv"):
    close = pd.read_csv(path, index_col="Date", parse_dates=True)["SPY"]
    returns = np.log(close).diff().dropna()
    dates = returns.index
    y = returns.values.reshape(-1, 1)

    n_train = int((dates <= train_end).sum())
    n_valid = int((dates <= valid_end).sum())
    train = y[:n_train]
    valid = y[n_train:n_valid]
    test = y[n_valid - window - 1:-1]

    mean, std = train.mean(), train.std()

    splits = {}
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        X, Y = make_windows((part - mean) / std, window)
        splits[name] = (torch.tensor(X, dtype=torch.float32),
                        torch.tensor(Y, dtype=torch.float32))

    return splits, (mean, std), y[n_valid:], dates[n_valid:]

if __name__ == "__main__":
    splits, (mean, std), y_test, d_test = load_toy()
    for name, (X, Y) in splits.items():
        print(f"{name}: X {tuple(X.shape)}, Y {tuple(Y.shape)}")
    print("Train mean/std:", mean, std)
    print("Test targets:", y_test.shape, "| test regimes:", d_test.shape)

    X_test, Y_test = splits["test"]
    last_known = Y_test[:, -1, 0].numpy() * std + mean
    print("Windows line up with targets:", np.allclose(last_known[1:], y_test[:-1, 0]))

    print("--- SPY ---")
    splits, (mean, std), y_test, test_dates = load_spy()
    for name, (X, Y) in splits.items():
        print(f"{name}: X {tuple(X.shape)}, Y {tuple(Y.shape)}")
    print("Train mean/std of daily log returns:", mean, std)
    print("Test period:", test_dates[0].date(), "to", test_dates[-1].date(), "|", len(y_test), "days")

    X_test, Y_test = splits["test"]
    last_known = Y_test[:, -1, 0].numpy() * std + mean
    print("Windows line up with targets:", np.allclose(last_known[1:], y_test[:-1, 0]))