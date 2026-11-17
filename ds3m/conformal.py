import numpy as np


def scores(df):
    center = ((df["lo"] + df["hi"]) / 2).values
    half = ((df["hi"] - df["lo"]) / 2).values
    s = np.abs(df["y"].values - center) / half
    return s, center, half


def conformal_quantile(s, level):
    n = len(s)
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return s.max()
    return np.sort(s)[k - 1]


def recalibrate(df, q):
    out = df.copy()
    _, center, half = scores(df)
    out["lo"] = center - q * half
    out["hi"] = center + q * half
    out["covered"] = (out["y"] >= out["lo"]) & (out["y"] <= out["hi"])
    out["width"] = out["hi"] - out["lo"]
    return out


def split_conformal(calib, test, alpha=0.10):
    s_cal, _, _ = scores(calib)
    q = conformal_quantile(s_cal, 1 - alpha)
    return recalibrate(test, q), q


def aci(calib, test, alpha=0.10, gamma=0.005):
    s_cal, _, _ = scores(calib)
    s_test, _, _ = scores(test)
    history = list(s_cal)
    alpha_t = alpha
    qs = []
    for s in s_test:
        hist = np.array(history)
        if alpha_t <= 0:
            q = hist.max()
        elif alpha_t >= 1:
            q = 0.0
        else:
            q = conformal_quantile(hist, 1 - alpha_t)
        qs.append(q)
        err = float(s > q)
        alpha_t = alpha_t + gamma * (alpha - err)
        history.append(s)
    qs = np.array(qs)
    return recalibrate(test, qs), qs