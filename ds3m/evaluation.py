import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, norm, t as student_t


def kupiec(miss, p=0.10):
    n, x = len(miss), int(miss.sum())
    pi = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = -2 * (ll_null - ll_alt)
    return lr, chi2.sf(lr, df=1)


def christoffersen(miss):
    m = np.asarray(miss, dtype=int)
    prev, curr = m[:-1], m[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi0) + xlogy(n01, pi0)
              + xlogy(n10, 1 - pi1) + xlogy(n11, pi1))
    lr = -2 * (ll_null - ll_alt)
    return lr, chi2.sf(lr, df=1)

def interval_score(df, alpha=0.10):
    y, lo, hi = df["y"].values, df["lo"].values, df["hi"].values
    return ((hi - lo)
            + (2 / alpha) * (lo - y) * (y < lo)
            + (2 / alpha) * (y - hi) * (y > hi))


def diebold_mariano(loss_a, loss_b, lags=None, hln=True):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if lags is None:
        lags = int(np.floor(n ** (1 / 3)))
    d_c = d - d.mean()
    lrv = np.sum(d_c ** 2) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * np.sum(d_c[k:] * d_c[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    if hln:
        stat = stat * np.sqrt((n - 1) / n)
        p = 2 * student_t.sf(abs(stat), df=n - 1)
    else:
        p = 2 * norm.sf(abs(stat))
    return d.mean(), stat, p