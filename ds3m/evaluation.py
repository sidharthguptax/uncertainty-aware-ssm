import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


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