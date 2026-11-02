import numpy as np
import statsmodels.api as sm
from scipy.stats import invwishart


def add_const(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def pooled_ols(long, cols):
    y = long["rating"]
    Z = sm.add_constant(long[cols].astype(float))
    return sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": long["respondent"]})


def individual_ols(R, X):
    Z = add_const(X)
    B, *_ = np.linalg.lstsq(Z, R.T, rcond=None)
    return B.T


def hb(R, X, n_iter=6000, burn=2000, thin=10, seed=0):
    """Hierarchical linear model: r_ij = z_j'b_i + e_ij, b_i ~ N(mu, Sigma), e_ij ~ N(0, s2)."""
    rng = np.random.default_rng(seed)
    Z = add_const(X)
    n, m = R.shape
    p = Z.shape[1]
    ZZ = Z.T @ Z
    ZR = R @ Z

    nu0 = p + 3
    V0 = nu0 * 0.1 * np.eye(p)
    a0, b0 = 1.0, 1.0

    B = individual_ols(R, X)
    mu = B.mean(0)
    Sigma = np.cov(B.T)
    s2 = 1.0

    keep = {"B": [], "mu": [], "Sigma": [], "s2": []}
    for it in range(n_iter):
        Sinv = np.linalg.inv(Sigma)
        cov = np.linalg.inv(ZZ / s2 + Sinv)
        mean = (ZR / s2 + Sinv @ mu) @ cov
        B = mean + rng.standard_normal((n, p)) @ np.linalg.cholesky(cov).T

        mu = rng.multivariate_normal(B.mean(0), Sigma / n)

        D = B - mu
        Sigma = invwishart.rvs(df=nu0 + n, scale=V0 + D.T @ D, random_state=rng)

        resid = R - B @ Z.T
        s2 = 1 / rng.gamma(a0 + n * m / 2, 1 / (b0 + (resid ** 2).sum() / 2))

        if it >= burn and (it - burn) % thin == 0:
            keep["B"].append(B)
            keep["mu"].append(mu)
            keep["Sigma"].append(Sigma)
            keep["s2"].append(s2)

    return {k: np.array(v) for k, v in keep.items()}


def rmse(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2))


def hit_rate(pred, actual):
    # share of respondents whose predicted best profile is among their actually top-rated ones
    best = pred.argmax(1)
    return np.mean(actual[np.arange(len(actual)), best] == actual.max(1))
