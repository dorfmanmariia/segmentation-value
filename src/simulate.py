import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models import hb, add_const
from segments import latent_class, lc_assign
from optimize import profile_matrix, best_single, best_line, line_revenue, targeted_revenue, oracle_revenue


def calibrate(R, X, K=3, seed=0):
    """Population of partworths estimated from the data:
    b_i = mu + sep * d_k(i) + w_i, w_i ~ N(0, Sw), where d_k are LC segment deviations
    and Sw is the HB covariance minus the part explained by segments."""
    res = hb(R, X, seed=seed)
    mu = res["mu"].mean(0)
    Sigma = res["Sigma"].mean(0)
    lc = latent_class(R, X, K, seed=seed)
    D = np.column_stack([np.zeros(K), lc["B"]])
    D = D - lc["pi"] @ D
    Sb = (lc["pi"][:, None] * D).T @ D
    vals, vecs = np.linalg.eigh(Sigma - Sb)
    Sw = vecs @ np.diag(np.clip(vals, 0, None)) @ vecs.T
    return {"mu": mu, "D": D, "pi": lc["pi"], "Sw": Sw, "s2": res["s2"].mean()}


def draw_population(cal, n, sep, within, rng):
    k = rng.choice(len(cal["pi"]), n, p=cal["pi"])
    w = rng.multivariate_normal(np.zeros(len(cal["mu"])), cal["Sw"], n)
    return cal["mu"] + sep * cal["D"][k] + np.sqrt(within) * w


def draw_ratings(B, X, s2, rng):
    r = B @ add_const(X).T + rng.normal(0, np.sqrt(s2), (len(B), len(X)))
    return np.clip(np.round(r), 1, 5)


def posterior_partworths(R, X, mu, Sigma, s2):
    # partworths of new respondents given the population estimated by HB (posterior means)
    Z = add_const(X)
    Sinv = np.linalg.inv(Sigma)
    cov = np.linalg.inv(Z.T @ Z / s2 + Sinv)
    return (R @ Z / s2 + Sinv @ mu) @ cov


def run_rep(cal, X, n, sep, within, Ks=(2, 3), n_eval=1000, seed=0):
    rng = np.random.default_rng(seed)
    _, Zp, p = profile_matrix()

    B = draw_population(cal, n, sep, within, rng)
    R = draw_ratings(B, X, cal["s2"], rng)
    B_eval = draw_population(cal, n_eval, sep, within, rng)
    R_eval = draw_ratings(B_eval, X, cal["s2"], rng)
    U_eval = B_eval @ Zp.T

    res = hb(R, X, n_iter=2000, burn=500, thin=5, seed=seed)
    B_hat = res["B"].mean(0)
    U_hat = B_hat @ Zp.T
    B_eval_hat = posterior_partworths(R_eval, X, res["mu"].mean(0), res["Sigma"].mean(0), res["s2"].mean())

    rev = lambda line: line_revenue(U_eval, p, line) / n_eval
    out = {
        "mass realized": rev([best_single(U_hat, p)]),
        "mass potential": rev([best_single(U_eval, p)]),
        "oracle": oracle_revenue(U_eval, p) / n_eval,
    }
    for K in Ks:
        out[f"line{K} realized"] = rev(best_line(U_hat, p, K, n_starts=3, seed=seed))
        out[f"line{K} potential"] = rev(best_line(U_eval, p, K, n_starts=3, seed=seed))

        lc = latent_class(R, X, K, n_starts=5, seed=seed)
        prods = [best_single(U_hat[lc["labels"] == k], p) for k in range(K)]
        lab = lc_assign(R_eval, X, lc).argmax(1)
        out[f"targeted LC{K} realized"] = targeted_revenue(U_eval, p, lab, prods) / n_eval

        sc = StandardScaler().fit(B_hat[:, 1:])
        km = KMeans(K, n_init=5, random_state=seed).fit(sc.transform(B_hat[:, 1:]))
        prods = [best_single(U_hat[km.labels_ == k], p) for k in range(K)]
        lab = km.predict(sc.transform(B_eval_hat[:, 1:]))
        out[f"targeted km{K} realized"] = targeted_revenue(U_eval, p, lab, prods) / n_eval

        sc = StandardScaler().fit(B_eval[:, 1:])
        km = KMeans(K, n_init=5, random_state=seed).fit(sc.transform(B_eval[:, 1:]))
        prods = [best_single(U_eval[km.labels_ == k], p) for k in range(K)]
        out[f"targeted {K} potential"] = targeted_revenue(U_eval, p, km.labels_, prods) / n_eval
    return out


STRUCTURES = {
    "none": (0.0, 0.0),
    "continuous": (0.0, 1.0),
    "calibrated": (1.0, 1.0),
    "strong segments": (2.0, 1.0),
    "distinct segments": (2.0, 0.25),
}


def run_grid(cal, X, sizes, n_reps, structures=STRUCTURES, seed=0):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows = []
    for name, (sep, within) in structures.items():
        for n in sizes:
            for r in range(n_reps):
                row = run_rep(cal, X, n, sep, within, seed=seed + 1000 * r + n)
                row.update({"structure": name, "n": n, "rep": r})
                rows.append(row)
    return pd.DataFrame(rows)
