import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models import hb
from segments import latent_class, lc_assign
from optimize import profile_matrix, best_single, best_line, line_revenue, targeted_revenue, oracle_revenue


def segment_labels(method, K, R_fit, B_fit, X, R_new, B_new, seed=0):
    # segments estimated on (R_fit, B_fit), then assigned for both groups of respondents
    if K == 1:
        return np.zeros(len(R_fit), int), np.zeros(len(R_new), int)
    if method == "LC":
        fit = latent_class(R_fit, X, K, n_starts=10, seed=seed)
        return fit["labels"], lc_assign(R_new, X, fit).argmax(1)
    scaler = StandardScaler().fit(B_fit[:, 1:])
    km = KMeans(K, n_init=10, random_state=seed).fit(scaler.transform(B_fit[:, 1:]))
    return km.labels_, km.predict(scaler.transform(B_new[:, 1:]))


def strategies(U_fit, labels_fit, U_eval, labels_eval, p, K, seed=0):
    """Products are chosen with U_fit and evaluated with U_eval. Returns revenue per consumer."""
    n = len(U_eval)
    seg_products = [best_single(U_fit[labels_fit == k], p) for k in range(K)]
    line = best_line(U_fit, p, K, seed=seed)
    return {
        "targeted": targeted_revenue(U_eval, p, labels_eval, seg_products) / n,
        "segment products, open": line_revenue(U_eval, p, seg_products) / n,
        "optimal line": line_revenue(U_eval, p, line) / n,
    }


def run_fold(R, X, B_true, train, test, Ks, methods=("LC", "k-means"), seed=0):
    """Segments and products are chosen on the train respondents and evaluated on the test
    respondents ("realized"). "potential" repeats the same choices using the test respondents'
    own partworths, i.e. as if the firm knew their preferences."""
    _, Zp, p = profile_matrix()
    U_test = B_true[test] @ Zp.T
    B_train = hb(R[train], X, n_iter=4000, burn=1000, seed=seed)["B"].mean(0)
    U_train = B_train @ Zp.T

    rows = [{"K": 1, "method": "-", "strategy": "oracle", "realized": oracle_revenue(U_test, p) / len(test),
             "potential": oracle_revenue(U_test, p) / len(test)}]
    for method in methods:
        for K in Ks:
            if K == 1 and method != methods[0]:
                continue
            lab_tr, lab_te = segment_labels(method, K, R[train], B_train, X, R[test], B_true[test], seed)
            lab_pot, _ = segment_labels(method, K, R[test], B_true[test], X, R[test], B_true[test], seed)
            real = strategies(U_train, lab_tr, U_test, lab_te, p, K, seed)
            pot = strategies(U_test, lab_pot, U_test, lab_pot, p, K, seed)
            for s in real:
                if K == 1 and s != "targeted":
                    continue
                if s == "optimal line" and method != methods[0]:
                    continue
                rows.append({"K": K, "method": method if K > 1 and s != "optimal line" else "-",
                             "strategy": s if K > 1 else "mass market",
                             "realized": real[s], "potential": pot[s]})
    return rows


def repeated_cv(R, X, B_true, Ks, n_reps=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    n = len(R)
    for r in range(n_reps):
        perm = rng.permutation(n)
        halves = [perm[: n // 2], perm[n // 2:]]
        for f in range(2):
            for row in run_fold(R, X, B_true, halves[1 - f], halves[f], Ks, seed=2 * r + f):
                row["fold"] = 2 * r + f
                rows.append(row)
    return rows
