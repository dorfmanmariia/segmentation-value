import numpy as np
from scipy.stats import norm

from data import all_profiles, encode, prices
from models import add_const

# threshold: pick the offered product with the highest rating, buy it if the rating is >= thr
# smooth: pick the same product, buy it with probability Phi((rating - thr) / sd)
THRESHOLD = {"type": "threshold", "thr": 3}


def profile_matrix():
    P = all_profiles()
    return P, add_const(encode(P)), prices(P)


def buy_prob(u, demand):
    if demand["type"] == "threshold":
        return (u >= demand["thr"]).astype(float)
    return norm.cdf((u - demand["thr"]) / demand["sd"])


def single_revenue(U, p, demand=THRESHOLD):
    # revenue of every possible product if it is the only one offered
    return buy_prob(U, demand).sum(0) * p


def best_single(U, p, demand=THRESHOLD):
    return int(single_revenue(U, p, demand).argmax())


def line_revenue(U, p, line, demand=THRESHOLD):
    line = np.asarray(line)
    Ul = U[:, line]
    choice = Ul.argmax(1)
    return (p[line][choice] * buy_prob(Ul.max(1), demand)).sum()


def _swap_values(U, p, line, slot, demand):
    # revenue for every candidate product put into one slot of the line
    others = np.delete(np.asarray(line, dtype=int), slot)
    if len(others) == 0:
        return single_revenue(U, p, demand)
    Uo = U[:, others]
    u_other = Uo.max(1)
    rev_other = p[others][Uo.argmax(1)] * buy_prob(u_other, demand)
    rev_new = p[None, :] * buy_prob(U, demand)
    takes_new = U > u_other[:, None]
    vals = np.where(takes_new, rev_new, rev_other[:, None]).sum(0)
    vals[others] = -np.inf
    return vals


def best_line(U, p, J, demand=THRESHOLD, n_starts=5, seed=0):
    """Product line of J products for consumers who pick their favourite one.
    Greedy start plus random starts, then improve one slot at a time until nothing changes."""
    rng = np.random.default_rng(seed)
    greedy = []
    for _ in range(J):
        greedy.append(0)
        greedy[-1] = int(_swap_values(U, p, greedy, len(greedy) - 1, demand).argmax())
    starts = [greedy] + [list(rng.choice(U.shape[1], J, replace=False)) for _ in range(n_starts - 1)]

    best, best_rev = None, -1
    for line in starts:
        line = list(line)
        rev = line_revenue(U, p, line, demand)
        improved = True
        while improved:
            improved = False
            for s in range(J):
                vals = _swap_values(U, p, line, s, demand)
                j = int(vals.argmax())
                if vals[j] > rev + 1e-9:
                    line[s] = j
                    rev = vals[j]
                    improved = True
        if rev > best_rev:
            best, best_rev = sorted(line), rev
    return best


def targeted_revenue(U, p, labels, products, demand=THRESHOLD):
    # each consumer can only buy the product made for their segment
    prod = np.asarray(products)[labels]
    u = U[np.arange(len(U)), prod]
    return (p[prod] * buy_prob(u, demand)).sum()


def oracle_revenue(U, p, demand=THRESHOLD):
    # every consumer gets the product that gives the most expected revenue from them
    return (buy_prob(U, demand) * p[None, :]).max(1).sum()
