import numpy as np

from data import all_profiles, encode, prices
from models import add_const


def profile_matrix():
    P = all_profiles()
    return P, add_const(encode(P)), prices(P)


def single_revenue(U, p, thr=3):
    # revenue of every possible product if it is the only one offered
    return (U >= thr).sum(0) * p


def best_single(U, p, thr=3):
    return int(single_revenue(U, p, thr).argmax())


def line_revenue(U, p, line, thr=3):
    # each consumer buys the offered product with the highest rating if it is >= thr
    line = np.asarray(line)
    Ul = U[:, line]
    choice = Ul.argmax(1)
    buy = Ul.max(1) >= thr
    return (p[line][choice] * buy).sum()


def _swap_values(U, p, line, slot, thr):
    # revenue for every candidate product put into one slot of the line
    others = np.delete(np.asarray(line), slot)
    if len(others) == 0:
        return single_revenue(U, p, thr)
    Uo = U[:, others]
    u_other = Uo.max(1)
    p_other = p[others][Uo.argmax(1)] * (u_other >= thr)
    takes_new = (U >= thr) & (U > u_other[:, None])
    return (takes_new * p[None, :]).sum(0) + ((~takes_new) * p_other[:, None]).sum(0)


def best_line(U, p, J, thr=3, n_starts=5, seed=0):
    """Product line of J products for consumers who pick their favourite one.
    Greedy start plus random starts, then improve one slot at a time until nothing changes."""
    rng = np.random.default_rng(seed)
    starts = []
    greedy = []
    for _ in range(J):
        greedy.append(0)
        greedy[-1] = int(_swap_values(U, p, greedy, len(greedy) - 1, thr).argmax())
    starts.append(greedy)
    for _ in range(n_starts - 1):
        starts.append(list(rng.choice(U.shape[1], J, replace=False)))

    best, best_rev = None, -1
    for line in starts:
        line = list(line)
        rev = line_revenue(U, p, line, thr)
        improved = True
        while improved:
            improved = False
            for s in range(J):
                vals = _swap_values(U, p, line, s, thr)
                j = int(vals.argmax())
                if vals[j] > rev + 1e-9:
                    line[s] = j
                    rev = vals[j]
                    improved = True
        if rev > best_rev:
            best, best_rev = sorted(line), rev
    return best


def targeted_revenue(U, p, labels, products, thr=3):
    # each consumer can only buy the product made for their segment
    prod = np.asarray(products)[labels]
    u = U[np.arange(len(U)), prod]
    return (p[prod] * (u >= thr)).sum()


def oracle_revenue(U, p, thr=3):
    # every consumer gets the most expensive product they would still buy
    return ((U >= thr) * p[None, :]).max(1).sum()
