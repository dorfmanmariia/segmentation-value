
import itertools

import numpy as np
import pandas as pd

ATTRS = ["exterior", "size", "price", "strap_pad", "pocket", "interior"]

LEVELS = {
    "exterior": ["black", "reflective", "colorful", "blue"],
    "size": ["small", "large"],
    "price": [120, 140, 160, 180],
    "strap_pad": ["no", "yes"],
    "pocket": ["no", "yes"],
    "interior": ["no_dividers", "file_divider", "laptop"],
}


def load_data(path="../data"):
    design = pd.read_csv(f"{path}/design.csv", header=None, names=ATTRS)
    online = pd.read_csv(f"{path}/responses_online.csv", header=None).values.astype(float)
    offline = pd.read_csv(f"{path}/responses_offline.csv", header=None).values.astype(float)
    return design, online, offline


def encode(design):
    # price goes from -0.5 ($120) to 0.5 ($180), so its coefficient is the effect of moving
    # from the lowest to the highest price; other attributes are dummy coded (first level is the base)
    X = pd.DataFrame(index=design.index)
    X["price"] = (prices(design) - 150) / 60
    for a in ATTRS:
        if a == "price":
            continue
        for i, level in enumerate(LEVELS[a][1:], start=2):
            X[f"{a}_{level}"] = (design[a] == i).astype(int)
    return X


def prices(design):
    return design["price"].map(lambda x: LEVELS["price"][x - 1]).to_numpy()


def all_profiles():
    grid = itertools.product(*[range(1, len(LEVELS[a]) + 1) for a in ATTRS])
    return pd.DataFrame(list(grid), columns=ATTRS)


def to_long(ratings, X):
    n, m = ratings.shape
    df = pd.DataFrame({
        "respondent": np.repeat(np.arange(n), m),
        "profile": np.tile(np.arange(m), n),
        "rating": ratings.ravel(),
    })
    return df.join(X.reset_index(drop=True), on="profile")
