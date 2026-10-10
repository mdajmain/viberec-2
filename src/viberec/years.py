"""Release years in search (PROJECT.md §7 release-year proximity; Adil: older films should not be the top pick).

Two curves, both measured from ratings, not set by hand:
- year-gap lift: how much more often one person likes two films that are g years apart than two random liked films;
- age lift: how much present-day viewers watch films of each age, relative to that age's share of the catalog.
"""
import numpy as np
import pandas as pd


def liked_pair_gaps(liked, n_users, pairs_per_user, seed=0):
    """liked: DataFrame userId, year (one row per liked rating). Returns (within, random): year gaps between two
    films liked by the same user, and between two random liked films (same number of pairs)."""
    rng = np.random.default_rng(seed)
    users = liked["userId"].unique()
    users = rng.choice(users, min(n_users, len(users)), replace=False)
    sub = liked[liked["userId"].isin(users)]
    within = []
    for ys in sub.groupby("userId")["year"].apply(np.asarray):
        if len(ys) < 2:
            continue
        i, j = rng.integers(0, len(ys), pairs_per_user), rng.integers(0, len(ys), pairs_per_user)
        keep = i != j
        within.append(np.abs(ys[i[keep]] - ys[j[keep]]))
    within = np.concatenate(within)
    ys = sub["year"].to_numpy()
    random = np.abs(ys[rng.integers(0, len(ys), len(within))] - ys[rng.integers(0, len(ys), len(within))])
    return within, random


def lift_by_value(observed, reference, max_value):
    """Share of observed at each whole value 0..max_value (values above pooled into max_value) divided by the share
    of reference there. Returns a Series indexed 0..max_value."""
    def shares(x):
        x = np.minimum(np.round(np.asarray(x, float)[np.isfinite(x)]).astype(int), max_value)
        x = x[x >= 0]
        return np.bincount(x, minlength=max_value + 1) / len(x)
    o, r = shares(observed), shares(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.where(r > 0, o / r, np.nan), index=np.arange(max_value + 1))


def lookup(curve, values):
    """curve value for each number in values (rounded, clipped to the curve's range); NaN stays NaN."""
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    idx = np.clip(np.round(v[ok]).astype(int), curve.index.min(), curve.index.max())
    out[ok] = curve.reindex(idx).to_numpy()
    return out
