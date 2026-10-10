"""Credibility of a movie's mean rating (Q4.1).

Everything works from a per-movie histogram of the 10 possible ratings (0.5 ... 5.0).
That table is tiny compared to the 25M raw ratings, and resampling from it is equivalent
to resampling the raw ratings.
"""
import numpy as np
import pandas as pd

RATING_LEVELS = np.arange(0.5, 5.01, 0.5)
LEVEL_COLS = [f"r{r:.1f}" for r in RATING_LEVELS]


def build_rating_histogram(ratings_csv, movie_ids, chunksize=5_000_000):
    """One row per movie in movie_ids, one column per rating level holding the number of ratings."""
    movie_ids = np.sort(np.asarray(movie_ids))
    size = (int(movie_ids.max()) + 1) * 10
    counts = np.zeros(size, dtype=np.int64)
    for chunk in pd.read_csv(ratings_csv, usecols=["movieId", "rating"],
                             dtype={"movieId": "int32", "rating": "float32"}, chunksize=chunksize):
        level = (chunk["rating"].to_numpy() * 2 - 1).astype(np.int64)      # 0.5 -> 0, ..., 5.0 -> 9
        keys = chunk["movieId"].to_numpy().astype(np.int64) * 10 + level
        keys = keys[keys < size]                                           # movies above the largest requested id
        counts += np.bincount(keys, minlength=size)
    hist = pd.DataFrame(counts.reshape(-1, 10)[movie_ids], columns=LEVEL_COLS)
    hist.insert(0, "movieId", movie_ids)
    return hist


def hist_stats(counts):
    """Number of ratings, mean and unbiased variance per movie (variance is NaN when n < 2)."""
    n = counts.sum(axis=1)
    mean = (counts @ RATING_LEVELS) / n
    sum_sq = counts @ RATING_LEVELS**2
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (sum_sq - n * mean**2) / (n - 1)
    var[n < 2] = np.nan
    return n, mean, var


def split_half_means(counts, rng):
    """Split each movie's ratings at random into two halves of n // 2 ratings and return both means.

    Drawing the halves level by level from a hypergeometric distribution is the same as
    shuffling the ratings and cutting them in two. With an odd n, one rating is left out.
    """
    n = counts.sum(axis=1)
    half = n // 2

    def draw(available, size):
        remaining_total = available.sum(axis=1)
        remaining_draw = size.copy()
        taken = np.zeros_like(available)
        for level in range(available.shape[1]):
            good = available[:, level]
            taken[:, level] = rng.hypergeometric(good, remaining_total - good, remaining_draw)
            remaining_total = remaining_total - good
            remaining_draw = remaining_draw - taken[:, level]
        return taken

    half_a = draw(counts, half)
    half_b = draw(counts - half_a, half)
    safe = np.where(half > 0, half, 1)
    mean_a = (half_a @ RATING_LEVELS) / safe
    mean_b = (half_b @ RATING_LEVELS) / safe
    mean_a[half == 0] = np.nan
    mean_b[half == 0] = np.nan
    return half, mean_a, mean_b


def bootstrap_summary(counts, n_boot, rng, chunk=1000):
    """Bootstrap variance and 95% interval of each movie's mean (n_boot resamples per movie)."""
    n = counts.sum(axis=1)
    p = counts / n[:, None]
    var = np.empty(len(n))
    lo = np.empty(len(n))
    hi = np.empty(len(n))
    for start in range(0, len(n), chunk):
        s = slice(start, start + chunk)
        draws = rng.multinomial(n[s], p[s], size=(n_boot, len(n[s])))       # (n_boot, movies, 10)
        means = (draws @ RATING_LEVELS) / n[s][None, :]
        var[s] = means.var(axis=0, ddof=1)
        lo[s], hi[s] = np.percentile(means, [2.5, 97.5], axis=0)
    return var, lo, hi


def make_strata(n, n_bins):
    """Stratum 0 = movies with a single rating; the rest in quantile bins of n (ties merged)."""
    strata = np.zeros(len(n), dtype=int)
    many = n >= 2
    bins = pd.qcut(n[many], n_bins, duplicates="drop")
    upper = np.array([interval.right for interval in bins.categories])
    strata[many] = np.searchsorted(upper, n[many], side="left") + 1
    return strata


def split_half_r(mean_a, mean_b):
    """Correlation of the two half-means across movies, one value per repeat (columns)."""
    a = mean_a - mean_a.mean(axis=0)
    b = mean_b - mean_b.mean(axis=0)
    return (a * b).sum(axis=0) / np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))


def spearman_brown(r):
    """Reliability of the full-length mean from the correlation between two half-length means."""
    return 2 * r / (1 + r)


def stratum_curve(movie_idx, n, strata, half_means, min_movies=50, n_repeats=None):
    """Split-half reliability by stratum for the movies in movie_idx.

    Returns an array with one row per stratum: [harmonic-mean n, reliability].
    half_means has shape (movies, repeats, 2).
    """
    rows = []
    for s in range(1, strata.max() + 1):
        members = movie_idx[strata[movie_idx] == s]
        if len(members) < min_movies:
            continue
        a = half_means[members, :n_repeats, 0]
        b = half_means[members, :n_repeats, 1]
        rows.append((1 / np.mean(1 / n[members]), spearman_brown(split_half_r(a, b).mean())))
    return np.array(rows)


def n_at_level(curve, level):
    """Number of ratings at which the reliability curve reaches `level` (log-linear interpolation)."""
    n_values, reliability = curve[:, 0], np.maximum.accumulate(curve[:, 1])
    if reliability.max() < level or reliability.min() > level:
        return np.nan
    return float(np.exp(np.interp(level, reliability, np.log(n_values))))


def add_enough_ratings_flags(movies, nstar_table):
    """Add one True/False column per reliability level: does the movie have enough ratings?

    nstar_table has columns `reliability` and `ratings_needed`. The needed count is rounded up,
    because a movie with fewer ratings than the interpolated value has not reached that level.
    Nothing is dropped; the flags only mark which movies pass. Returns (movies, {column: minimum n}).
    """
    out = movies.copy()
    minimums = {}
    for level, needed in zip(nstar_table["reliability"], nstar_table["ratings_needed"]):
        col = f"enough_r{round(level * 100)}"
        minimums[col] = int(np.ceil(needed))
        out[col] = out["rating_count"] >= minimums[col]
    return out, minimums


def build_early_counts(ratings_csv, movie_ids, quantiles):
    """For each cutoff (a quantile of all rating timestamps): per movie, how many ratings came before it
    and their sum. Ratings at or after the cutoff are the "late" ratings (total minus early).

    Returns a long table: movieId, quantile, cutoff_ts, n_early, sum_early.
    """
    ratings = pd.read_csv(ratings_csv, usecols=["movieId", "rating", "timestamp"],
                          dtype={"movieId": "int32", "rating": "float32", "timestamp": "int64"})
    cutoffs = np.quantile(ratings["timestamp"].to_numpy(), quantiles, method="lower")
    movie_ids = np.sort(np.asarray(movie_ids))
    size = int(movie_ids.max()) + 1
    ids = ratings["movieId"].to_numpy().astype(np.int64)
    keep = ids < size
    tables = []
    for q, cutoff in zip(quantiles, cutoffs):
        early = keep & (ratings["timestamp"].to_numpy() < cutoff)
        n_early = np.bincount(ids[early], minlength=size)
        sum_early = np.bincount(ids[early], weights=ratings["rating"].to_numpy()[early].astype(np.float64), minlength=size)
        tables.append(pd.DataFrame({"movieId": movie_ids, "quantile": q, "cutoff_ts": int(cutoff),
                                    "n_early": n_early[movie_ids], "sum_early": sum_early[movie_ids]}))
    return pd.concat(tables, ignore_index=True)


def shrunk_mean(n, mean, prior, k):
    """Mean pulled toward `prior` as if the movie had k extra ratings equal to the prior. n = 0 gives the prior."""
    n = np.asarray(n, dtype=float)
    return (n * np.nan_to_num(mean) + k * prior) / (n + k)


def hard_weight_mean(n, mean, prior, min_n):
    """The raw mean when the movie has at least min_n ratings, otherwise the prior."""
    return np.where(np.asarray(n) >= min_n, mean, prior)


def best_k(n, mean, target, prior, k_max=200.0):
    """The k in [0, k_max] that minimises squared error between shrunk_mean and target."""
    from scipy.optimize import minimize_scalar
    loss = lambda k: np.mean((shrunk_mean(n, mean, prior, k) - target) ** 2)
    return float(minimize_scalar(loss, bounds=(0.0, k_max), method="bounded").x)
