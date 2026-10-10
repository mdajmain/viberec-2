import numpy as np
import pandas as pd
import pytest

from src.viberec import credibility as cr


def counts_from_ratings(ratings_per_movie):
    rows = []
    for ratings in ratings_per_movie:
        rows.append([np.sum(np.isclose(ratings, level)) for level in cr.RATING_LEVELS])
    return np.array(rows, dtype=np.int64)


def test_hist_stats_matches_numpy():
    data = [[4.0, 5.0, 3.5, 4.0], [1.0, 2.5], [5.0]]
    n, mean, var = cr.hist_stats(counts_from_ratings(data))
    assert list(n) == [4, 2, 1]
    assert mean == pytest.approx([np.mean(x) for x in data])
    assert var[:2] == pytest.approx([np.var(x, ddof=1) for x in data[:2]])
    assert np.isnan(var[2])


def test_build_rating_histogram(tmp_path):
    csv = tmp_path / "ratings.csv"
    pd.DataFrame({"movieId": [1, 1, 2, 3], "rating": [4.0, 4.0, 0.5, 5.0]}).to_csv(csv, index=False)
    hist = cr.build_rating_histogram(csv, [1, 2], chunksize=2)
    assert hist["movieId"].tolist() == [1, 2]                      # movie 3 is not requested
    assert hist.loc[0, "r4.0"] == 2 and hist.loc[1, "r0.5"] == 1
    assert hist[cr.LEVEL_COLS].to_numpy().sum() == 3


def test_split_half_sizes_and_content():
    rng = np.random.default_rng(0)
    counts = counts_from_ratings([[1.0] * 3 + [5.0] * 4, [3.0] * 10, [4.0]])
    half, mean_a, mean_b = cr.split_half_means(counts, rng)
    assert list(half) == [3, 5, 0]                                  # n // 2 ratings in each half
    assert mean_a[1] == pytest.approx(3.0) and mean_b[1] == pytest.approx(3.0)
    assert np.isnan(mean_a[2]) and np.isnan(mean_b[2])              # a single rating cannot be split


def test_split_half_means_average_to_full_mean():
    rng = np.random.default_rng(1)
    counts = counts_from_ratings([[1.0] * 500 + [5.0] * 500])
    _, mean_a, mean_b = cr.split_half_means(np.repeat(counts, 2000, axis=0), rng)
    assert mean_a.mean() == pytest.approx(3.0, abs=0.01)
    assert mean_b.mean() == pytest.approx(3.0, abs=0.01)


def test_bootstrap_matches_analytic_standard_error():
    rng = np.random.default_rng(2)
    ratings = [np.r_[np.full(300, 2.0), np.full(500, 4.0), np.full(200, 5.0)]]
    counts = counts_from_ratings(ratings)
    var, lo, hi = cr.bootstrap_summary(counts, 2000, rng)
    plug_in = np.var(ratings[0]) / len(ratings[0])
    assert var[0] == pytest.approx(plug_in, rel=0.1)
    assert lo[0] < np.mean(ratings[0]) < hi[0]


def test_bootstrap_of_a_single_rating_has_no_spread():
    var, lo, hi = cr.bootstrap_summary(counts_from_ratings([[4.0]]), 50, np.random.default_rng(3))
    assert var[0] == 0 and lo[0] == hi[0] == 4.0


def test_make_strata_puts_single_ratings_in_stratum_zero():
    n = np.array([1, 1, 2, 3, 4, 10, 50, 200, 1000, 5000])
    strata = cr.make_strata(n, 4)
    assert list(strata[:2]) == [0, 0] and (strata[2:] >= 1).all()
    assert (np.diff(strata[2:]) >= 0).all()                         # higher n never lands in a lower stratum


def test_split_half_r_and_spearman_brown():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(5000, 1))
    a = truth + rng.normal(scale=1.0, size=(5000, 1))
    b = truth + rng.normal(scale=1.0, size=(5000, 1))
    r = cr.split_half_r(a, b)[0]
    assert r == pytest.approx(0.5, abs=0.03)                        # signal var 1, noise var 1
    assert cr.spearman_brown(0.5) == pytest.approx(2 / 3)


def test_n_at_level_interpolates_and_handles_out_of_range():
    curve = np.array([[2.0, 0.2], [8.0, 0.5], [32.0, 0.8], [128.0, 0.95]])
    assert cr.n_at_level(curve, 0.5) == pytest.approx(8.0)
    assert 8.0 < cr.n_at_level(curve, 0.65) < 32.0
    assert np.isnan(cr.n_at_level(curve, 0.99))


def test_add_enough_ratings_flags_rounds_up_and_keeps_rows():
    movies = pd.DataFrame({"rating_count": [7, 8, 27, 28, 100]})
    table = pd.DataFrame({"reliability": [0.5, 0.8], "ratings_needed": [7.3, 27.4]})
    out, minimums = cr.add_enough_ratings_flags(movies, table)
    assert minimums == {"enough_r50": 8, "enough_r80": 28}
    assert list(out["enough_r50"]) == [False, True, True, True, True]
    assert list(out["enough_r80"]) == [False, False, False, True, True]
    assert len(out) == len(movies)
    assert "enough_r50" not in movies.columns          # input untouched


def test_build_early_counts(tmp_path):
    csv = tmp_path / "ratings.csv"
    pd.DataFrame({"userId": [1, 2, 3, 4, 5], "movieId": [1, 1, 1, 2, 9],
                  "rating": [4.0, 2.0, 5.0, 3.0, 1.0], "timestamp": [10, 20, 30, 40, 50]}).to_csv(csv, index=False)
    out = cr.build_early_counts(csv, [1, 2], [0.5])
    cutoff = out["cutoff_ts"].iloc[0]
    assert cutoff == 30                                    # lower quantile of [10, 20, 30, 40, 50]
    row1 = out[out["movieId"] == 1].iloc[0]
    assert (row1["n_early"], row1["sum_early"]) == (2, 6.0)    # ratings 4.0 and 2.0 came before 30
    assert out[out["movieId"] == 2].iloc[0]["n_early"] == 0
    assert set(out["movieId"]) == {1, 2}                   # movie 9 is not requested


def test_shrunk_mean_limits():
    assert cr.shrunk_mean(0, np.nan, 3.0, 7.0) == pytest.approx(3.0)           # no ratings -> prior
    assert cr.shrunk_mean(1_000_000, 4.5, 3.0, 7.0) == pytest.approx(4.5, abs=1e-4)
    assert cr.shrunk_mean(7, 5.0, 3.0, 7.0) == pytest.approx(4.0)              # equal weight at n = k


def test_hard_weight_mean():
    out = cr.hard_weight_mean(np.array([5, 28, 100]), np.array([5.0, 4.0, 4.5]), 3.0, 28)
    assert list(out) == [3.0, 4.0, 4.5]


def test_best_k_recovers_known_shrinkage():
    rng = np.random.default_rng(0)
    true = rng.normal(3.0, 0.5, 20000)                      # true movie quality
    n = rng.integers(1, 40, 20000)
    mean = true + rng.normal(0, 1.0, 20000) / np.sqrt(n)    # noisy observed mean, noise variance 1/n
    k = cr.best_k(n, mean, true, prior=3.0)                 # ideal k = noise var / true var = 1 / 0.25 = 4
    assert k == pytest.approx(4.0, rel=0.15)
