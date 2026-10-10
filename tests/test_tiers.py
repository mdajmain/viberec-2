import numpy as np
import pandas as pd

from src.viberec import tiers


def toy_movies():
    # one genre, 20 movies; counts 1..20 so Q1 = 5.75, median = 10.5, Q3 = 15.25
    counts = np.arange(1, 21)
    scores = np.full(20, 3.0)
    scores[[1, 9, 18]] = 1.0                 # low rating at count 2 (low), 10 (mid), 19 (high)
    scores[11] = 5.0                         # too high at count 12 (mid, above the median)
    scores[6] = 5.0                          # too high at count 7 (mid, below the median)
    return pd.DataFrame({"movieId": np.arange(20), "genres": "Drama", "year": 2000,
                         "rating_count": counts, "mean_rating": scores})


def test_approve_rule_cases():
    out = tiers.approve_rule(toy_movies(), min_votes=1, data_year=2019).set_index("movieId")
    assert out.loc[1, "reason"] == "low_count"            # low count is rejected
    assert out.loc[9, "reason"] == "mid_too_low"          # mid count, low rating
    assert out.loc[18, "approve"]                         # high count approved even with low rating
    assert out.loc[11, "approve"]                         # mid, too high, count >= median
    assert out.loc[6, "reason"] == "mid_too_high_few"     # mid, too high, count < median
    assert out.loc[8, "approve"]                          # mid, normal


def test_recent_movies_always_approved():
    movies = toy_movies().assign(year=2018)
    out = tiers.approve_rule(movies, min_votes=1, data_year=2019, margin_years=8)
    assert out["approve"].all()


def test_assign_tier():
    out = tiers.approve_rule(toy_movies(), min_votes=1, data_year=2019).set_index("movieId")
    tier = pd.Series(tiers.assign_tier(out), index=out.index)
    assert tier[9] == 1          # mid count, too low -> excluded
    assert tier[1] == 1          # low count, too low -> excluded
    assert tier[0] == 2          # low count, normal rating -> reduced weight
    assert tier[6] == 2          # mid count, too high on few ratings -> reduced weight
    assert tier[18] == 3 and tier[8] == 3


def test_upper_score_col_decides_too_high_only():
    movies = toy_movies()
    movies["other"] = movies["mean_rating"]
    movies.loc[6, "other"] = 3.0                          # movie 6 is too high on mean_rating, normal on "other"
    movies.loc[9, "other"] = 3.0                          # movie 9 is too low on mean_rating, normal on "other"
    out = tiers.approve_rule(movies, min_votes=1, data_year=2019, upper_score_col="other").set_index("movieId")
    assert out.loc[6, "approve"]                          # too-high check uses "other"
    assert out.loc[9, "reason"] == "mid_too_low"          # too-low check still uses mean_rating
