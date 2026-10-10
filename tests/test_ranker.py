import numpy as np
import pandas as pd

from src.viberec import ranker as rk


def test_recent_counts_window():
    r = pd.DataFrame({"movieId": [1, 1, 2, 2], "timestamp": [5, 15, 18, 25]})
    c = rk.recent_counts(r, cutoff_ts=20, window_seconds=10)
    assert c.to_dict() == {1: 1, 2: 1}                       # 5 too old, 25 after the cutoff


def test_user_rows_features():
    movies = pd.DataFrame({"shrunk": [3.5, 4.0, 2.0], "tier": [3, 3, 2], "log_count": [1.0, 2.0, 0.5],
                           "log_recent": [0.0, 1.0, 0.0], "year": [2000, 2010, 1990]}, index=[10, 20, 30])
    m_index = pd.Series([0, 1, 2], index=[10, 20, 30])
    ratings = pd.DataFrame({"movieId": [10], "rating": [5.0]})
    rows = rk.user_rows(7, ratings, 3.0, [20, 30], np.array([0, 2.0, 1.0]), np.array([np.nan, np.nan, np.nan]),
                        np.array([0, 5.0, 1.0]), movies, m_index)
    assert list(rows["blend_rank"]) == [1, 2] and list(rows["year_gap"]) == [10, 10]   # liked year 2000
    assert (rows["vibe_z"] == 0).all() and rows["als_z"].iloc[0] > rows["als_z"].iloc[1]


def test_top_k_and_familiarity():
    rows = pd.DataFrame({"userId": [1, 1, 1], "movieId": [5, 6, 7]})
    assert rk.top_k_by(rows, np.array([0.1, 0.9, 0.5]), k=2) == {1: [6, 7]}
    assert rk.familiarity([6, 7], {6: 0.9, 7: 0.5}) == 0.7
