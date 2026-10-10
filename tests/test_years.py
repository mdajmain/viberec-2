import numpy as np
import pandas as pd

from src.viberec import years as yr


def test_lift_by_value_pools_tail_and_divides_shares():
    lift = yr.lift_by_value([0, 0, 1, 9], [0, 1, 1, 5], max_value=3)
    assert list(lift.index) == [0, 1, 2, 3]
    assert lift[0] == 0.5 / 0.25 and lift[1] == 0.25 / 0.5 and np.isnan(lift[2]) and lift[3] == 1.0


def test_liked_pair_gaps_same_user_closer_when_users_watch_one_era():
    rows = [(u, 1950 + 10 * (u % 6) + k % 3) for u in range(300) for k in range(20)]
    liked = pd.DataFrame(rows, columns=["userId", "year"])
    within, random = yr.liked_pair_gaps(liked, n_users=300, pairs_per_user=20)
    assert within.mean() < random.mean()


def test_lookup_clips_and_keeps_nan():
    curve = pd.Series([3.0, 2.0, 1.0], index=[0, 1, 2])
    out = yr.lookup(curve, [0, 1.4, 7, np.nan])
    assert list(out[:3]) == [3.0, 2.0, 1.0] and np.isnan(out[3])
