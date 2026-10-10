import numpy as np
import pandas as pd
import pytest

from src.viberec import evaluation as ev
from src.viberec import recommenders as rc


def test_recall_and_ndcg():
    assert ev.recall_at_k([1, 2, 3], {2, 9}, 2) == 0.5
    assert ev.ndcg_at_k([2, 1], {2}, 10) == 1.0                       # hit at the top = perfect
    assert ev.ndcg_at_k([1, 2], {2}, 10) == pytest.approx(1 / np.log2(3))
    assert np.isnan(ev.recall_at_k([1], set(), 1))


def test_coverage_diversity_novelty():
    assert ev.coverage([[1, 2], [2, 3]], pool_size=10, k=2) == 0.3
    vec = {1: np.array([1.0, 0]), 2: np.array([1.0, 0]), 3: np.array([0, 1.0])}
    assert ev.diversity([1, 2], vec, 10) == pytest.approx(0)          # identical vibes
    assert ev.diversity([1, 3], vec, 10) == pytest.approx(1)          # orthogonal vibes
    assert ev.novelty([1], {1: 0.25}, 10) == pytest.approx(2)


def test_make_benchmark_uses_only_past_and_user_relative_hits():
    r = pd.DataFrame({"userId": [1, 1, 1, 1, 2, 2], "movieId": [10, 11, 12, 13, 10, 14],
                      "rating": [2.0, 4.0, 5.0, 2.0, 3.0, 3.0], "timestamp": [1, 2, 10, 11, 1, 10]})
    train, hits, mean, eligible = ev.make_benchmark(r, cutoff_ts=5, n_users=10)
    assert set(train["timestamp"]) == {1, 2}                         # nothing from the future
    assert hits == {1: {12}}                                         # user 1 mean 3.0: 5.0 is a hit, 2.0 is not
    assert eligible == 1                                             # user 2's 3.0 is not above their 3.0 mean


def test_recommenders_skip_seen_movies_and_rank():
    ranked = rc.make_ranked_list([1, 2, 3, 4], {1: 9, 2: 8, 3: 7, 4: 1}, k=2)
    recs, scanned = ranked(7, pd.DataFrame({"movieId": [1], "rating": [5.0]}))
    assert recs == [2, 3] and scanned == 4
    vibes = pd.DataFrame({"dark": [1.0, 0.9, 0.0, 0.1], "funny": [0.0, 0.1, 1.0, 0.9]}, index=[1, 2, 3, 4])
    rec = rc.make_vibe_match(vibes, [1, 2, 3, 4], pd.Series({7: 3.0}), k=2)
    recs, _ = rec(7, pd.DataFrame({"movieId": [1, 3], "rating": [5.0, 1.0]}))   # liked the dark movie
    assert recs[0] == 2                                              # closest dark movie, not already seen


def test_scoreboard_is_append_only(tmp_path):
    p = tmp_path / "s.csv"
    ev.append_scoreboard(p, "x", "a", {"recall@10": 0.1})
    ev.append_scoreboard(p, "x", "b", {"recall@10": 0.2, "new_metric": 1.0})
    s = pd.read_csv(p)
    assert s["method"].tolist() == ["a", "b"] and s.loc[0, "recall@10"] == 0.1


def test_paired_gain():
    a = pd.DataFrame({"recall@10": [0.2, 0.4, 0.6]}, index=[1, 2, 3])
    b = pd.DataFrame({"recall@10": [0.1, 0.3, 0.5]}, index=[1, 2, 3])
    diff, lo, hi, rel = ev.paired_gain(a, b, "recall@10", n_boot=200)
    assert diff == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    assert rel == pytest.approx(0.1 / 0.3)


def test_scoreboard_does_not_duplicate_reruns(tmp_path):
    p = tmp_path / "s.csv"
    ev.append_scoreboard(p, "x", "a", {"recall@10": 0.1}, settings={"k": 1})
    ev.append_scoreboard(p, "x", "a", {"recall@10": 0.1}, settings={"k": 1})      # same run again
    ev.append_scoreboard(p, "x", "a", {"recall@10": 0.2}, settings={"k": 2})      # a real new variant
    assert len(pd.read_csv(p)) == 2
