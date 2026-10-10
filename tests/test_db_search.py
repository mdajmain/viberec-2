from src.viberec import db_search as ds
from src.viberec import retrieval as rt
import numpy as np


def test_rrf_scores_matches_array_rrf_with_ties():
    a, b = np.array([3.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 2.0, 5.0])
    arr = rt.rrf([a, b])
    d = ds.rrf_scores([dict(enumerate(a)), dict(enumerate(b))])
    assert all(abs(d[i] - arr[i]) < 1e-12 for i in range(4))


def test_rrf_scores_missing_films_rank_just_below_the_list():
    # part 1 scores only films 0 and 1 (e.g. a keyword found in two films); film 2 is tied below them, rank 3
    d = ds.rrf_scores([{0: 2.0, 1: 1.0}, {2: 1.0, 0: 0.5}])
    assert abs(d[2] - (1 / 63 + 1 / 61)) < 1e-12


def test_min_ranks_share_ties():
    assert ds.min_ranks({"a": 3, "b": 3, "c": 1}) == {"a": 1, "b": 1, "c": 3}


def test_ranked_orders_by_score():
    assert ds.ranked({"x": 0.1, "y": 0.3, "z": 0.2}) == ["y", "z", "x"]


def test_analyzer_matches_bm25_tokens():
    bm = rt.BM25(["Café life in the CITY", "two working people"])
    assert ds.bm25_analyzer()("Café life, CITY!") == bm.vec.build_analyzer()("Café life, CITY!")


def test_store_vec_skips_missing_and_zero():
    from src.viberec.store import vec
    assert vec([0, 0]) is None and vec([np.nan, 1]) is None and vec([1, 2.5]) == "[1,2.5]"


def test_search_class_has_the_app_methods():
    # the app calls these; a mis-indented method once ended up outside the class
    for name in ("search", "two_step", "blended", "from_llm"):
        assert callable(getattr(ds.DbSearch, name))
