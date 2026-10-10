import numpy as np
import pandas as pd

from src.viberec import cf


def toy_ratings():
    # two taste groups: users 1-3 love movies 10-12, users 4-6 love movies 20-22
    rows = []
    for u in (1, 2, 3):
        rows += [(u, m, 5.0) for m in (10, 11, 12)] + [(u, 20, 1.0)]
    for u in (4, 5, 6):
        rows += [(u, m, 5.0) for m in (20, 21, 22)] + [(u, 10, 1.0)]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def test_build_matrix_kinds():
    r = toy_ratings()
    w, ui, mi = cf.build_matrix(r, "watched")
    l, _, _ = cf.build_matrix(r, "liked")
    s, _, _ = cf.build_matrix(r, "rating")
    assert w.sum() == len(r) and l.sum() == 18 and s[ui[1], mi[20]] == 1.0


def test_svd_recommends_the_users_group():
    r = toy_ratings()
    m, ui, mi = cf.build_matrix(r, "liked")
    v = cf.pure_svd(m, k=2)
    seen = pd.DataFrame({"movieId": [10, 11], "rating": [5.0, 5.0]})        # a new-ish view of user 1
    rec = cf.make_scorer_recommender(cf.svd_user_scores(m, ui, v), mi.index, mi, k=1)
    ids, scanned = rec(1, seen)
    assert ids == [12] and scanned == 6                                       # the unseen movie from its own group


def test_blend_standardises_each_scorer():
    a = lambda u, r: np.array([1.0, 2.0, 3.0])
    b = lambda u, r: np.array([300.0, 200.0, 100.0])                           # opposite order, much larger scale
    s = cf.blend_user_scores([a, b], [1.0, 1.0])(0, None)
    assert np.allclose(s, 0)                                                    # equal weight cancels despite scale


def test_blend_treats_unscorable_movies_as_neutral():
    a = lambda u, r: np.array([1.0, 2.0, 3.0])
    b = lambda u, r: np.array([np.nan, 0.0, 1.0])                          # first movie has no vibe score
    s = cf.blend_user_scores([a, b], [0.5, 0.5])(0, None)
    assert np.isfinite(s).all() and s[2] > s[1] > s[0]


def test_vibe_user_scores_aligns_with_cf_columns():
    vibes = pd.DataFrame({"dark": [1.0, 0.0, 0.9], "funny": [0.0, 1.0, 0.1]}, index=[10, 20, 30])
    m_index = pd.Series([0, 1, 2, 3], index=[30, 10, 99, 20])                # 99 has no vibe scores
    sc = cf.vibe_user_scores(vibes, m_index, pd.Series({7: 3.0}))
    s = sc(7, pd.DataFrame({"movieId": [10], "rating": [5.0]}))             # liked the dark movie
    assert np.isnan(s[2]) and s[0] > s[3] and s[1] > s[3]


def test_blend_skips_a_scorer_with_nothing_to_say():
    a = lambda u, r: np.array([1.0, 2.0, 3.0])
    empty = lambda u, r: np.full(3, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # any RuntimeWarning fails the test
        s = cf.blend_user_scores([a, empty], [0.5, 0.5])(0, None)
    assert s[2] > s[1] > s[0]


def test_fold_in_matches_als_user_factors():
    """A user folded in from their liked movies gets the factors ALS itself learned for them."""
    import pytest
    implicit = pytest.importorskip("implicit")
    from implicit.als import AlternatingLeastSquares
    rng = np.random.default_rng(0)
    m = (rng.random((300, 80)) < 0.1).astype(np.float32)
    import scipy.sparse as sp
    m = sp.csr_matrix(m)
    model = AlternatingLeastSquares(factors=8, regularization=0.1, alpha=10.0, iterations=20, random_state=0, use_cg=False)
    model.fit(m, show_progress=False)
    y = np.asarray(model.item_factors)
    for u in (0, 5, 17):
        reference = np.asarray(model.recalculate_user(u, m[u])).ravel()     # implicit's own exact user step
        folded = cf.fold_in_user(y, y.T @ y, m[u].indices, alpha=10.0, reg=0.1)
        assert np.allclose(folded, reference, atol=1e-4)


def test_question_weights_prefer_seen_and_divisive():
    w = cf.question_weights([0.9, 0.9, 0.1], [0.5, 0.95, 0.5])
    assert w[0] > w[1] and w[0] > w[2]                     # widely seen and divisive beats loved-by-all or rarely seen


def test_informative_questions_spread_across_directions():
    y = np.array([[1, 0], [0.99, 0.01], [0, 1]], float)    # two near-identical movies and one different
    chosen = cf.informative_questions(y, np.ones(3), n=2, reg=0.1)
    assert set(chosen) in ({0, 2}, {1, 2})                 # never both near-duplicates


def test_informative_questions_incremental_matches_direct():
    """The fast rank-one bookkeeping picks the same questions as recomputing y' cov y from scratch."""
    rng = np.random.default_rng(4)
    y, w = rng.normal(size=(60, 5)), rng.random(60)
    fast = cf.informative_questions(y, w, n=10, reg=0.5)
    yw = y * np.sqrt(w)[:, None]
    cov, chosen = np.eye(5) / 0.5, []
    for _ in range(10):
        g = np.einsum("ij,jk,ik->i", yw, cov, yw)
        g[chosen] = -np.inf
        b = int(np.argmax(g)); chosen.append(b)
        v = cov @ yw[b]; cov -= np.outer(v, v) / (1 + yw[b] @ v)
    assert fast == chosen


def test_informative_questions_respect_quota():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(40, 4))
    groups = np.array(["1990s"] * 20 + ["2000+"] * 20)
    chosen = cf.informative_questions(y, np.ones(40), n=10, reg=0.5, groups=groups, quota={"1990s": 2, "2000+": 8})
    picked = groups[chosen]
    assert len(chosen) == 10 and (picked == "1990s").sum() == 2 and (picked == "2000+").sum() == 8
    assert len(set(chosen)) == 10


def test_informative_questions_continue_from_a_start():
    rng = np.random.default_rng(6)
    y = rng.normal(size=(50, 4))
    first = cf.informative_questions(y, np.ones(50), n=10, reg=0.5)
    longer = cf.informative_questions(y, np.ones(50), n=25, reg=0.5, start=first)
    assert longer[:10] == first and len(set(longer)) == 25
    assert longer == cf.informative_questions(y, np.ones(50), n=25, reg=0.5)   # greedy: continuing = starting over
