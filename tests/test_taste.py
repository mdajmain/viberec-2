import numpy as np

from src.viberec import cf
from src.viberec import taste as ts


def notebook_recommend(ids, factors, yty, z, pool_mask, rating_count, alpha, reg, w_als, answers, k):
    """A line-by-line copy of notebook 10's recommend_new_user (cell 6), on plain arrays."""
    pool_ids = ids[pool_mask]
    pool_cols = np.where(pool_mask)[0]
    m_index = {m: i for i, m in enumerate(ids)}
    vibe_unit = (z / np.linalg.norm(z, axis=1, keepdims=True))[pool_mask]
    popular_order = pool_ids[np.argsort(-rating_count[pool_mask], kind="stable")]
    asked = {m for m, _ in answers}
    likes = [m for m, a in answers if a == "like" and m in m_index]
    if not likes:
        return [m for m in popular_order if m not in asked][:k]
    als = factors[pool_cols] @ cf.fold_in_user(factors, yty, [m_index[m] for m in likes], alpha, reg)
    taste = z[[m_index[m] for m in likes]].mean(axis=0)
    vib = vibe_unit @ (taste / (np.linalg.norm(taste) or 1))
    score = w_als * ts.standardised(als) + (1 - w_als) * ts.standardised(vib)
    score[np.isin(pool_ids, list(asked))] = -np.inf
    top = np.argpartition(-score, k)[:k]
    return list(pool_ids[top[np.argsort(-score[top])]])


def make_model(seed=0, n=300, f=8, a=5):
    rng = np.random.default_rng(seed)
    ids = np.arange(1000, 1000 + n)
    factors = rng.normal(size=(n, f)).astype(np.float32)
    z = rng.normal(size=(n, a))
    tier = np.where(rng.random(n) < 0.8, 3, 2)
    count = rng.integers(1, 1000, n)
    yty = factors.T.astype(float) @ factors.astype(float)
    m = ts.TasteModel(ids, factors, z, tier, count, yty, alpha=10.0, reg=0.1, w_als=0.2, attrs=[f"a{i}" for i in range(a)])
    return m, ids, factors, z, tier == 3, count, yty


def test_feed_matches_notebook_10_exactly():
    m, ids, factors, z, pool, count, yty = make_model()
    rng = np.random.default_rng(1)
    for _ in range(20):
        picked = rng.choice(ids, 12, replace=False)
        answers = [(int(x), rng.choice(["like", "dislike", None])) for x in picked]
        likes = [x for x, a in answers if a == "like"]
        expected = notebook_recommend(ids, factors, yty, z, pool, count, 10.0, 0.1, 0.2, answers, 10)
        assert m.recommend(likes, exclude=[x for x, _ in answers], k=10) == [int(x) for x in expected]


def test_no_likes_falls_back_to_most_rated_unasked():
    m, ids, factors, z, pool, count, yty = make_model()
    first = int(ids[pool][np.argmax(count[pool])])
    assert m.recommend([], exclude=[], k=1) == [first]
    assert first not in m.recommend([], exclude=[first], k=5)


def test_quiz_stops_at_20_likes_or_100_answers():
    qs = list(range(1, 151))
    assert ts.next_question(qs, {}) == 1
    assert ts.next_question(qs, {1: "unseen", 2: "like"}) == 3
    assert ts.next_question(qs, {i: "like" for i in range(1, 21)}) is None
    assert ts.next_question(qs, {i: "unseen" for i in range(1, 101)}) is None
    assert ts.next_question([1, 2], {1: "like", 2: "dislike"}) is None


def test_summary_reads_the_likes():
    m, ids, *_ = make_model()
    m.year = np.full(len(ids), 2005.0)
    m.genres = np.array([["Comedy", "Romance"]] * len(ids), dtype=object)
    s = m.summary([int(ids[0]), int(ids[1])], n_dislikes=1)
    assert s["likes"] == 2 and s["dislikes"] == 1 and s["era"] == "2005" and s["top_genres"] == ["Comedy", "Romance"]
    assert len(s["top_vibes"]) >= 1
