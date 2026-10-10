import numpy as np
import pandas as pd

from src.viberec import onboarding as ob


def make_q():
    ids = np.array([10, 20, 30, 40])
    y = np.array([[1, 0], [0.99, 0.01], [0, 1], [0.5, 0.5]], float)
    seen = np.array([0.9, 0.8, 0.5, 0.1])
    like = np.array([0.5, 0.5, 0.5, 0.5])
    cowatch = np.array([[1, .9, .1, .1], [.9, 1, .1, .1], [.1, .1, 1, .9], [.1, .1, .9, 1]], float)
    return ob.Questionnaire(ids, y, seen, like, cowatch, reg=0.1)


def test_answers_from_history():
    ans = ob.answer_from_history(pd.DataFrame({"movieId": [10, 20], "rating": [5.0, 2.0]}), user_mean=3.5)
    assert ans(10) == "like" and ans(20) == "dislike" and ans(99) is None


def test_fixed_strategies():
    q = make_q()
    assert q.fixed_order("popular", 2) == [0, 1]
    assert set(q.fixed_order("informative", 2)) != {0, 1}           # informative avoids two near-duplicates
    assert len(set(q.fixed_order("random", 4, seed=1))) == 4


def test_adaptive_never_repeats_and_uses_seen_pattern():
    q = make_q()
    out = q.run("adaptive", lambda m: "like" if m in (30, 40) else None, n=4)
    asked = [m for m, _ in out]
    assert len(set(asked)) == 4
    # once the user has seen 30, the co-watched 40 becomes a likely-seen, worthwhile question
    assert asked.index(40) > asked.index(30) if 30 in asked[:3] else True
