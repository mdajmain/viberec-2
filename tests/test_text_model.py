import numpy as np
import pandas as pd
import pytest

from src.viberec import text_model as tm


def test_movie_text_with_and_without_reception():
    assert tm.movie_text("A plot.", "Critics loved it.") == "A plot.\n\nCritics loved it."
    assert tm.movie_text("A plot.", "Critics loved it.", use_reception=False) == "A plot."
    assert tm.movie_text("A plot.", None) == "A plot."


def test_split_ids_is_deterministic_and_disjoint():
    train, test = tm.split_ids(range(1000), 0.2, seed=1)
    train2, test2 = tm.split_ids(list(range(1000))[::-1], 0.2, seed=1)     # order of input does not matter
    assert set(train) & set(test) == set() and len(train) + len(test) == 1000
    assert (np.sort(test) == np.sort(test2)).all()
    assert 150 < len(test) < 250


def test_model_learns_a_word_signal():
    """Synthetic movies: 'ghost' makes a movie scary, 'wedding' makes it romantic. The model must find that."""
    sklearn = pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    filler = ["city", "man", "road", "house", "night", "friend", "money", "car", "letter", "town"]
    texts, y = [], []
    for _ in range(400):
        scary, romantic = rng.random(), rng.random()
        words = list(rng.choice(filler, 30)) + ["ghost"] * int(scary * 6) + ["wedding"] * int(romantic * 6)
        rng.shuffle(words)
        texts.append(" ".join(words))
        y.append([scary, romantic])
    grid = {"tfidf__ngram_range": [(1, 1)], "tfidf__min_df": [1], "ridge__alpha": [0.1, 1.0]}
    search = tm.fit_tfidf_ridge(texts[:300], np.array(y[:300]), grid, cv=3)
    pred = search.best_estimator_.predict(texts[300:])
    assert tm.mean_spearman(np.array(y[300:]), pred) > 0.8
    pos, _ = tm.top_words(search.best_estimator_, 0, n=1)
    assert pos == ["ghost"]


def test_strip_review_scores_keeps_vibe_sentences():
    rec = ("On Rotten Tomatoes the film holds an approval rating of 84% based on 122 reviews. "
           "On Metacritic it has a score of 71 out of 100, indicating generally favorable reviews. "
           "Critics called it a bittersweet slice of observational comedy. "
           "The film received mixed reviews from critics.")
    out = tm.strip_review_scores(rec)
    assert out == "Critics called it a bittersweet slice of observational comedy."
    assert tm.movie_text("P.", rec, strip_scores=True) == "P.\n\nCritics called it a bittersweet slice of observational comedy."


def test_fit_tfidf_logistic_learns_word_cue():
    texts = ["they lived happily ever after"] * 20 + ["she died alone in the rain"] * 20
    labels = ["happy"] * 20 + ["sad"] * 20
    search = tm.fit_tfidf_logistic(texts, labels, {"logreg__C": [1.0, 10.0], "tfidf__min_df": [1]}, cv=2)
    assert list(search.best_estimator_.predict(["happily ever after", "died in the rain"])) == ["happy", "sad"]
