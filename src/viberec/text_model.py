"""Predict Genome vibe attributes from a movie's text: a free alternative to LLM tagging (Step 5).

Training data: movies that have both a Wikipedia text and Genome scores. Input = text, target = the
Genome-checkable vibe attributes (0-1). Model = TF-IDF features + ridge regression, one weight per
word and attribute. All settings (TF-IDF options, ridge strength alpha) are chosen by cross-validation
on the training movies only; the test movies are used once, for the final score.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REVIEW_SCORE = re.compile(r"rotten tomatoes|metacritic|cinemascore|approval rating|average rating|average score|"
                          r"weighted average|out of 100|out of 10\b|review aggregat|polled audiences|"
                          r"(generally|universal) (favorable|unfavorable|acclaim)|mixed or average|"
                          r"(positive|negative|mixed) reviews", re.I)


def strip_review_scores(text):
    """Drop sentences that report review scores or how reviews leaned overall. These are about quality, not vibe,
    and they appear in most Reception sections in near-identical wording."""
    if not isinstance(text, str):
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if not REVIEW_SCORE.search(s))


def movie_text(plot, reception=None, use_reception=True, strip_scores=False):
    """The text the model reads: the plot, optionally followed by the critics' reception
    (optionally without the review-score sentences)."""
    parts = [plot if isinstance(plot, str) else ""]
    if use_reception and isinstance(reception, str):
        parts.append(strip_review_scores(reception) if strip_scores else reception)
    return "\n\n".join(p for p in parts if p)


def split_ids(ids, test_share=0.2, seed=42):
    """Random train/test split of movie ids (sorted first, so the split does not depend on input order)."""
    ids = np.sort(np.asarray(ids))
    rng = np.random.default_rng(seed)
    test_mask = rng.random(len(ids)) < test_share
    return ids[~test_mask], ids[test_mask]


def mean_spearman(y_true, y_pred):
    """Average over attributes of the rank correlation between true and predicted scores."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rhos = [spearmanr(y_true[:, j], y_pred[:, j])[0] for j in range(y_true.shape[1])]
    return float(np.nanmean(rhos))


def fit_tfidf_ridge(texts, targets, param_grid, cv=3, seed=42):
    """TF-IDF + ridge, with the settings in param_grid chosen by cross-validation (score = mean Spearman).

    Returns the fitted GridSearchCV; .best_estimator_ is the pipeline refitted on all training texts.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True)),
                     ("ridge", Ridge())])
    search = GridSearchCV(pipe, param_grid, scoring=make_scorer(mean_spearman),
                          cv=KFold(cv, shuffle=True, random_state=seed), n_jobs=-1, refit=True)
    search.fit(list(texts), np.asarray(targets))
    return search


def fit_tfidf_logistic(texts, labels, param_grid, cv=3, seed=42):
    """TF-IDF + logistic regression for a category (e.g. ending type), settings chosen by cross-validated accuracy.

    Returns the fitted GridSearchCV; .best_estimator_.predict_proba gives each category's probability.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True)),
                     ("logreg", LogisticRegression(max_iter=5000))])
    search = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                          n_jobs=-1, refit=True)
    search.fit(list(texts), np.asarray(labels))
    return search


def predict_frame(model, texts, index, columns):
    """Predictions as a DataFrame (movies x attributes)."""
    return pd.DataFrame(model.predict(list(texts)), index=index, columns=columns)


def top_words(model, attribute_index, n=10):
    """The words with the largest positive and negative weights for one attribute (what the model learned)."""
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    w = model.named_steps["ridge"].coef_[attribute_index]
    order = np.argsort(w)
    return list(vocab[order[::-1][:n]]), list(vocab[order[:n]])
