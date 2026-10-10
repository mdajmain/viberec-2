"""Step 7: collaborative filtering. Learn taste from patterns across all users' ratings.

All models are trained only on ratings before the benchmark cutoff. The rating table is a sparse
users x movies matrix in one of three forms (which works best is tested, not assumed):
  "watched": 1 for every rating;  "liked": 1 where the rating is above the user's own mean;  "rating": the stars.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_matrix(ratings, kind="watched", user_ids=None, movie_ids=None):
    """Sparse users x movies matrix plus the id -> row/column maps."""
    user_ids = np.sort(ratings["userId"].unique()) if user_ids is None else np.asarray(user_ids)
    movie_ids = np.sort(ratings["movieId"].unique()) if movie_ids is None else np.asarray(movie_ids)
    u_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    m_index = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    r = ratings[ratings["movieId"].isin(m_index.index) & ratings["userId"].isin(u_index.index)]
    if kind == "watched":
        values = np.ones(len(r), dtype=np.float32)
    elif kind == "liked":
        mean = r.groupby("userId")["rating"].transform("mean")
        keep = (r["rating"] > mean).to_numpy()
        r, values = r[keep], np.ones(int(keep.sum()), dtype=np.float32)
    elif kind == "rating":
        values = r["rating"].to_numpy(np.float32)
    else:
        raise ValueError(kind)
    m = sp.csr_matrix((values, (u_index.loc[r["userId"]].to_numpy(), m_index.loc[r["movieId"]].to_numpy())),
                      shape=(len(user_ids), len(movie_ids)))
    return m, u_index, m_index


def pure_svd(matrix, k, seed=42):
    """PureSVD: the top-k right singular vectors V (movies x k). A user's scores are (user_row @ V) @ V.T."""
    from scipy.sparse.linalg import svds
    _, _, vt = svds(matrix.astype(np.float64), k=k, random_state=seed)
    return np.ascontiguousarray(vt.T[:, ::-1]).astype(np.float32)


def make_scorer_recommender(user_scores, pool_ids, m_index, k=50):
    """Turn user_scores(user, user_ratings) -> scores for every movie column into a recommender restricted to the
    pool, skipping movies the user already rated. Returns recommend(user, user_ratings) -> (ids, scanned)."""
    pool_ids = np.asarray([m for m in pool_ids if m in m_index.index])
    cols = m_index.loc[pool_ids].to_numpy()

    def recommend(user, user_ratings):
        s = np.asarray(user_scores(user, user_ratings), dtype=np.float64)[cols]
        s[np.isin(pool_ids, user_ratings["movieId"].to_numpy())] = -np.inf
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return list(pool_ids[top]), len(pool_ids)
    return recommend


def svd_user_scores(matrix, u_index, item_factors):
    """Scores for PureSVD: fold the user's row into the factor space and back."""
    def scores(user, _ratings):
        row = matrix[u_index[user]]
        return np.asarray((row @ item_factors) @ item_factors.T).ravel()
    return scores


def factor_user_scores(user_factors, item_factors, u_index):
    """Scores for models with learned user and item factors (ALS): dot products."""
    def scores(user, _ratings):
        return item_factors @ user_factors[u_index[user]]
    return scores


def blend_user_scores(score_fns, weights):
    """Weighted sum of several scorers, each first standardised within the user (mean 0, sd 1 over all movies),
    so scores on different scales can be combined. A movie a scorer cannot score (NaN, e.g. no vibe scores)
    counts as neutral (0) in that scorer's part, so it is neither favoured nor pushed to the bottom."""
    def scores(user, ratings):
        total = 0.0
        for fn, w in zip(score_fns, weights):
            s = np.asarray(fn(user, ratings), dtype=np.float64)
            finite = np.isfinite(s)
            if not finite.any():                                  # e.g. no vibe taste for this user: skip this part
                continue
            mu, sd = s[finite].mean(), s[finite].std() or 1.0
            total = total + w * np.where(finite, (s - mu) / sd, 0.0)
        return total
    return scores


def vibe_user_scores(vibes, m_index, user_mean):
    """vibe_match as a scorer over the CF movie columns: cosine between the user's vibe taste (mean standardised vibe
    of movies rated above their own mean) and each movie; NaN for movies without vibe scores or users without taste."""
    from .recommenders import standardise
    z = standardise(vibes)
    unit = z.div(np.linalg.norm(z.to_numpy(), axis=1), axis=0)
    full = unit.reindex(m_index.index).to_numpy()                       # rows aligned with CF columns, NaN if missing

    def scores(user, user_ratings):
        liked = user_ratings.loc[user_ratings["rating"] > user_mean[user], "movieId"]
        liked = liked[liked.isin(z.index)]
        if liked.empty:
            return np.full(len(m_index), np.nan)
        taste = z.loc[liked].mean().to_numpy()
        return full @ (taste / (np.linalg.norm(taste) or 1))
    return scores


# ---------------------------------------------------------------- new users (Step 8: onboarding)
def fold_in_user(item_factors, yty, liked_cols, alpha, reg):
    """Taste factors of a user the ALS model has never seen, from the movies they liked: the same least-squares
    step the `implicit` library's ALS uses for every user. Its confidence for a liked movie is alpha (alpha x the
    1 in the liked table; checked against implicit.cpu.als.user_linear_equation), so
        u = (Y'Y + (alpha - 1) * Y_L'Y_L + reg * I)^-1  alpha * Y_L' 1
    yty = item_factors.T @ item_factors (precomputed once)."""
    k = item_factors.shape[1]
    if len(liked_cols) == 0:
        return np.zeros(k, dtype=np.float32)
    yl = item_factors[np.asarray(liked_cols)]
    a = yty + (alpha - 1) * yl.T @ yl + reg * np.eye(k)
    return np.linalg.solve(a, alpha * yl.sum(axis=0)).astype(np.float32)


def question_weights(seen_share, like_share):
    """How much one answer about each movie is expected to tell us: P(user has seen it) times the uncertainty of
    like vs dislike, p(1 - p). Widely seen and divisive movies score high."""
    p = np.clip(np.asarray(like_share, float), 0, 1)
    return np.asarray(seen_share, float) * p * (1 - p)


def informative_questions(item_factors, weights, n, reg, groups=None, quota=None, start=None):
    """Fixed questionnaire by greedy D-optimal design: repeatedly add the movie that most increases
    log det(reg*I + sum of w_i y_i y_i') over the chosen movies (most information about an average user's factors).
    Popularity, divisiveness (through the weights) and variety (through the determinant) are combined with no
    hand-set mixing weights. Returns row indices into item_factors.

    groups / quota (optional): a group label per movie and the maximum number of picks per group (a product rule,
    e.g. at most 5 films from the 1990s); movies whose group is full or missing from quota are skipped.
    start (optional): row indices already asked, kept first and in order; the list continues from them, so an
    extended questionnaire keeps an existing one as its opening. They count towards n and the quota."""
    y = item_factors * np.sqrt(np.asarray(weights, float))[:, None]
    cov = np.eye(y.shape[1]) / reg                       # inverse of reg*I
    quad = (y * y).sum(axis=1) / reg                     # y_i' cov y_i: information the next answer adds
    chosen = []
    groups = None if groups is None else np.asarray(groups)
    used = {}
    for best in (start or []):                           # replay the opening questions into the bookkeeping
        chosen.append(int(best))
        if groups is not None:
            used[groups[best]] = used.get(groups[best], 0) + 1
        v = cov @ y[best]
        denom = 1 + y[best] @ v
        quad -= (y @ v) ** 2 / denom
        cov -= np.outer(v, v) / denom
    for _ in range(n - len(chosen)):
        gain = quad.copy()
        gain[chosen] = -np.inf
        if groups is not None:
            full = [g for g in np.unique(groups) if used.get(g, 0) >= quota.get(g, 0)]
            gain[np.isin(groups, full)] = -np.inf
            if not np.isfinite(gain).any():
                break
        best = int(np.argmax(gain))
        chosen.append(best)
        if groups is not None:
            used[groups[best]] = used.get(groups[best], 0) + 1
        v = cov @ y[best]
        denom = 1 + y[best] @ v
        quad -= (y @ v) ** 2 / denom                     # same rank-one update applied to every candidate at once
        cov -= np.outer(v, v) / denom                    # Sherman-Morrison
    return chosen
