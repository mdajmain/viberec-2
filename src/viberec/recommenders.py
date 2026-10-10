"""Recommenders for the testing zone. Each make_* function returns recommend(user_id, user_ratings) ->
(ranked movieIds, number of candidates scanned), the interface evaluation.evaluate() expects.

Movies the user already rated are never recommended.
"""
import numpy as np
import pandas as pd


def make_ranked_list(pool_ids, score, k=50):
    """Non-personal recommender: one fixed ranking of the pool (e.g. by popularity or by fair score);
    each user gets the top k they have not rated yet."""
    ranked = pd.Series(score).reindex(pool_ids).dropna().sort_values(ascending=False).index.to_numpy()

    def recommend(user, user_ratings):
        seen = set(user_ratings["movieId"])
        out = []
        for m in ranked:
            if m not in seen:
                out.append(m)
                if len(out) == k:
                    break
        return out, len(ranked)
    return recommend


def standardise(vibes):
    """Each vibe column on a common scale: (value - mean) / std across movies."""
    return (vibes - vibes.mean()) / vibes.std(ddof=0)


def make_vibe_match(vibes, pool_ids, user_mean, k=50, standardize=True):
    """Personal recommender from vibe alone: the user's taste = mean standardised vibe vector of the movies they
    rated above their own average; candidates are ranked by cosine similarity to that taste vector.

    vibes: movies x vibe attributes (index movieId). Users with no liked movie that has vibe scores get no
    recommendations (an empty list, scored as a miss).
    """
    z = standardise(vibes) if standardize else vibes          # PCA scores are already on a meaningful scale
    pool = z.reindex(pd.Index(pool_ids)).dropna()
    pool_ids_arr = pool.index.to_numpy()
    pool_unit = pool.to_numpy() / np.linalg.norm(pool.to_numpy(), axis=1, keepdims=True)

    def recommend(user, user_ratings):
        liked = user_ratings.loc[user_ratings["rating"] > user_mean[user], "movieId"]
        liked = liked[liked.isin(z.index)]
        if liked.empty:
            return [], len(pool_ids_arr)
        taste = z.loc[liked].mean().to_numpy()
        taste = taste / (np.linalg.norm(taste) or 1)
        sim = pool_unit @ taste
        seen = np.isin(pool_ids_arr, user_ratings["movieId"].to_numpy())
        sim[seen] = -np.inf
        top = np.argpartition(-sim, min(k, len(sim) - 1))[:k]
        top = top[np.argsort(-sim[top])]
        return list(pool_ids_arr[top]), len(pool_ids_arr)
    return recommend


def make_cluster_vibe_match(vibes, pool_ids, user_mean, cluster_of, centroids, n_probe, k=50):
    """vibe_match that only searches the n_probe clusters whose centres are closest to the user's taste
    (the way an IVF vector index works). cluster_of: Series movieId -> cluster; centroids: unit vectors in the
    same standardised vibe space. Returns candidates scanned = movies in the probed clusters."""
    z = standardise(vibes)
    pool = z.reindex(pd.Index(pool_ids)).dropna()
    ids = pool.index.to_numpy()
    unit = pool.to_numpy() / np.linalg.norm(pool.to_numpy(), axis=1, keepdims=True)
    clusters = cluster_of.reindex(ids).to_numpy()
    members = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}

    def recommend(user, user_ratings):
        liked = user_ratings.loc[user_ratings["rating"] > user_mean[user], "movieId"]
        liked = liked[liked.isin(z.index)]
        if liked.empty:
            return [], 0
        taste = z.loc[liked].mean().to_numpy()
        taste = taste / (np.linalg.norm(taste) or 1)
        probe = np.argsort(-(centroids @ taste))[:n_probe]
        rows = np.concatenate([members.get(c, np.array([], int)) for c in probe])
        sim = unit[rows] @ taste
        seen = np.isin(ids[rows], user_ratings["movieId"].to_numpy())
        sim[seen] = -np.inf
        top = np.argsort(-sim)[:k]
        return list(ids[rows][top]), len(rows)
    return recommend
