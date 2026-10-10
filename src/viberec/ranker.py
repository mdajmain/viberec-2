"""Step 10: the ranker. Stage 1 (candidates) takes each user's top-N from the ALS + vibe blend; stage 2 reorders
them with all signals, learned from data (weighted score vs Random Forest vs LightGBM).

Every feature is computed from what was known at the cutoff only.
"""
import numpy as np
import pandas as pd


def recent_counts(ratings, cutoff_ts, window_seconds):
    """Ratings per movie in [cutoff - window, cutoff): how much the movie was being watched just before the cutoff
    (present-day popularity, as opposed to all-time counts)."""
    r = ratings[(ratings["timestamp"] >= cutoff_ts - window_seconds) & (ratings["timestamp"] < cutoff_ts)]
    return r.groupby("movieId").size()


def movie_table(known, recent, years):
    """Per-movie features at a cutoff. known: movies_known_at() output (rating_count, shrunk, tier)."""
    t = known.set_index("movieId")[["rating_count", "shrunk", "tier"]].copy()
    t["log_count"] = np.log1p(t["rating_count"])
    t["log_recent"] = np.log1p(recent.reindex(t.index).fillna(0))
    t["year"] = years.reindex(t.index)
    t["recent_pct"] = t["log_recent"].rank(pct=True)                      # 0-1, used for the familiarity metric
    return t


FEATURES = ["als_z", "vibe_z", "blend_rank", "shrunk", "tier", "log_count", "log_recent", "year_gap"]


def user_rows(user, user_ratings, user_mean, candidate_ids, als_scores, vibe_scores, blend_scores, movies, m_index):
    """Feature rows for one user's candidates. *_scores: arrays over all CF movie columns (m_index)."""
    cols = m_index.loc[candidate_ids].to_numpy()

    def within_user_z(s):
        s = np.asarray(s, float)[cols]
        finite = np.isfinite(s)
        if not finite.any():
            return np.zeros(len(s))
        return np.where(finite, (s - s[finite].mean()) / (s[finite].std() or 1.0), 0.0)

    liked = user_ratings.loc[user_ratings["rating"] > user_mean, "movieId"]
    liked_years = movies["year"].reindex(liked).dropna()
    centre = liked_years.median() if len(liked_years) else movies["year"].median()
    m = movies.reindex(candidate_ids)
    blend = np.asarray(blend_scores, float)[cols]
    return pd.DataFrame({
        "userId": user, "movieId": candidate_ids,
        "als_z": within_user_z(als_scores), "vibe_z": within_user_z(vibe_scores),
        "blend_rank": np.argsort(np.argsort(-blend)) + 1,                     # 1 = top of the current recommender
        "shrunk": m["shrunk"].to_numpy(), "tier": m["tier"].to_numpy(), "log_count": m["log_count"].to_numpy(),
        "log_recent": m["log_recent"].to_numpy(), "year_gap": np.abs(m["year"].to_numpy() - centre),
    })


def top_k_by(rows, score, k=50):
    """Per user, movieIds ordered by score (descending)."""
    d = rows.assign(_s=score).sort_values(["userId", "_s"], ascending=[True, False])
    return d.groupby("userId")["movieId"].apply(lambda s: list(s[:k])).to_dict()


def familiarity(recs, recent_pct, k=10):
    """Mean present-day popularity percentile (0-1) of the top-k: how well-known today the picks are."""
    vals = [recent_pct.get(m) for m in recs[:k]]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan
