"""The testing zone: one fixed benchmark that every version of the recommender is scored on (PROJECT.md §12).

Time split: a user's ratings before the cutoff are what the recommender may see; ratings after it are the
future. A "hit" is a movie the user rated after the cutoff ABOVE their own pre-cutoff average (relative to
each user, so harsh and generous raters are treated alike). Scores are appended to a scoreboard file, so
every step's gain or cost stays visible.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- benchmark
def make_benchmark(ratings, cutoff_ts, n_users=5000, seed=42):
    """Sample test users and split their ratings at cutoff_ts.

    A test user has at least one rating before the cutoff (something to recommend from) and at least one hit
    after it (something to find). Returns (train, hits, user_mean): train = the users' pre-cutoff ratings;
    hits = {userId: set of movieIds rated after the cutoff above the user's pre-cutoff mean}.
    """
    before = ratings[ratings["timestamp"] < cutoff_ts]
    after = ratings[ratings["timestamp"] >= cutoff_ts]
    user_mean = before.groupby("userId")["rating"].mean()
    after = after[after["userId"].isin(user_mean.index)]
    liked_after = after[after["rating"] > after["userId"].map(user_mean)]
    eligible = np.sort(liked_after["userId"].unique())
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))
    train = before[before["userId"].isin(users)]
    hits = liked_after[liked_after["userId"].isin(users)].groupby("userId")["movieId"].apply(set).to_dict()
    return train, hits, user_mean.loc[users], len(eligible)


# ---------------------------------------------------------------- metrics
def recall_at_k(recs, hits, k):
    """Share of the user's future hits that appear in the top k."""
    return len(set(recs[:k]) & hits) / len(hits) if hits else np.nan


def ndcg_at_k(recs, hits, k):
    """Like recall, but a hit near the top counts more (binary relevance, log2 position discount)."""
    gains = [1.0 / np.log2(i + 2) for i, m in enumerate(recs[:k]) if m in hits]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(hits))))
    return sum(gains) / ideal if ideal else np.nan


def coverage(all_recs, pool_size, k):
    """Share of the candidate pool that appears in anyone's top k."""
    shown = set(m for recs in all_recs for m in recs[:k])
    return len(shown) / pool_size if pool_size else np.nan


def diversity(recs, vectors, k):
    """Mean pairwise distance (1 - cosine) between the top-k movies' vibe vectors (movies without one skipped)."""
    v = np.array([vectors[m] for m in recs[:k] if m in vectors])
    if len(v) < 2:
        return np.nan
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    return float(1 - sim[np.triu_indices(len(v), 1)].mean())


def novelty(recs, popularity_share, k):
    """Mean self-information -log2(share of all ratings) of the top-k movies: higher = less obvious picks."""
    shares = [popularity_share.get(m) for m in recs[:k]]
    shares = [s for s in shares if s]
    return float(np.mean([-np.log2(s) for s in shares])) if shares else np.nan


# ---------------------------------------------------------------- running and recording
def evaluate(recommend, train, hits, ks=(10, 50), vibe_vectors=None, popularity_share=None, pool_size=None,
             return_per_user=False):
    """Run recommend(user_id, user_train_ratings) -> (ranked movieIds, candidates_scanned) for every test user.

    Returns a dict of mean metrics plus timing. Movies the user already rated before the cutoff must be
    excluded by the recommender itself (it sees them in user_train_ratings).
    """
    by_user = dict(tuple(train.groupby("userId")))
    k_max = max(ks)
    rows, all_recs, elapsed, scanned = [], [], 0.0, []
    for user, user_hits in hits.items():
        start = time.perf_counter()
        recs, n_scanned = recommend(user, by_user[user])
        elapsed += time.perf_counter() - start
        recs = list(recs)[:k_max]
        all_recs.append(recs)
        scanned.append(n_scanned)
        row = {f"recall@{k}": recall_at_k(recs, user_hits, k) for k in ks}
        row["ndcg@10"] = ndcg_at_k(recs, user_hits, 10)
        if vibe_vectors is not None:
            row["diversity@10"] = diversity(recs, vibe_vectors, 10)
        if popularity_share is not None:
            row["novelty@10"] = novelty(recs, popularity_share, 10)
        row["userId"] = user
        rows.append(row)
    per_user = pd.DataFrame(rows).set_index("userId")
    result = per_user.mean().to_dict()
    result["coverage@10"] = coverage(all_recs, pool_size, 10) if pool_size else np.nan
    result["ms_per_user"] = 1000 * elapsed / len(hits)
    result["candidates_scanned"] = float(np.mean(scanned))
    result["users"] = len(hits)
    return (result, per_user) if return_per_user else result


def paired_gain(per_user_a, per_user_b, metric, n_boot=2000, seed=0):
    """Mean of metric for A minus B over the same users, with a paired bootstrap 95% interval
    (resampling users). Returns (difference, low, high, relative change vs B)."""
    both = pd.concat([per_user_a[metric], per_user_b[metric]], axis=1, keys=["a", "b"]).dropna()
    d = (both["a"] - both["b"]).to_numpy()
    picks = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[picks].mean(axis=1), [2.5, 97.5])
    return d.mean(), lo, hi, d.mean() / both["b"].mean() if both["b"].mean() else np.nan


def append_scoreboard(path, step, method, result, settings=None, note=""):
    """Add one row to the scoreboard CSV (append-only: earlier rows are never changed). A run whose step,
    method and settings are already on the scoreboard is not added again, so rerunning a notebook does not
    duplicate rows; change the settings (or the method name) to record a new variant."""
    path = Path(path)
    settings_json = json.dumps(settings or {}, default=str, sort_keys=True)
    row = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "step": step, "method": method,
           **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in result.items()},
           "settings": settings_json, "note": note}
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        same = (old["step"] == step) & (old["method"] == method) & (old["settings"] == settings_json)
        if same.any():
            return old
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def movies_known_at(early_rows, movies, k_shrink, cutoff_year):
    """Per movie, what was known at a cutoff (from pre-cutoff ratings only): count, mean, fair (shrunk) score and
    the Step 4 tier. early_rows: one quantile of q43_early_counts (movieId, n_early, sum_early);
    movies: movieId, genres ("|"), year."""
    from . import credibility as cr
    from . import tiers as tr
    known = early_rows[early_rows["n_early"] > 0].merge(movies[["movieId", "genres", "year"]], on="movieId")
    known = known.assign(rating_count=known["n_early"], mean_rating=known["sum_early"] / known["n_early"])
    prior = known["mean_rating"].mean()
    known["shrunk"] = cr.shrunk_mean(known["rating_count"], known["mean_rating"], prior, k_shrink)
    rule = tr.approve_rule(known, score_col="mean_rating", upper_score_col="shrunk", data_year=cutoff_year)
    known["tier"] = tr.assign_tier(rule.set_index("movieId").loc[known["movieId"]].reset_index())
    return known
