"""Taste profiles and the "For you" feed (Steps 8 and 13), served from the database.

The recommendation is exactly the Step 8 method that was tested (notebook 10, `recommend_new_user`):
- fold the person into ALS from the films they liked (cf.fold_in_user, the same least-squares step as the library);
- vibe taste = the average standardised vibe vector of the films they liked; vibe match = cosine to each film;
- both scores standardised over the approved pool, blended 0.2 ALS + 0.8 vibe (Step 7's tuned weight);
- films already answered are never recommended; with no likes yet, the most-rated films come first.
Notebook 17 checks this class against a copy of the notebook function.
"""
import numpy as np

from .cf import fold_in_user
from .explain import LIGHT_WORDS, PACE_WORDS, SPOILER_VIBES, VIBE_WORDS

QUIZ_TARGET_LIKES, QUIZ_CAP = 20, 100            # Step 8 v2 (tested): ask until 20 likes, at most 100 questions


def standardised(s):
    """(s - mean) / sd over the finite values; non-finite values count as 0 (neutral), as in notebook 10."""
    s = np.asarray(s, float)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    return np.where(finite, (s - s[finite].mean()) / (s[finite].std() or 1.0), 0.0)


class TasteModel:
    """Everything the feed needs, in memory (about 25 MB for the approved films)."""

    def __init__(self, ids, factors, z, tier, rating_count, yty, alpha, reg, w_als, attrs, year=None, genres=None,
                 light_heavy=None, pacing=None):
        self.ids = np.asarray(ids)                                   # every film with ALS factors and vibe scores
        self.pos = {int(m): i for i, m in enumerate(self.ids)}
        self.factors = np.asarray(factors, np.float32)                # ALS item factors, rows aligned with ids
        self.z = np.asarray(z, float)                                 # standardised vibe vectors
        self.unit = self.z / np.linalg.norm(self.z, axis=1, keepdims=True)
        self.yty = np.asarray(yty, float)                             # Y'Y over ALL films in the ALS model
        self.alpha, self.reg, self.w_als = float(alpha), float(reg), float(w_als)
        self.attrs = list(attrs)
        pool = np.asarray(tier) == 3                                  # recommendations come from approved films
        self.pool_rows = np.where(pool)[0]
        self.popular_rows = self.pool_rows[np.argsort(-np.asarray(rating_count)[self.pool_rows], kind="stable")]
        self.year, self.genres = year, genres
        self.light_heavy, self.pacing = light_heavy, pacing

    # ------------------------------------------------------------------ building blocks
    def word(self, i):
        """How vibe attribute i is named to people."""
        return VIBE_WORDS.get(self.attrs[i], self.attrs[i].replace("_", " "))

    def rows_of(self, movie_ids):
        return [self.pos[int(m)] for m in movie_ids if int(m) in self.pos]

    def taste_vector(self, likes):
        """The person's 128 ALS taste numbers (zeros before any like)."""
        return fold_in_user(self.factors, self.yty, self.rows_of(likes), self.alpha, self.reg)

    def vibe_taste(self, likes):
        rows = self.rows_of(likes)
        return self.z[rows].mean(axis=0) if rows else None

    # ------------------------------------------------------------------ the feed
    def scores(self, likes):
        """(blended score, ALS part, vibe part) over the pool; parts are the weighted standardised scores."""
        rows = self.rows_of(likes)
        u = fold_in_user(self.factors, self.yty, rows, self.alpha, self.reg)
        als = self.factors[self.pool_rows] @ u
        t = self.vibe_taste(likes)
        vib = self.unit[self.pool_rows] @ (t / (np.linalg.norm(t) or 1)) if t is not None else np.full(len(self.pool_rows), np.nan)
        als_part, vibe_part = self.w_als * standardised(als), (1 - self.w_als) * standardised(vib)
        return als_part + vibe_part, als_part, vibe_part

    def recommend(self, likes, exclude=(), k=10):
        """Film ids for the "For you" feed, best first (notebook 10's recommend_new_user)."""
        asked = {int(m) for m in exclude} | {int(m) for m in likes}
        if not self.rows_of(likes):
            return [int(self.ids[r]) for r in self.popular_rows if int(self.ids[r]) not in asked][:k]
        score = self.scores(likes)[0]
        pool_ids = self.ids[self.pool_rows]
        score[np.isin(pool_ids, list(asked))] = -np.inf
        top = np.argpartition(-score, k)[:k]
        return [int(m) for m in pool_ids[top[np.argsort(-score[top])]]]

    def parts_for(self, likes, movie_ids):
        """{film: (ALS part, vibe part)} for explaining feed picks."""
        _, a, v = self.scores(likes)
        where = {int(m): i for i, m in enumerate(self.ids[self.pool_rows])}
        return {int(m): (float(a[where[int(m)]]), float(v[where[int(m)]])) for m in movie_ids if int(m) in where}

    def shared_feel(self, likes, movie_id, k=3):
        """The vibes a film shares most with the person's vibe taste (cosine split per vibe), spoilers left out."""
        t = self.vibe_taste(likes)
        if t is None or int(movie_id) not in self.pos:
            return []
        parts = self.unit[self.pos[int(movie_id)]] * (t / (np.linalg.norm(t) or 1))
        order = [i for i in np.argsort(-parts) if parts[i] > 0 and self.attrs[i] not in SPOILER_VIBES]
        return [self.word(i) for i in order[:k]]

    # ------------------------------------------------------------------ the readable profile
    def summary(self, likes, n_dislikes=0, n_unseen=0):
        """What the "Your taste" page shows."""
        rows = self.rows_of(likes)
        out = {"likes": len(likes), "dislikes": int(n_dislikes), "unseen": int(n_unseen), "top_vibes": [], "light_heavy": None,
               "pacing": None, "era": None, "top_genres": []}
        if not rows:
            return out
        t = self.z[rows].mean(axis=0)
        order = [i for i in np.argsort(-t) if t[i] > 0 and self.attrs[i] not in SPOILER_VIBES]
        out["top_vibes"] = [self.word(i) for i in order[:4]]
        for key, words, col in (("light_heavy", LIGHT_WORDS, self.light_heavy), ("pacing", PACE_WORDS, self.pacing)):
            if col is not None:
                vals = np.asarray([col[r] for r in rows], float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    out[key] = words[int(round(vals.mean()))] or "in between"
        if self.year is not None:
            ys = np.asarray([self.year[r] for r in rows], float)
            ys = ys[np.isfinite(ys)]
            if len(ys):
                lo, hi = np.percentile(ys, [25, 75])
                out["era"] = f"{int(lo)}–{int(hi)}" if int(lo) != int(hi) else str(int(lo))
        if self.genres is not None:
            counts = {}
            for r in rows:
                gs = self.genres[r]
                for g in (list(gs) if gs is not None else []):
                    counts[g] = counts.get(g, 0) + 1
            out["top_genres"] = [g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        return out


def next_question(questions, answers, target_likes=QUIZ_TARGET_LIKES, cap=QUIZ_CAP):
    """The quiz (Step 8 v2): the next unanswered film in the fixed order, or None when done
    (target likes reached, cap reached, or the list ran out). answers: {movie_id: verdict}."""
    likes = sum(v == "like" for v in answers.values())
    if likes >= target_likes or len(answers) >= cap:
        return None
    for m in questions:
        if int(m) not in answers:
            return int(m)
    return None
