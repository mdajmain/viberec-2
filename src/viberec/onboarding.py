"""Step 8: onboarding a new user with a short like / dislike / haven't-seen questionnaire (PROJECT.md §6).

Simulation: a test user's real pre-cutoff ratings answer the questions (like = above their own mean, dislike =
at or below it, haven't seen = never rated). After n answers the user is folded into ALS from the liked movies
and recommended to, exactly as a brand-new user would be.

Question strategies (all draw from the same candidate set):
  random      - random candidates (floor)
  popular     - most-rated first
  informative - one fixed list from greedy D-optimal design weighted by P(seen) x like-uncertainty (cf.py)
  adaptive    - per user: the next question maximises expected information given the answers so far, using
                P(seen | movies this user has already seen) from co-watching patterns
"""
import numpy as np


class Questionnaire:
    """Candidate movies with what the strategies need: ALS factors, P(seen), like-uncertainty, co-watch rates."""

    def __init__(self, movie_ids, item_factors, seen_share, like_share, cowatch, reg, groups=None, quota=None):
        self.ids = np.asarray(movie_ids)
        self.y = np.asarray(item_factors, float)
        self.seen = np.asarray(seen_share, float)
        p = np.clip(np.asarray(like_share, float), 0, 1)
        self.like_unc = p * (1 - p)
        self.cowatch = np.asarray(cowatch, float)          # cowatch[i, j] = P(seen i | seen j)
        self.reg = reg
        self.groups, self.quota = groups, quota            # optional product rule for the fixed list

    def fixed_order(self, strategy, n, seed=0):
        if strategy == "random":
            return list(np.random.default_rng(seed).permutation(len(self.ids))[:n])
        if strategy == "popular":
            return list(np.argsort(-self.seen)[:n])
        if strategy == "informative":
            from .cf import informative_questions
            return informative_questions(self.y, self.seen * self.like_unc, n, self.reg, self.groups, self.quota)
        raise ValueError(strategy)

    def run(self, strategy, answer, n, seed=0):
        """Ask n questions; answer(movieId) -> "like" | "dislike" | None (haven't seen). Returns [(movieId, answer)]."""
        if strategy != "adaptive":
            return [(self.ids[i], answer(self.ids[i])) for i in self.fixed_order(strategy, n, seed)]
        cov = np.eye(self.y.shape[1]) / self.reg
        quad = (self.y * self.y).sum(axis=1) / self.reg          # y_i' cov y_i, kept up to date below
        cow_sum = np.zeros(len(self.ids))
        asked, seen_idx, out = [], [], []
        for _ in range(n):
            p_seen = cow_sum / len(seen_idx) if seen_idx else self.seen
            gain = p_seen * self.like_unc * quad
            gain[asked] = -np.inf
            i = int(np.argmax(gain))
            a = answer(self.ids[i])
            asked.append(i)
            out.append((self.ids[i], a))
            if a is not None:                                # seen: its answer informs the user's factors
                seen_idx.append(i)
                cow_sum += self.cowatch[:, i]
                yw = self.y[i] * np.sqrt(self.like_unc[i])
                v = cov @ yw
                denom = 1 + yw @ v
                quad -= (self.y @ v) ** 2 / denom
                cov -= np.outer(v, v) / denom
        return out


def answer_from_history(user_ratings, user_mean):
    """Simulated answers from a user's real ratings."""
    r = dict(zip(user_ratings["movieId"], user_ratings["rating"]))

    def answer(movie_id):
        if movie_id not in r:
            return None
        return "like" if r[movie_id] > user_mean else "dislike"
    return answer
