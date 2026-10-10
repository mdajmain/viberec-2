"""The approve rule from notebooks/04_eda.ipynb, as a function so it can be rerun with other settings (Step 4).

Rating test: score between mu - lower_sd * sigma and mu + upper_sd * sigma of the movie's genre
(most forgiving range among its genres). Count test: per-genre Q1 / Q3 of rating_count (lowest
among its genres) gives low / mid / high. Approve: recent, or high count, or mid count with a normal
rating, or mid count with a too-high rating and at least the genre median count.
"""
import numpy as np
import pandas as pd

EXCLUDE = {"(no genres listed)", "IMAX"}


def _genre_rows(movies, top_genres):
    """One row per (movie, genre group); genres outside top_genres become Other."""
    rows = (movies[["movieId", "genres"]].assign(genre=movies["genres"].str.split("|"))
            .explode("genre").loc[lambda d: ~d["genre"].isin(EXCLUDE)])
    rows["genre_group"] = rows["genre"].where(rows["genre"].isin(top_genres), "Other")
    return rows.drop_duplicates(["movieId", "genre_group"])


def approve_rule(movies, score_col="mean_rating", lower_sd=1.0, upper_sd=2.0,
                 margin_years=8, data_year=None, min_votes=10, top_n=10, upper_score_col=None):
    """Return one row per movie: movieId, rating_side, count_level, recent, approve, reason.

    movies needs movieId, genres ("|"-separated), year, rating_count and score_col.
    score_col decides "too low"; upper_score_col (default: score_col) decides "too high",
    each against its own genre range.
    data_year defaults to the latest release year in movies (as in 04_eda).
    """
    movies = movies.reset_index(drop=True)
    upper_col = upper_score_col or score_col
    score_cols = list(dict.fromkeys([score_col, upper_col]))
    if data_year is None:
        data_year = int(movies["year"].max())

    # genre range of each score, measured on movies with at least min_votes ratings
    base = movies[movies["rating_count"] >= min_votes]
    base_rows = (base[["movieId", "genres"]].assign(genre=base["genres"].str.split("|"))
                 .explode("genre").loc[lambda d: ~d["genre"].isin(EXCLUDE)])
    top_genres = base_rows["genre"].value_counts().head(top_n).index
    base_rows = _genre_rows(base, top_genres).join(base.set_index("movieId")[score_cols], on="movieId")
    low_stats = base_rows.groupby("genre_group")[score_col].agg(["mean", "std"])
    up_stats = base_rows.groupby("genre_group")[upper_col].agg(["mean", "std"])
    stats = pd.DataFrame({"lower": low_stats["mean"] - lower_sd * low_stats["std"],
                          "upper": up_stats["mean"] + upper_sd * up_stats["std"] if upper_sd is not None else np.inf},
                         index=low_stats.index)

    # count quartiles per genre, on all movies
    rows = _genre_rows(movies, top_genres).join(movies.set_index("movieId")["rating_count"], on="movieId")
    quart = rows.groupby("genre_group")["rating_count"].quantile([0.25, 0.5, 0.75]).unstack()
    quart.columns = ["q1", "median", "q3"]

    per_movie = (rows.join(stats[["lower", "upper"]], on="genre_group").join(quart, on="genre_group")
                 .groupby("movieId").agg(lower=("lower", "min"), upper=("upper", "max"),
                                         q1=("q1", "min"), q3=("q3", "min"), median=("median", "min")))
    m = movies[["movieId", "year", "rating_count"] + score_cols].join(per_movie, on="movieId")

    side = np.select([m[score_col] < m["lower"], m[upper_col] > m["upper"]], ["too_low", "too_high"], "normal")
    level = np.select([m["rating_count"] < m["q1"], m["rating_count"] > m["q3"]], ["low", "high"], "mid")
    has_genre = m["lower"].notna() & m["q3"].notna()
    recent = m["year"].notna() & ((data_year - m["year"]) < margin_years)

    approve = (recent
               | (has_genre & (level == "high"))
               | (has_genre & (level == "mid") & (side == "normal"))
               | (has_genre & (level == "mid") & (side == "too_high") & (m["rating_count"] >= m["median"])))
    reason = np.select([approve, ~has_genre, level == "low", side == "too_low"],
                       ["approved", "no_genre", "low_count", "mid_too_low"], "mid_too_high_few")
    return pd.DataFrame({"movieId": m["movieId"], "rating_side": side, "count_level": level,
                         "recent": recent.to_numpy(), "approve": approve.to_numpy(), "reason": reason})


def assign_tier(rule_out):
    """Three tiers from approve_rule output (PROJECT.md 4.1/4.2).

    1 = excluded: rejected with a too-low rating (low rating, not a high count, not recent).
    2 = reduced weight: rejected for any other reason (few ratings or a too-high mean on few ratings).
    3 = approved.
    """
    too_low = (~rule_out["approve"]) & (rule_out["rating_side"] == "too_low")
    return np.where(rule_out["approve"], 3, np.where(too_low, 1, 2))
