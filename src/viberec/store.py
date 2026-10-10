"""Step 13: the catalog in PostgreSQL + pgvector (locally in Docker, on AWS in RDS).

Tables
- movies: one row per searchable film (tiers 2 + 3), with two vectors: `vibe` (the 41 standardised vibe scores) and
  `taste` (the 128 ALS factors), plus title, year, genres, popularity, premise, Light/Heavy, Pacing.
- bm25_idf, bm25_postings: the keyword index as an inverted index (word -> films and that word's BM25 weight in each
  film). It reproduces retrieval.BM25 exactly, with no extension, so it also runs on RDS.
- titles: names (with and without year) -> film, for resolving the sample movie Claude names.
- meta: the vibe attribute order and the vibe words the parser uses.
- feedback: the app's thumbs up / down / "not this vibe".
"""
import json
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value jsonb);
CREATE TABLE IF NOT EXISTS movies (
    movie_id integer PRIMARY KEY, title text NOT NULL, display_title text NOT NULL, year integer, genres text[],
    tier smallint, rating_count integer, recent_log real, premise text, light_heavy real, pacing real,
    vibe vector({n_vibe}), taste vector({n_taste}));
CREATE TABLE IF NOT EXISTS titles (norm_title text NOT NULL, year integer, movie_id integer NOT NULL REFERENCES movies);
CREATE INDEX IF NOT EXISTS titles_norm ON titles (norm_title);
CREATE TABLE IF NOT EXISTS bm25_idf (term text PRIMARY KEY, idf real NOT NULL);
CREATE TABLE IF NOT EXISTS bm25_postings (term text NOT NULL, movie_id integer NOT NULL, w real NOT NULL);
CREATE INDEX IF NOT EXISTS bm25_postings_term ON bm25_postings (term);
CREATE TABLE IF NOT EXISTS feedback (id bigserial PRIMARY KEY, created timestamptz DEFAULT now(), request text,
    movie_id integer, verdict text);
"""


def vec(x):
    """A numpy vector as pgvector text ('[1,2,3]'); None for missing or all-zero (no direction to compare)."""
    if x is None:
        return None
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or not np.any(x):
        return None
    return "[" + ",".join(f"{v:.6g}" for v in x) + "]"


def create_schema(conn, n_vibe, n_taste):
    with conn.cursor() as cur:
        cur.execute(SCHEMA.format(n_vibe=n_vibe, n_taste=n_taste))
    conn.commit()


def load_engine(conn, engine, three=None, premise=None):
    """Copy a search_engine.SearchEngine's catalog into the database (replacing what is there).
    three: vibe_three table (index movieId); premise: function movieId -> premise text."""
    from .explain import display_title
    from .labels import norm
    create_schema(conn, engine.z.shape[1], engine.als.shape[1])
    with conn.cursor() as cur:
        cur.execute("TRUNCATE titles, bm25_postings, bm25_idf, movies CASCADE")
        cur.execute("INSERT INTO meta VALUES ('attrs', %s), ('attr_tags', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (json.dumps(list(engine.attrs)), json.dumps(engine.attr_tags)))
        rows = []
        for i, m in enumerate(engine.ids):
            t = engine.tiers.loc[m]
            lh = three.at[m, "light_heavy"] if three is not None and m in three.index else None
            pc = three.at[m, "pacing"] if three is not None and m in three.index else None
            rows.append((int(m), t["title"], display_title(t["title"]),
                         int(engine.year[i]) if np.isfinite(engine.year[i]) else None,
                         [g for g in engine.genres[i] if g], int(t["tier"]), int(t["rating_count"]), float(engine.recent[i]),
                         premise(m) if premise else None, None if lh is None or lh != lh else float(lh),
                         None if pc is None or pc != pc else float(pc), vec(engine.z[i]), vec(engine.als[i])))
        with cur.copy("COPY movies FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
        with cur.copy("COPY titles FROM STDIN") as cp:
            for name, m in engine.title_index.items():
                cp.write_row((name, None, int(m)))
            for (name, year), m in engine.title_year_index.items():
                cp.write_row((name, int(year) if year else None, int(m)))
        vocab = engine.bm25.vec.get_feature_names_out()
        with cur.copy("COPY bm25_idf FROM STDIN") as cp:
            for term, idf in zip(vocab, engine.bm25.idf):
                cp.write_row((term, float(idf)))
        w = engine.bm25.weights.tocsc()
        with cur.copy("COPY bm25_postings FROM STDIN") as cp:
            for j, term in enumerate(vocab):
                lo, hi = w.indptr[j], w.indptr[j + 1]
                for row, val in zip(w.indices[lo:hi], w.data[lo:hi]):
                    cp.write_row((term, int(engine.ids[row]), float(val)))
    conn.commit()


# ---------------------------------------------------------------------- app v2: feed settings and posters
def set_meta(conn, **values):
    with conn.cursor() as cur:
        for k, v in values.items():
            cur.execute("INSERT INTO meta VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (k, json.dumps(v)))
    conn.commit()


def load_feed_meta(conn, item_factors_all, alpha, reg, w_als, quiz_ids):
    """What the "For you" feed needs besides each film's vectors: Y'Y over ALL films in the ALS model (for the fold-in),
    the ALS settings, the blend weight and the quiz order (Step 8 v2)."""
    y = np.asarray(item_factors_all, float)
    set_meta(conn, yty=(y.T @ y).tolist(), als={"alpha": float(alpha), "reg": float(reg)}, w_als=float(w_als),
             quiz=[int(m) for m in quiz_ids])


def load_posters(conn, poster_dirs):
    """poster_path per film from cached TMDB answers (<dir>/<tmdbId>.json). poster_dirs: one folder or several, the
    first that has a film's poster wins (English posters in poster_v2, then the language-default ones in poster_v1).
    Returns how many films have a poster."""
    from pathlib import Path
    conn.execute("ALTER TABLE movies ADD COLUMN IF NOT EXISTS poster_path text")
    conn.execute("ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_id integer")
    conn.commit()
    paths = {}
    for folder in reversed([poster_dirs] if isinstance(poster_dirs, (str, Path)) else list(poster_dirs)):
        for f in Path(folder).glob("*.json"):
            d = json.loads(f.read_text())
            if d.get("found") and d.get("poster_path"):
                paths[int(d["key"])] = d["poster_path"]
    tmdb = dict(conn.execute("SELECT movie_id, tmdb_id FROM movies").fetchall())
    with conn.cursor() as cur:
        cur.executemany("UPDATE movies SET poster_path = %s WHERE movie_id = %s",
                        [(paths[t], m) for m, t in tmdb.items() if t is not None and t in paths])
    conn.commit()
    return conn.execute("SELECT count(*) FROM movies WHERE poster_path IS NOT NULL").fetchone()[0]


def load_tmdb_ids(conn, tmdb_of):
    """tmdb_id per film (tmdb_of: Series movieId -> tmdbId)."""
    conn.execute("ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_id integer")
    with conn.cursor() as cur:
        cur.executemany("UPDATE movies SET tmdb_id = %s WHERE movie_id = %s",
                        [(int(t), int(m)) for m, t in tmdb_of.items() if t == t])
    conn.commit()


def taste_model(conn):
    """The feed's TasteModel, built from the database (films with ALS factors and vibe scores)."""
    from pgvector.psycopg import register_vector
    from .taste import TasteModel
    register_vector(conn)
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    rows = conn.execute("SELECT movie_id, tier, rating_count, year, genres, light_heavy, pacing, vibe, taste FROM movies "
                        "WHERE taste IS NOT NULL AND vibe IS NOT NULL ORDER BY movie_id").fetchall()
    ids = np.array([r[0] for r in rows])
    return TasteModel(ids, np.vstack([r[8] for r in rows]), np.vstack([r[7] for r in rows]), [r[1] for r in rows],
                      np.array([r[2] for r in rows]), np.array(meta["yty"]), meta["als"]["alpha"], meta["als"]["reg"],
                      meta["w_als"], meta["attrs"], year=np.array([r[3] if r[3] is not None else np.nan for r in rows], float),
                      genres=[r[4] for r in rows], light_heavy=np.array([np.nan if r[5] is None else r[5] for r in rows], float),
                      pacing=np.array([np.nan if r[6] is None else r[6] for r in rows], float))
