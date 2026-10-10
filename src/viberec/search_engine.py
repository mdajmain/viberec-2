"""Step 9: the search engine (moved out of notebook 11 so several notebooks use exactly the same one).

A parsed request (from the rule-based parser or the LLM parser, same format) becomes rankings that are fused:
vibe (embedding vibe profile, or the LLM's vibe_up / vibe_down) fused with BM25 on the vibe words; BM25 on literal
words; closeness to each reference movie in vibe and ALS taste. Genres filter. Then two-stage popularity: the top
n_relevant by relevance are reordered with present-day popularity (decided 2026-10-10).

Parsed format: {"reference_ids": [...], "genres": [...], "vibe_phrases": [...], "literal_phrases": [...],
                "vibe_up": [...], "vibe_down": [...]}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import genome as gn
from . import request as rq
from . import retrieval as rt
from . import text_model as tm
from .labels import movielens_keys, norm

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# the decided search setup (2026-10-10): tiers 2 + 3 searchable, Claude's parse with its own vibe picks (LLM-B),
# release year: only films older than the sample movie are pushed down (notebook 15, Adil's choice)
SEARCH_SETTINGS = {"tiers": (2, 3), "vibe_mode": "llm_attrs", "year_mode": "stage2_older"}


class SearchEngine:
    def __init__(self, root, encoder, version="v1", n_relevant=50, tiers=(3,)):
        """tiers: which quality tiers are searchable (default tier 3 only; (2, 3) adds the few-ratings tier 2)."""
        root = Path(root)
        P, A, M = root / "data" / "processed", root / "artifacts", root / "models"
        ml = root / "data" / "raw" / "movielens" / "ml-25m"
        self.encoder, self.n_relevant = encoder, n_relevant
        self.tiers = pd.read_parquet(P / "movie_tiers_v2.parquet").set_index("movieId")
        vibes = pd.read_parquet(P / f"vibe_scores_v2.parquet").set_index("movieId")
        self.attrs = [a for d in gn.VIBE_MAP.values() for a in d if a not in gn.NOT_GENOME_CHECKABLE]
        self.ids = vibes.index[vibes.index.isin(self.tiers.index[self.tiers["tier"].isin(tiers)])].to_numpy()
        v = vibes.loc[self.ids, self.attrs].astype(float).to_numpy()
        self.z = (v - v.mean(axis=0)) / v.std(axis=0)
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)
        self.year = self.tiers.loc[self.ids, "year"].to_numpy(float)
        self.age_curve, self.now_year = None, None                       # set by the caller to switch the age vote on
        wt = pd.read_parquet(P / f"wiki_text_{version}.parquet").set_index("movieId")
        tmdb_dir, tmdb_of = root / "data" / "raw" / "tmdb" / f"movie_{version}", self.tiers["tmdbId"].dropna().astype(int)

        def text(m):
            if m in wt.index:
                return tm.movie_text(wt.at[m, "plot"], wt.at[m, "reception"])
            f = tmdb_dir / f"{tmdb_of.get(m, -1)}.json"
            d = json.loads(f.read_text()) if f.exists() else {}
            return (d.get("overview") or "") + " " + " ".join(d.get("keywords") or [])
        self.texts = [text(m) for m in self.ids]
        self.bm25 = rt.BM25(self.texts)
        self.genres = self.tiers.loc[self.ids, "genres"].fillna("").str.split("|").to_numpy()
        als = np.load(M / f"als_item_factors_{version}.npy")
        als_ix = pd.Series(np.arange(len(als)), index=pd.read_parquet(M / f"als_movie_index_{version}.parquet")["movieId"])
        self.als = np.vstack([als[als_ix[m]] if m in als_ix.index else np.zeros(als.shape[1]) for m in self.ids])
        recent = pd.read_parquet(P / f"recent_popularity_latest_{version}.parquet").set_index("movieId")["recent_ratings"]
        self.recent = np.log1p(recent.reindex(self.ids).fillna(0).to_numpy())
        movies = pd.read_csv(ml / "movies.csv")
        counts = self.tiers["rating_count"].to_dict()
        self.title_index = rq.build_title_index(movies, set(self.ids), counts)
        self.title_year_index = {}
        for mid, title in zip(movies["movieId"], movies["title"]):
            if mid in self.pos.index:
                year, names = movielens_keys(title)
                for t in names:
                    self.title_year_index[(norm(t), year)] = mid
        tags = gn.map_table().groupby("attribute")["tag"].apply(list)
        self.attr_tags = {a: tags[a] for a in self.attrs}
        self.attr_texts = ["a movie that is " + ", ".join(tags[a]) for a in self.attrs]
        # the embedding model is only needed for the old embedding vibe parser; the app (Claude's vibe picks) runs without it
        self.attr_vecs = encoder.encode(self.attr_texts, normalize_embeddings=True) if encoder is not None else None
        self.router = None

    # ------------------------------------------------------------------ parsing
    def free_parse(self, request):
        p = rq.parse_request(request, self.title_index, self.router)
        return {"reference_ids": [p["reference"]] if p["reference"] is not None else [], "genres": p["genres"],
                "vibe_phrases": p["vibe_phrases"], "literal_phrases": p["literal_phrases"], "vibe_up": [], "vibe_down": []}

    def resolve_titles(self, refs):
        """LLM reference titles (+ optional year) -> catalog movieIds."""
        out = []
        for r in refs:
            key = norm(str(r.get("title", "")))
            mid = self.title_year_index.get((key, r.get("year"))) or self.title_index.get(key)
            if mid is None and key.startswith("the "):
                mid = self.title_index.get(key[4:])
            if mid is not None and mid not in out:
                out.append(mid)
        return out

    def from_llm(self, parsed):
        return {"reference_ids": self.resolve_titles(parsed.get("reference_movies", [])),
                "genres": parsed.get("required_genres", []), "vibe_phrases": parsed.get("vibe_phrases", []),
                "literal_phrases": parsed.get("literal_phrases", []), "vibe_up": parsed.get("vibe_up", []),
                "vibe_down": parsed.get("vibe_down", [])}

    # ------------------------------------------------------------------ ranking
    def relevance_parts(self, parsed, vibe_mode="embedding", rows=None, year_mode=None):
        """The named parts that relevance fuses, each a score over rows: "vibe" (the requested feel), "keywords"
        (literal words), and one "ref:<movieId>" per reference movie (its closeness in vibe and in ALS taste).
        Also returns the sub-parts of each reference part, for explanations."""
        rows = np.arange(len(self.ids)) if rows is None else np.asarray(rows)
        z, parts, sub = self.z[rows], {}, {}
        vq = " ".join(parsed["vibe_phrases"])
        if vibe_mode == "llm_attrs" and (parsed["vibe_up"] or parsed["vibe_down"]):
            prof = np.array([(a in parsed["vibe_up"]) - (a in parsed["vibe_down"]) for a in self.attrs], float)
            vibe = rt.cosine_rank_scores(z, prof)
            parts["vibe"] = rt.rrf([vibe, self.bm25.scores(vq)[rows]]) if vq else vibe
        elif vq:
            prof = rt.embedding_vibe_profile(self.encoder, vq, self.attr_texts, self.attr_vecs)
            parts["vibe"] = rt.rrf([rt.cosine_rank_scores(z, prof), self.bm25.scores(vq)[rows]])
        if parsed["literal_phrases"]:
            parts["keywords"] = self.bm25.scores(" ".join(parsed["literal_phrases"]))[rows]
        for ref in parsed["reference_ids"]:
            r = self.pos[ref]
            a = self.als[r]
            als_sim = self.als[rows] @ a / (np.linalg.norm(self.als[rows], axis=1) * (np.linalg.norm(a) or 1) + 1e-9)
            pieces = {"same_feel": rt.cosine_rank_scores(z, self.z[r]), "similar_taste": als_sim}
            if year_mode == "closeness":
                pieces["year"] = np.nan_to_num(-np.abs(self.year[rows] - self.year[r]), nan=-np.inf)
            parts[f"ref:{ref}"] = rt.rrf(list(pieces.values()))
            sub[f"ref:{ref}"] = pieces
        return parts, sub

    def relevance(self, parsed, vibe_mode="embedding", rows=None, year_mode=None, skip_parts=()):
        """Fused relevance score over the catalog (or over the given catalog rows). None if nothing to rank on.
        year_mode="closeness": closeness to a reference movie also counts its release year (nearer = closer).
        skip_parts: part names to leave out (for testing what each part does)."""
        rows = np.arange(len(self.ids)) if rows is None else np.asarray(rows)
        parts, _ = self.relevance_parts(parsed, vibe_mode, rows, year_mode)
        rankings = [v for k, v in parts.items() if k not in skip_parts]
        if not rankings:
            return None
        score = rt.rrf(rankings) if len(rankings) > 1 else rankings[0]
        if parsed["genres"]:
            ok = np.array([all(g in gs for g in parsed["genres"]) for gs in self.genres[rows]])
            score = np.where(ok, score, -np.inf)
        for ref in parsed["reference_ids"]:
            hit = np.where(self.ids[rows] == ref)[0]
            score[hit] = -np.inf
        return score

    def rank(self, parsed, k=10, vibe_mode="embedding", rows=None, popularity=True, year_mode=None, age=False, skip_parts=()):
        """movieIds: top n_relevant by relevance, reordered with present-day popularity (two-stage).
        year_mode="closeness": year inside reference closeness; "stage2": closeness in year to the nearest reference
        is an extra vote in the reorder; "stage2_older": the same, but only films older than the reference are pushed down. age=True: the age curve (self.age_curve at self.now_year) is an extra vote."""
        rows = np.arange(len(self.ids)) if rows is None else np.asarray(rows)
        score = self.relevance(parsed, vibe_mode, rows, year_mode=year_mode if year_mode == "closeness" else None,
                               skip_parts=skip_parts)
        if score is None:
            return []
        finite = np.where(np.isfinite(score))[0]
        top = finite[np.argsort(-score[finite])][:self.n_relevant]
        if popularity and len(top):
            votes = [-np.arange(len(top), dtype=float), self.recent[rows][top]]
            refs = [self.pos[m] for m in parsed["reference_ids"]]
            if year_mode in ("stage2", "stage2_older") and refs:
                diff = [self.year[r] - self.year[rows][top] for r in refs]          # > 0: older than the reference
                gap = np.min([np.abs(d) if year_mode == "stage2" else np.maximum(d, 0) for d in diff], axis=0)
                votes.append(np.nan_to_num(-gap, nan=-np.inf))
            if age:
                from .years import lookup
                votes.append(np.nan_to_num(lookup(self.age_curve, self.now_year - self.year[rows][top]), nan=-np.inf))
            order = top[np.argsort(-rt.rrf(votes))]
        else:
            order = top
        return list(self.ids[rows][order[:k]])


def tag_benchmark(root, version="v1", min_users=2, min_movies=10):
    """The Step 9 tag test: Step 5 held-out movies (pool) and MovieLens tags applied by >= min_users users to
    >= min_movies of them. Returns (pool_ids, relevant: Series tag -> set of movieIds, is_vibe: Series tag -> bool)."""
    root = Path(root)
    P = root / "data" / "processed"
    wt = pd.read_parquet(P / f"wiki_text_{version}.parquet").set_index("movieId")
    genome_ids = pd.read_parquet(P / f"genome_matrix_{version}.parquet", columns=[]).index
    _, pool = tm.split_ids(wt.index[wt["plot_words"] > 0].intersection(genome_ids), test_share=0.2, seed=42)
    tags = pd.read_csv(root / "data" / "raw" / "movielens" / "ml-25m" / "tags.csv", usecols=["userId", "movieId", "tag"])
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["movieId"].isin(pool)]
    per = tags.groupby(["tag", "movieId"])["userId"].nunique().reset_index()
    rel = per[per["userId"] >= min_users].groupby("tag")["movieId"].apply(set)
    rel = rel[rel.map(len) >= min_movies]
    vocab = set(gn.map_table()["tag"])
    return pool, rel, pd.Series({q: q in vocab for q in rel.index})


def fit_router(engine, tag_texts, tag_is_vibe):
    """The Step 9 router: tag queries + the taxonomy's vibe words, classes balanced (decided 2026-10-09)."""
    extra = sorted(set(gn.map_table()["tag"]) - set(tag_texts))
    engine.router = rq.Router(engine.encoder).fit(list(tag_texts) + extra,
                                                  np.concatenate([np.asarray(tag_is_vibe, int), np.ones(len(extra), int)]),
                                                  balanced=True)
    return engine.router


# ---------------------------------------------------------------------- two-step design (Adil, 2026-10-10)
NEIGHBOUR_FEATURES = ["same_feel", "similar_taste", "log_recent", "years_older"]


def neighbour_features(engine, ref, rows=None):
    """Per catalog row, the features of closeness to one reference movie: vibe cosine (same feel), ALS cosine
    (same fans), present-day popularity (log), and how many years older than the reference (0 if newer)."""
    rows = np.arange(len(engine.ids)) if rows is None else np.asarray(rows)
    r = engine.pos[ref]
    a = engine.als[r]
    als_sim = engine.als[rows] @ a / (np.linalg.norm(engine.als[rows], axis=1) * (np.linalg.norm(a) or 1) + 1e-9)
    older = np.maximum(engine.year[r] - engine.year[rows], 0)
    return pd.DataFrame({"same_feel": rt.cosine_rank_scores(engine.z[rows], engine.z[r]), "similar_taste": als_sim,
                         "log_recent": engine.recent[rows], "years_older": np.nan_to_num(older, nan=np.nanmax(older))})


def rank_two_step(engine, parsed, k=10, n_neighbours=30, vibe_mode="llm_attrs", learned=None, n_candidates=300):
    """Step 1: the n_neighbours films closest to the sample movie(s). Step 2: rank only those by equal votes of what
    the prompt adds (vibe, keywords) plus closeness, popularity and "not older than the sample".

    learned: optional (scaler, model) over NEIGHBOUR_FEATURES; its score then defines closeness (step 1) and replaces
    the closeness / popularity / year votes in step 2. It only re-scores the n_candidates films closest by the equal
    vote (as the Step 10 ranker does), so a weight on popularity cannot pull in unrelated famous films.
    Without a sample movie: the current search (engine.rank)."""
    refs = [m for m in parsed["reference_ids"] if m in engine.pos.index]
    if not refs:
        return engine.rank(parsed, k=k, vibe_mode=vibe_mode, year_mode="stage2_older")
    rows = np.arange(len(engine.ids))
    if parsed["genres"]:
        rows = rows[np.array([all(g in gs for g in parsed["genres"]) for gs in engine.genres])]
    rows = rows[~np.isin(engine.ids[rows], refs)]
    feats = [neighbour_features(engine, ref, rows) for ref in refs]
    if learned is not None:
        equal = np.max([rt.rrf([f["same_feel"].to_numpy(), f["similar_taste"].to_numpy()]) for f in feats], axis=0)
        keep = np.argsort(-equal)[:n_candidates]
        rows, feats = rows[keep], [f.iloc[keep].reset_index(drop=True) for f in feats]
        scaler, model = learned
        close = np.max([model.decision_function(scaler.transform(f[NEIGHBOUR_FEATURES].to_numpy())) for f in feats], axis=0)
        votes_fixed = [close]
    else:
        close = np.max([rt.rrf([f["same_feel"].to_numpy(), f["similar_taste"].to_numpy()]) for f in feats], axis=0)
        older = np.min([f["years_older"].to_numpy() for f in feats], axis=0)
        votes_fixed = [close, engine.recent[rows], -older]
    hood = np.argsort(-close)[:n_neighbours]
    parts, _ = engine.relevance_parts(dict(parsed, reference_ids=[]), vibe_mode, rows[hood])
    votes = [v[hood] for v in votes_fixed] + list(parts.values())
    order = hood[np.argsort(-rt.rrf(votes))]
    return list(engine.ids[rows][order[:k]])



from .db_search import blend  # noqa: E402,F401  (defined there so the app image does not need this module)
