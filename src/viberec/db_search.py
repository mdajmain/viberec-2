"""Step 13: the decided search (SEARCH_SETTINGS), answered from PostgreSQL + pgvector instead of numpy arrays.

Same parts and the same fusion as search_engine.SearchEngine (notebook 15's "current" design):
- vibe: cosine to Claude's vibe profile (pgvector `<=>`), fused with BM25 on the vibe words;
- keywords: BM25 on the literal words (inverted index table);
- one part per sample movie: cosine in vibe (same feel) and in ALS taste (same fans), fused;
- equal-vote fusion (rrf), genre filter, the sample itself removed;
- top n_relevant reordered by present-day popularity and "not older than the sample".

The one difference: each part fetches only its top `depth` films, as vector databases do, instead of ranking the whole
catalog. A film outside a part's top `depth` gets nothing from that part. Notebook 16 checks this against the in-memory
search on Adil's requests.
"""
import json

import numpy as np

from .labels import norm

RRF_K = 60


def min_ranks(scores):
    """{id: score} -> {id: rank}, highest score rank 1, tied scores share a rank (1, 1, 3, ...), as retrieval.rrf."""
    out, prev, rank = {}, None, 0
    for i, (m, sc) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]), 1):
        if sc != prev:
            rank, prev = i, sc
        out[m] = rank
    return out


def rrf_scores(score_dicts, k=RRF_K):
    """Reciprocal rank fusion of {id: score} dicts, each holding only its top films (as a database returns them).
    A film missing from a dict ranks just below everything in it (len + 1), as a tie: what retrieval.rrf gives the
    films a part scores lowest (e.g. zero keyword score) when the dict holds all films the part scores above that."""
    universe = set().union(*[set(d) for d in score_dicts])
    out = dict.fromkeys(universe, 0.0)
    for d in score_dicts:
        ranks, miss = min_ranks(d), 1.0 / (k + len(d) + 1)
        for m in universe:
            out[m] += 1.0 / (k + ranks[m]) if m in ranks else miss
    return out


def blend(first, second, n_first, k=5):
    """A k-film list: the first n_first films of `first`, then films of `second` not already taken; if one list runs
    short, the other fills the gap."""
    out = list(dict.fromkeys(first[:n_first]))
    for m in list(second) + list(first[n_first:]):
        if len(out) >= k:
            break
        if m not in out:
            out.append(m)
    return out[:k]


def ranked(scores):
    """{id: score} -> ids, highest score first (ties in insertion order)."""
    return [m for m, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


class DbSearch:
    def __init__(self, conn, depth=4000, n_relevant=50, meta=None):          # depth chosen in notebook 16
        """meta: the meta table as a dict, when the caller already has it (the app passes it to avoid a query per request)."""
        self.conn, self.depth, self.n_relevant = conn, depth, n_relevant
        if meta is None:
            with conn.cursor() as cur:
                cur.execute("SELECT key, value FROM meta")
                meta = dict(cur.fetchall())
        self.attrs, self.attr_tags = meta["attrs"], meta["attr_tags"]

    # ------------------------------------------------------------------ helpers
    def _rows(self, sql, params):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def vector_top(self, column, query_vec, genres=(), within=None, exclude=()):
        """{film: cosine similarity of `column` to query_vec} for the `depth` most similar (exact search).
        genres: only films with all of them; within: only these films; exclude: never these films."""
        q = "[" + ",".join(f"{v:.6g}" for v in query_vec) + "]"
        where, params = [f"{column} IS NOT NULL"], [q]
        if genres:
            where.append("genres @> %s"); params.append(list(genres))
        if within is not None:
            where.append("movie_id = ANY(%s)"); params.append([int(m) for m in within])
        if exclude:
            where.append("NOT movie_id = ANY(%s)"); params.append([int(m) for m in exclude])
        return dict(self._rows(f"SELECT movie_id, 1 - ({column} <=> %s::vector) AS s FROM movies WHERE {' AND '.join(where)} "
                               f"ORDER BY {column} <=> %s::vector, movie_id LIMIT %s", (*params, q, self.depth)))

    def bm25_top(self, text, analyzer, within=None):
        terms = list(dict.fromkeys(analyzer(text)))
        if not terms:
            return {}
        extra, params = ("AND p.movie_id = ANY(%s) ", [terms, [int(m) for m in within]]) if within is not None else ("", [terms])
        return dict(self._rows(
            "SELECT p.movie_id, SUM(p.w * i.idf) AS s FROM bm25_postings p JOIN bm25_idf i USING (term) "
            f"WHERE p.term = ANY(%s) {extra}GROUP BY p.movie_id ORDER BY s DESC, p.movie_id LIMIT %s", (*params, self.depth)))

    def movie_vectors(self, movie_id):
        r = self._rows("SELECT vibe::text, taste::text FROM movies WHERE movie_id = %s", (int(movie_id),))
        if not r:
            return None, None
        parse = lambda s: np.array(json.loads(s)) if s else None
        return parse(r[0][0]), parse(r[0][1])

    # ------------------------------------------------------------------ parsing
    def resolve_titles(self, refs):
        out = []
        for r in refs:
            key = norm(str(r.get("title", "")))
            found = self._rows("SELECT movie_id FROM titles WHERE norm_title = %s AND year IS NOT DISTINCT FROM %s LIMIT 1",
                               (key, r.get("year"))) if r.get("year") else []
            for k in [key] + ([key[4:]] if key.startswith("the ") else []):
                if found:
                    break
                found = self._rows("SELECT movie_id FROM titles WHERE norm_title = %s AND year IS NULL LIMIT 1", (k,))
            if found and found[0][0] not in out:
                out.append(found[0][0])
        return out

    def from_llm(self, parsed):
        return {"reference_ids": self.resolve_titles(parsed.get("reference_movies", [])),
                "genres": parsed.get("required_genres", []), "vibe_phrases": parsed.get("vibe_phrases", []),
                "literal_phrases": parsed.get("literal_phrases", []), "vibe_up": parsed.get("vibe_up", []),
                "vibe_down": parsed.get("vibe_down", [])}

    # ------------------------------------------------------------------ ranking
    def parts(self, parsed, analyzer):
        """{part name: {film: score}}; reference parts also keep their two sub-parts for explanations."""
        parts, sub = {}, {}
        vq = " ".join(parsed["vibe_phrases"])
        if parsed["vibe_up"] or parsed["vibe_down"]:
            prof = [float((a in parsed["vibe_up"]) - (a in parsed["vibe_down"])) for a in self.attrs]
            lists = [self.vector_top("vibe", prof)] + ([self.bm25_top(vq, analyzer)] if vq else [])
            parts["vibe"] = rrf_scores(lists) if len(lists) > 1 else lists[0]
        elif vq:
            parts["vibe"] = self.bm25_top(vq, analyzer)          # no embedding model in the app: vibe words only
        if parsed["literal_phrases"]:
            parts["keywords"] = self.bm25_top(" ".join(parsed["literal_phrases"]), analyzer)
        for ref in parsed["reference_ids"]:
            v, t = self.movie_vectors(ref)
            pieces = {}
            if v is not None:
                pieces["same_feel"] = self.vector_top("vibe", v)
            if t is not None:
                pieces["similar_taste"] = self.vector_top("taste", t)
            if pieces:
                parts[f"ref:{ref}"] = rrf_scores(list(pieces.values()))
                sub[f"ref:{ref}"] = pieces
        return parts, sub

    def search(self, parsed, analyzer, k=5):
        """Top k movie ids, plus the parts (for the 'why')."""
        parts, sub = self.parts(parsed, analyzer)
        if not parts:
            return [], parts, sub
        fused = rrf_scores(list(parts.values())) if len(parts) > 1 else next(iter(parts.values()))
        refs = set(parsed["reference_ids"])
        cand = [m for m in ranked(fused) if m not in refs]
        info = {m: (rl, y, g) for m, rl, y, g in self._rows(
            "SELECT movie_id, recent_log, year, genres FROM movies WHERE movie_id = ANY(%s)", (cand,))}
        if parsed["genres"]:
            cand = [m for m in cand if all(x in (info[m][2] or []) for x in parsed["genres"])]
        top = cand[: self.n_relevant]
        if not top:
            return [], parts, sub
        votes = [{m: -i for i, m in enumerate(top)}, {m: info[m][0] for m in top}]
        ref_years = [y for (y,) in self._rows("SELECT year FROM movies WHERE movie_id = ANY(%s)", (list(refs),))
                     if y is not None] if refs else []
        if ref_years:
            older = {m: (min(max(ry - info[m][1], 0) for ry in ref_years) if info[m][1] is not None else 1e9) for m in top}
            votes.append({m: -g for m, g in older.items()})
        final = ranked(rrf_scores(votes))
        return final[:k], parts, sub


    # ------------------------------------------------------------------ Adil's two-step design and the 3/2 blend
    def two_step(self, parsed, analyzer, k=5, n_neighbours=30):
        """Step 1: the n_neighbours films closest to the sample (same feel + same fans, among films with the required
        genres). Step 2: rank them by equal votes of closeness, present-day popularity, "not older than the sample" and
        what the prompt adds (vibe, keywords), each computed within the neighbourhood. As search_engine.rank_two_step."""
        refs = list(parsed["reference_ids"])
        if not refs:
            return self.search(parsed, analyzer, k=k)[0]
        closeness = {}
        for ref in refs:
            v, t = self.movie_vectors(ref)
            pieces = [d for d in (self.vector_top("vibe", v, parsed["genres"], exclude=refs) if v is not None else None,
                                  self.vector_top("taste", t, parsed["genres"], exclude=refs) if t is not None else None) if d]
            for m, sc in (rrf_scores(pieces) if pieces else {}).items():
                closeness[m] = max(closeness.get(m, -1.0), sc)
        hood = ranked(closeness)[:n_neighbours]
        if not hood:
            return []
        info = {m: (rl, y) for m, rl, y in self._rows("SELECT movie_id, recent_log, year FROM movies WHERE movie_id = ANY(%s)", (hood,))}
        ref_years = [y for (y,) in self._rows("SELECT year FROM movies WHERE movie_id = ANY(%s)", (refs,)) if y is not None]
        older = {m: (min(max(ry - info[m][1], 0) for ry in ref_years) if info[m][1] is not None and ref_years else None) for m in hood}
        worst = max([g for g in older.values() if g is not None], default=0)
        votes = [{m: closeness[m] for m in hood}, {m: info[m][0] for m in hood},
                 {m: -(g if g is not None else worst) for m, g in older.items()}]
        vq = " ".join(parsed["vibe_phrases"])
        if parsed["vibe_up"] or parsed["vibe_down"]:
            prof = [float((a in parsed["vibe_up"]) - (a in parsed["vibe_down"])) for a in self.attrs]
            lists = [self.vector_top("vibe", prof, within=hood)] + ([self.bm25_top(vq, analyzer, within=hood)] if vq else [])
            votes.append(rrf_scores(lists) if len(lists) > 1 else lists[0])
        elif vq:
            votes.append(self.bm25_top(vq, analyzer, within=hood))
        if parsed["literal_phrases"]:
            votes.append(self.bm25_top(" ".join(parsed["literal_phrases"]), analyzer, within=hood))
        fused = rrf_scores([v if v else {} for v in votes if v is not None])
        return ranked({m: fused.get(m, 0.0) for m in hood})[:k]

    def blended(self, parsed, analyzer, k=5, n_first=3, n_neighbours=30):
        """The 3/2 blend (Adil, 2026-10-10): the first n_first films of the current search, then the two-step design's
        films not already taken. Returns (ids, parts, sub) like search(), the parts of the current search."""
        first, parts, sub = self.search(parsed, analyzer, k=max(k, 10))
        second = self.two_step(parsed, analyzer, k=max(k, 10), n_neighbours=n_neighbours)
        return blend(first, second, n_first, k=k), parts, sub

    def blended_with_designs(self, parsed, analyzer, k=5, n_first=3, n_neighbours=30):
        """blended() plus which design picked each film ("fusion" or "two_step"), for the history log."""
        first, parts, sub = self.search(parsed, analyzer, k=max(k, 10))
        second = self.two_step(parsed, analyzer, k=max(k, 10), n_neighbours=n_neighbours)
        top = blend(first, second, n_first, k=k)
        designs = ["fusion" if m in first[:n_first] or m not in second else "two_step" for m in top]
        return top, designs, parts, sub


def bm25_analyzer():
    """The same word splitting as retrieval.BM25 (lowercase, accents stripped, word characters)."""
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(lowercase=True, strip_accents="unicode", token_pattern=r"(?u)\b\w+\b").build_analyzer()
