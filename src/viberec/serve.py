"""Step 13: the VibeRec service behind the web app (app v2, 2026-10-10).

Search:
1. Claude reads the prompt (llm_parse, cached by prompt text);
2. the decided search runs on PostgreSQL + pgvector (db_search.DbSearch): 3 films from the current search, then 2 from
   the two-step design (Adil's 3/2 blend);
3. one Claude call writes the "why" for every result not already cached (Bedrock allows about 10 calls a minute);
4. each "why" is checked: a spoiler or a film not in the evidence -> the free template sentence; an extra mood word is
   kept and logged.
Profiles, quiz, "For you" and history (userdata.py, taste.py):
- every visitor is a guest or a member profile; reactions, searches and what was shown are logged per profile;
- the quiz asks the Step 8 list until 20 likes (cap 100); after every reaction the taste profile is recomputed;
- "For you" = the tested Step 8 method (ALS fold-in + vibe blend), explained like search results.
Each request takes its own connection from a pool, so several people can use the app at once.
"""
import hashlib
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from . import explain as ex
from . import llm_parse as lp
from . import userdata as ud
from .db_search import DbSearch, bm25_analyzer, min_ranks
from .llm_budget import guarded_converse
from .taste import QUIZ_CAP, QUIZ_TARGET_LIKES, next_question

BATCH_RULES = """
You will get several movies as a JSON list. Write one "why" per movie, in the same order, and record them with the
record_whys tool. Never use the word "twist" and never hint at how a story ends."""
BATCH_SYSTEM = ex.SEARCH_SYSTEM + BATCH_RULES
FEED_SYSTEM = """You write the short "why" shown under one movie in a person's personal picks ("For you").
Use ONLY the evidence given: the movie's genres, feel words, premise, and the reasons (listed most important first).
Rules:
- Start by saying what kind of film it is, using the feel words and genres given; you may mention its setup, but only
  from the premise, and never go beyond the premise.
- Then say why it suits this person, following the order of the reasons.
- Speak of taste as "viewers with similar taste" or "the films you like"; never name one film as the reason.
- No spoilers: never mention the ending, twists, deaths or how the story turns out.
- No algorithms, scores or numbers. Plain text, two sentences, at most 45 words. Do not start with the title.""" + BATCH_RULES
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
SEARCH_LIMITS = {"guest": 10, "member": 30, None: 20}      # searches per hour; None = calls without a profile (all together)
FEED_LIMITS = {"guest": 20, "member": 60}                  # new "For you" lists per hour
VERDICTS = {"up", "down", "not_this_vibe"}                 # the old page's feedback buttons
log = logging.getLogger("viberec")


class RateLimited(Exception):
    pass


def batch_tool(n):
    return {"tools": [{"toolSpec": {"name": "record_whys", "description": "Record one 'why' per movie, in order.",
                                    "inputSchema": {"json": {"type": "object", "properties": {
                                        "whys": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}},
                                        "required": ["whys"]}}}}],
            "toolChoice": {"tool": {"name": "record_whys"}}}


def feel_words(me):
    """Feel words for a film: Light/Heavy, its strongest vibes, Pacing."""
    feel = ex.top_feel(me["z"].to_numpy(), list(me["z"].index))
    if me["light_heavy"] is not None and ex.LIGHT_WORDS[int(round(me["light_heavy"]))]:
        feel = [ex.LIGHT_WORDS[int(round(me["light_heavy"]))]] + feel
    if me["pacing"] is not None:
        pw = ex.PACE_WORDS[int(round(me["pacing"]))]
        if pw and pw not in feel:
            feel.append(pw)
    return feel


class VibeRecService:
    def __init__(self, pool, client, guard, cache_dir, taste_model=None, model_id=lp.MODEL_ID):
        """pool: a psycopg_pool.ConnectionPool (or anything whose .connection() yields a connection)."""
        self.pool, self.client, self.guard, self.model_id = pool, client, guard, model_id
        self.cache = Path(cache_dir)
        self.analyzer = bm25_analyzer()
        with pool.connection() as conn:
            ud.create_schema(conn)
            self.meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            self.catalog = [t for (t,) in conn.execute("SELECT title FROM movies ORDER BY rating_count DESC LIMIT 5000")]
        self.attrs = self.meta["attrs"]
        self.quiz = self.meta.get("quiz", [])
        self.taste = taste_model

    # ------------------------------------------------------------------ films as cards
    def _movies(self, conn, ids):
        cols = "movie_id, display_title, year, genres, recent_log, premise, light_heavy, pacing, vibe::text, poster_path"
        out = {}
        for m, title, year, genres, rl, premise, lh, pace, vibe, poster in conn.execute(
                f"SELECT {cols} FROM movies WHERE movie_id = ANY(%s)", (list(map(int, ids)),)):
            out[m] = {"title": title, "year": year, "genres": [g for g in (genres or []) if g != "(no genres listed)"],
                      "recent": int(round(np.expm1(rl or 0))), "premise": premise or "", "light_heavy": lh, "pacing": pace,
                      "z": pd.Series(json.loads(vibe), index=self.attrs) if vibe else pd.Series(0.0, index=self.attrs),
                      "poster": f"{POSTER_BASE}{poster}" if poster else None}
        return out

    @staticmethod
    def card(m, me, **extra):
        return {"movie_id": int(m), "title": me["title"], "year": me["year"], "genres": me["genres"],
                "premise": me["premise"], "poster": me["poster"], **extra}

    # ------------------------------------------------------------------ evidence for search results
    def _words_found(self, conn, movie_id, phrases):
        terms = ex.words_in_text(phrases, " ".join(phrases))                     # the request's content words
        if not terms:
            return []
        have = {t for (t,) in conn.execute("SELECT term FROM bm25_postings WHERE movie_id = %s AND term = ANY(%s)",
                                           (int(movie_id), terms))}
        return [t for t in terms if t in have]

    def evidence(self, conn, request, parsed, movie_id, parts, sub, info):
        me = info[movie_id]
        ev = {"request": request, "movie": me["title"], "year": me["year"], "genres": me["genres"], "feel": feel_words(me),
              "premise": me["premise"], "reasons": []}
        contrib = {name: 1.0 / (60 + min_ranks(d).get(movie_id, len(d) + 1)) for name, d in parts.items()}
        for name in sorted(contrib, key=lambda n: -contrib[n]):
            if name == "vibe":
                prof = pd.Series([(a in parsed["vibe_up"]) - (a in parsed["vibe_down"]) for a in self.attrs],
                                 index=self.attrs, dtype=float)
                part = prof * me["z"]
                more = [ex.VIBE_WORDS[a] for a in part[(part > 0) & (prof > 0)].sort_values(ascending=False).index
                        if a not in ex.SPOILER_VIBES][:3]
                less = [ex.VIBE_WORDS[a] for a in part[(part > 0) & (prof < 0)].sort_values(ascending=False).index
                        if a not in ex.SPOILER_VIBES][:2]
                if more or less:
                    ev["reasons"].append({"reason": "matches the feel you asked for", "has": more, "is_not": less})
            elif name == "keywords":
                found = self._words_found(conn, movie_id, parsed["literal_phrases"])
                if found:
                    ev["reasons"].append({"reason": "its story has what you described", "words": found})
            elif name.startswith("ref:"):
                ref = int(name[4:])
                pieces = {p: 1.0 / (60 + min_ranks(d).get(movie_id, len(d) + 1)) for p, d in sub[name].items()}
                shared = ex.vibe_contributions(me["z"], info[ref]["z"]).drop(labels=list(ex.SPOILER_VIBES & set(self.attrs)))
                shared_words = [ex.VIBE_WORDS[a] for a in ex.top_positive(shared, 3)]
                for piece in sorted(pieces, key=lambda p: -pieces[p]):
                    if piece == "similar_taste":
                        ev["reasons"].append({"reason": "viewers with similar taste rate it highly"})
                    elif piece == "same_feel" and shared_words:
                        ev["reasons"].append({"reason": "same feel as the movie you gave",
                                              "sample_movie": info[ref]["title"], "shared_feel": shared_words})
        ev["reasons"].append({"reason": "well known", "recent_ratings": me["recent"]})
        return ev

    # ------------------------------------------------------------------ wording
    def whys(self, evidences, system=BATCH_SYSTEM, purpose="app why"):
        """Claude's "why" for each evidence; each is cached on its own, and the ones not cached yet are asked for in
        ONE call. Any that fails the checks becomes the free template."""
        keys = [hashlib.sha1((json.dumps(ev, sort_keys=True) + system).encode()).hexdigest()[:20] for ev in evidences]
        folder = self.cache / "why"
        got = {}
        for k in keys:
            f = folder / f"{k}.json"
            if f.exists():
                got[k] = json.loads(f.read_text())["why"]
        missing = [i for i, k in enumerate(keys) if k not in got]
        if missing:
            todo = [evidences[i] for i in missing]
            resp = guarded_converse(self.client, self.guard, purpose, self.model_id,
                                    [{"role": "user", "content": [{"text": json.dumps(todo, sort_keys=True)}]}],
                                    {"maxTokens": 120 * len(todo), "temperature": 0},
                                    system=[{"text": system}], toolConfig=batch_tool(len(todo)))
            blocks = [b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b]
            raw = blocks[0].get("whys", []) if blocks else []
            folder.mkdir(parents=True, exist_ok=True)
            for j, i in enumerate(missing):
                w = raw[j] if j < len(raw) and isinstance(raw[j], str) else ""
                if w:
                    (folder / f"{keys[i]}.json").write_text(json.dumps({"evidence": evidences[i], "why": w}, indent=1))
                got[keys[i]] = w
        out, sources = [], []
        for k, ev in zip(keys, evidences):
            w = ex.clean_text(got.get(k) or "")
            words = set(ev["feel"]) | {x for r in ev["reasons"] for key in ("has", "is_not", "shared_feel") for x in r.get(key, [])}
            allowed_titles = [ev["movie"]] + [r["sample_movie"] for r in ev["reasons"] if "sample_movie" in r]
            checks = ex.grounding_problems(w, allowed_titles, [a for a, x in ex.VIBE_WORDS.items() if x in words],
                                           self.catalog, allowed_genres=ev["genres"], evidence_text=ev["premise"]) if w else []
            # strict where Adil's rules are strict: a spoiler or a film not in the evidence -> template;
            # an extra mood / genre word ("witty") is kept and logged
            severe = ex.spoiler_problems(w, premise=ev["premise"]) + [c for c in checks if c.startswith("title:")] if w else ["empty"]
            mild = [c for c in checks if not c.startswith("title:")]
            if mild:
                log.info("why for %s kept with extra words: %s", ev["movie"], mild)
            out.append(w if not severe else ex.search_template(ev))
            sources.append("claude" if not severe else "template")
        return out, sources

    # ------------------------------------------------------------------ profiles
    def profile(self, guest_id=None, clerk_user_id=None):
        """The profile for this visitor: the member profile when signed in (merging this browser's guest history),
        else the guest profile for the cookie id."""
        with self.pool.connection() as conn:
            if clerk_user_id:
                pid = ud.sign_in(conn, clerk_user_id, guest_id)
                self._update_taste(conn, pid)
            else:
                pid = ud.ensure_guest(conn, guest_id)
            return {"profile_id": pid, "kind": ud.kind_of(conn, pid)}

    def _kind(self, conn, profile_id):
        kind = ud.kind_of(conn, profile_id)
        if kind not in ("guest", "member"):
            raise KeyError("unknown profile")
        return kind

    def _update_taste(self, conn, profile_id):
        answers = ud.answers(conn, profile_id)
        likes = [m for m, v in answers.items() if v == "like"]
        summary = self.taste.summary(likes, n_dislikes=sum(v == "dislike" for v in answers.values()),
                                     n_unseen=sum(v == "unseen" for v in answers.values()))
        vibe = self.taste.vibe_taste(likes)
        ud.save_taste(conn, profile_id, self.taste.taste_vector(likes), vibe, len(likes), len(answers), summary)
        return answers, summary

    # ------------------------------------------------------------------ quiz and reactions
    def quiz_next(self, profile_id):
        with self.pool.connection() as conn:
            self._kind(conn, profile_id)
            answers = ud.answers(conn, profile_id)
            m = next_question(self.quiz, answers)
            likes = sum(v == "like" for v in answers.values())
            out = {"done": m is None, "likes": likes, "answered": len(answers), "target_likes": QUIZ_TARGET_LIKES,
                   "cap": QUIZ_CAP, "movie": None}
            if m is not None:
                out["movie"] = self.card(m, self._movies(conn, [m])[m])
            return out

    def react(self, profile_id, movie_id, verdict, source, context_id=None):
        with self.pool.connection() as conn:
            self._kind(conn, profile_id)
            ud.add_reaction(conn, profile_id, movie_id, verdict, source, context_id)
            answers, summary = self._update_taste(conn, profile_id)
            return {"likes": summary["likes"], "answered": len(answers)}

    # ------------------------------------------------------------------ "For you"
    def feed(self, profile_id, k=10):
        with self.pool.connection() as conn:
            kind = self._kind(conn, profile_id)
            recent = conn.execute("SELECT count(DISTINCT context_id) FROM shown WHERE profile_id = %s AND context = 'feed' "
                                  "AND created > now() - interval '1 hour'", (profile_id,)).fetchone()[0]
            if recent >= FEED_LIMITS[kind]:
                raise RateLimited("Too many new lists this hour; try again later.")
            answers = ud.answers(conn, profile_id)
            likes = [m for m, v in answers.items() if v == "like"]
            top = self.taste.recommend(likes, exclude=list(answers), k=k)
            info = self._movies(conn, top)
            parts = self.taste.parts_for(likes, top) if likes else {}
            evidences = []
            for m in top:
                me = info[m]
                ev = {"movie": me["title"], "year": me["year"], "genres": me["genres"], "feel": feel_words(me),
                      "premise": me["premise"], "reasons": []}
                als_part, vibe_part = parts.get(m, (0.0, 0.0))
                reasons = {"als": {"reason": "viewers with similar taste rate it highly"},
                           "vibe": {"reason": "it has the feel of the films you like", "shared_feel": self.taste.shared_feel(likes, m)}}
                for key in sorted(("als", "vibe"), key=lambda x: -(als_part if x == "als" else vibe_part)):
                    if (als_part if key == "als" else vibe_part) > 0 and (key == "als" or reasons["vibe"]["shared_feel"]):
                        ev["reasons"].append(reasons[key])
                ev["reasons"].append({"reason": "well known", "recent_ratings": me["recent"]})
                evidences.append(ev)
            texts, sources = self.whys(evidences, FEED_SYSTEM, "app feed why") if likes else \
                ([ex.search_template(ev) for ev in evidences], ["template"] * len(evidences))
            feed_id = f"f_{uuid.uuid4().hex}"
            ud.log_shown(conn, profile_id, "feed", feed_id, top, ["feed"] * len(top))
            return {"feed_id": feed_id, "personal": bool(likes),
                    "results": [self.card(m, info[m], why=t, why_source=s) for m, t, s in zip(top, texts, sources)]}

    # ------------------------------------------------------------------ search
    def search(self, prompt, k=5, profile_id=None):
        with self.pool.connection() as conn:
            kind = self._kind(conn, profile_id) if profile_id else None
            used = ud.searches_since(conn, profile_id) if profile_id else conn.execute(
                "SELECT count(*) FROM searches WHERE profile_id IS NULL AND created > now() - interval '1 hour'").fetchone()[0]
            if used >= SEARCH_LIMITS[kind]:
                raise RateLimited("Too many searches this hour; try again later.")
            db = DbSearch(conn, meta=self.meta)
            parsed_llm = lp.parse(self.client, self.guard, prompt, db.attr_tags, self.cache / "parse")
            parsed = db.from_llm(parsed_llm)
            top, designs, parts, sub = db.blended_with_designs(parsed, self.analyzer, k=k)   # 3/2 blend (Adil)
            info = self._movies(conn, list(top) + list(parsed["reference_ids"]))
            texts, sources = self.whys([self.evidence(conn, prompt, parsed, m, parts, sub, info) for m in top])
            understood = {"sample_movies": [info[r]["title"] for r in parsed["reference_ids"] if r in info],
                          "genres": parsed["genres"], "more_of": parsed["vibe_up"], "less_of": parsed["vibe_down"],
                          "about": parsed["literal_phrases"]}
            search_id = ud.log_search(conn, profile_id, prompt, understood)
            ud.log_shown(conn, profile_id, "search", search_id, top, designs)
            return {"search_id": search_id, "prompt": prompt, "understood": understood,
                    "results": [self.card(m, info[m], why=t, why_source=s) for m, t, s in zip(top, texts, sources)]}

    # ------------------------------------------------------------------ the person's own pages
    def taste_summary(self, profile_id):
        with self.pool.connection() as conn:
            self._kind(conn, profile_id)
            row = conn.execute("SELECT summary FROM taste_profiles WHERE profile_id = %s", (profile_id,)).fetchone()
            return row[0] if row else self.taste.summary([])

    def history(self, profile_id):
        with self.pool.connection() as conn:
            self._kind(conn, profile_id)
            h = ud.history(conn, profile_id)
        for part in (h["reactions"], *[s["results"] for s in h["searches"]]):
            for r in part:
                r["poster"] = f"{POSTER_BASE}{r.pop('poster_path')}" if r.get("poster_path") else None
        return h

    def forget(self, profile_id):
        with self.pool.connection() as conn:
            self._kind(conn, profile_id)
            ud.forget(conn, profile_id)
        return {"forgotten": True}

    # ------------------------------------------------------------------ the first page's feedback buttons
    def feedback(self, request, movie_id, verdict):
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO feedback (request, movie_id, verdict) VALUES (%s, %s, %s)", (request, int(movie_id), verdict))
            conn.commit()
