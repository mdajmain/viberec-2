"""Step 11: the Torch button. Why was this movie recommended?

Three layers, each grounded in the actual score:
1. SHAP: how much each ranker feature pushed this movie up or down for this user (LightGBM's exact TreeSHAP,
   `pred_contrib=True`, so no extra library).
2. Evidence behind the features:
   - taste match (ALS): the user's liked movies that contribute most to this movie's ALS score. The ALS score splits
     exactly into one part per liked movie: score = sum_j alpha * y_i' W^-1 y_j, the same W as cf.fold_in_user;
   - vibe match: the vibes that contribute most to the cosine between the movie and the user's vibe taste;
   - popularity, fair score, year: the feature values themselves.
3. Wording: a fill-in-the-blank template (free) or an LLM that sees only this evidence.
"""
import re

import numpy as np
import pandas as pd

# how each vibe attribute is named in an explanation
VIBE_WORDS = {
    "dark_bleak": "dark", "heartwarming": "heartwarming", "tense": "tense", "melancholic": "melancholic",
    "whimsical": "whimsical", "scary": "scary", "funny": "funny", "atmospheric": "atmospheric", "slow": "slow",
    "fast": "fast-paced", "demanding": "thought-provoking", "underdog_rise": "underdog story", "redemption": "redemption",
    "tragedy": "tragedy", "revenge": "revenge", "survival": "survival", "happy_ending": "happy ending",
    "twist": "twist ending", "non_hollywood": "non-Hollywood ending", "big_city": "big-city", "small_town": "small-town",
    "suburbia": "suburbia", "workplace": "workplace", "school": "school", "period": "period piece", "future": "futuristic",
    "space": "space", "teen_coming_of_age": "coming-of-age", "family_parents": "family", "working_people": "working-class",
    "midlife": "midlife crisis", "marriage": "marriage", "dark_humor": "dark humor", "satire": "satire",
    "slapstick": "slapstick", "deadpan_witty": "witty", "violent": "violent", "romantic": "romantic", "steamy": "steamy",
    "christmas": "Christmas", "visually_striking": "visually stunning",
}


def display_title(title):
    """MovieLens 'Dark Knight, The (2008)' -> 'The Dark Knight (2008)'."""
    m = re.match(r"^(.*), (The|A|An)( \(\d{4}\))?$", title)
    return f"{m.group(2)} {m.group(1)}{m.group(3) or ''}" if m else title


def clean_text(text):
    """Remove markdown marks and line breaks from an LLM sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*_#`]+", "", text)).strip()


def shap_frame(booster, X, features):
    """Per-row SHAP contributions (one column per feature, plus 'base'); each row sums to the model's score."""
    contrib = booster.predict(np.asarray(X), pred_contrib=True)
    return pd.DataFrame(contrib, columns=list(features) + ["base"])


def top_drivers(shap_row, k=2, skip=("base",)):
    """The k features that pushed this movie up the most (positive contributions only), largest first."""
    s = shap_row.drop(labels=[c for c in skip if c in shap_row.index])
    s = s[s > 0].sort_values(ascending=False)
    return list(s.index[:k])


def liked_contributions(item_factors, yty, liked_cols, target_col, alpha, reg):
    """Exact split of the ALS score of target_col into one part per liked movie (same W as cf.fold_in_user):
    part_j = alpha * y_target' W^-1 y_j, W = Y'Y + (alpha - 1) Y_L'Y_L + reg I. The parts sum to the fold-in score."""
    yl = item_factors[np.asarray(liked_cols)]
    w = yty + (alpha - 1) * yl.T @ yl + reg * np.eye(item_factors.shape[1])
    v = np.linalg.solve(w, item_factors[target_col])
    return alpha * yl @ v


def vibe_contributions(movie_z, taste_z):
    """Exact split of vibe match (cosine between the movie's and the user's standardised vibe vectors) into one part
    per vibe. Parts sum to the cosine. Both inputs: Series over the same vibe attributes."""
    m = movie_z / (np.linalg.norm(movie_z.to_numpy()) or 1)
    t = taste_z / (np.linalg.norm(taste_z.to_numpy()) or 1)
    return m * t


def top_positive(series, k=3):
    s = series[series > 0].sort_values(ascending=False)
    return list(s.index[:k])


# ---------------------------------------------------------------- wording
def driver_phrase(driver, ev):
    """One plain phrase per driver, filled only from the evidence dict ev."""
    if driver in ("als_z", "blend_rank") and ev.get("because_liked"):
        return f"people who liked {' and '.join(ev['because_liked'])} also liked it"
    if driver == "vibe_z" and ev.get("shared_vibes"):
        return f"it has the {', '.join(ev['shared_vibes'])} feel of the movies you like"
    if driver == "log_recent":
        return f"it is popular right now ({ev['recent_ratings']:,} ratings in the last 18 months)"
    if driver == "shrunk":
        return f"it is well rated ({ev['fair_score']:.1f}/5)"
    if driver == "year_gap":
        return f"it is from {ev['year']}, close to the years of the movies you like"
    if driver == "log_count":
        return f"it is widely seen ({ev['all_ratings']:,} ratings)"
    if driver == "blend_rank":
        return "it is near the top of your taste and vibe match"
    return None


def template_sentence(drivers, ev):
    """Free baseline: 'Recommended because A, and B.' from the top drivers."""
    parts = [p for p in (driver_phrase(d, ev) for d in drivers) if p]
    if not parts:
        return "Recommended as a close match to your taste."
    return "Recommended because " + ", and ".join(parts) + "."


def _name_year(title):
    m = re.match(r"^(.*?)(?:\s*\((\d{4})\))?\s*$", title)
    return m.group(1).strip(), m.group(2)


def _names(title):
    """The title's name without the year; MovieLens 'Name, The' also as 'The Name'."""
    name, _ = _name_year(title)
    art = re.match(r"^(.*), (The|A|An)$", name)
    return {f"{art.group(2)} {art.group(1)}", art.group(1)} if art else {name}


def title_forms(title):
    """Ways a title may be written: with the year, and without it when the name is distinctive
    (2+ words or 8+ letters; 'Up' or 'Her' only count with their year)."""
    _, year = _name_year(title)
    forms = set()
    for n in _names(title):
        if year:
            forms.add(f"{n} ({year})")
        if len(n.split()) >= 2 or len(n) >= 8:
            forms.add(n)
    return forms or {title}


def _norm_text(s):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' -]", " ", s.lower())).strip() + " "


GENRE_WORDS = ["action", "adventure", "animation", "animated", "children's", "comedy", "crime", "documentary", "drama",
               "fantasy", "film-noir", "noir", "horror", "musical", "mystery", "romance", "rom-com", "sci-fi",
               "science fiction", "thriller", "war", "western"]


GENRE_SYNONYMS = {"animation": ["animated"], "romance": ["romantic comedy", "rom-com"], "sci-fi": ["science fiction"]}


def _cased(s):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9' .!?-]", " ", s)).strip() + " "


def grounding_problems(text, allowed_titles, allowed_vibes, catalog_titles, vibe_words=VIBE_WORDS, genre_words=GENRE_WORDS,
                       allowed_genres=(), evidence_text=""):
    """Things the text mentions that were not in the evidence: catalog titles, vibe words and genre words.

    - Titles count only when written as titles: case-sensitive, and a one-word title without its year not at the
      start of a sentence ("Following a gangster..." is not the film Following).
    - Allowed titles (and their year-less names) are cut out first, so words inside them ('dark' in The Dark Knight)
      do not count; a catalog title whose name is part of an allowed title's name is not flagged.
    - Vibe and genre words that appear in evidence_text (e.g. the premise) count as evidence."""
    cased = _cased(text)
    for f in sorted({f for t in allowed_titles for f in title_forms(t) | _names(t)}, key=len, reverse=True):
        cased = re.sub(re.escape(_cased(f).strip()), " | ", cased, flags=re.I)
    low = _norm_text(cased)
    allowed_names = {_norm_text(n) for t in allowed_titles for n in _names(t)}
    ev_low = _norm_text(evidence_text)

    def written_as_title(form):
        f = _cased(form).strip()
        if not f:
            return False
        if len(f.split()) == 1 and not re.search(r"\(\d{4}\)", form):
            return re.search(r"(?<![.!?|]\s)(?<!^\s)\b" + re.escape(f) + r"\b", cased[1:]) is not None and \
                re.search(r"[a-z,;:]\s" + re.escape(f) + r"\b", cased) is not None
        return re.search(r"(?<![A-Za-z])" + re.escape(f) + r"(?![A-Za-z])", cased) is not None

    def says(phrase):
        p = _norm_text(phrase)
        return p.strip() != "" and p in low
    problems = []
    for t in catalog_titles:
        if t in allowed_titles or any(_norm_text(n) in a for n in _names(t) for a in allowed_names):
            continue
        if any(written_as_title(f) for f in title_forms(t)):
            problems.append(f"title: {t}")
    allowed_v = {vibe_words.get(v, v).lower() for v in allowed_vibes}
    problems += [f"vibe: {w}" for w in set(vibe_words.values())
                 if w.lower() not in allowed_v and says(w) and _norm_text(w) not in ev_low]
    allowed_g = {g.lower() for g in allowed_genres}
    allowed_g |= {syn for g in allowed_g for syn in GENRE_SYNONYMS.get(g, [])}
    if {"romance", "comedy"} <= allowed_g:
        allowed_g |= {"rom-com", "romantic comedy"}
    plural = lambda g: g[:-1] + "ies" if g.endswith("y") else g + "s"
    problems += [f"genre: {g}" for g in genre_words
                 if g not in allowed_g and (says(g) or says(plural(g))) and _norm_text(g) not in ev_low]
    return problems


LLM_SYSTEM = """You write the one-line "why" shown next to a movie recommendation.
Use ONLY the evidence given. The reasons are listed most important first; keep that order.
Write one or two short, friendly sentences (at most 40 words) addressed to the viewer as "you".
Plain text only: no bold, italics, headings or line breaks. Do not start with the movie's title.
Do not mention any movie, actor, plot detail, genre or feeling that is not in the evidence. Numbers are optional.
Do not describe the movie itself (no genre, no adjectives about its story, style or quality): only say why it is
recommended, from the reasons given."""


def evidence_for_llm(movie_title, drivers, ev):
    """The only facts the LLM sees: the movie and its top drivers with their evidence."""
    reasons = []
    for d in drivers:
        if d in ("als_z", "blend_rank") and ev.get("because_liked"):
            reasons.append({"reason": "people with your taste liked it", "you_liked": ev["because_liked"]})
        elif d == "vibe_z" and ev.get("shared_vibes"):
            reasons.append({"reason": "it shares the feel of movies you like", "shared_feel": ev["shared_vibes"]})
        elif d == "log_recent":
            reasons.append({"reason": "popular right now", "ratings_in_last_18_months": ev["recent_ratings"]})
        elif d == "shrunk":
            reasons.append({"reason": "well rated", "average_out_of_5": round(ev["fair_score"], 1)})
        elif d == "year_gap":
            reasons.append({"reason": "from the years you watch most", "year": ev["year"]})
        elif d == "log_count":
            reasons.append({"reason": "widely seen", "total_ratings": ev["all_ratings"]})
        elif d == "blend_rank":
            reasons.append({"reason": "near the top of your taste and vibe match"})
    return {"movie": movie_title, "reasons": reasons}


def llm_wording(client, guard, model_id, evidence, cache_dir, max_tokens=120):
    """Claude's wording of the evidence (cached on disk by the evidence itself, so it is never paid for twice)."""
    import hashlib
    import json
    from pathlib import Path

    from .llm_budget import guarded_converse
    text = json.dumps(evidence, sort_keys=True)
    key = text + LLM_SYSTEM                     # a changed instruction is a new request, not a cache hit
    path = Path(cache_dir) / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if path.exists():
        return clean_text(json.loads(path.read_text())["text"])
    resp = guarded_converse(client, guard, "torch wording", model_id, [{"role": "user", "content": [{"text": text}]}],
                            {"maxTokens": max_tokens, "temperature": 0}, system=[{"text": LLM_SYSTEM}])
    out = clean_text("".join(b.get("text", "") for b in resp["output"]["message"]["content"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"evidence": evidence, "text": out, "usage": resp["usage"]}, indent=1))
    return out


# ================================================================ search results (the app's main flow)
# Adil's rules (PROJECT.md §9): describe the movie and connect it to the request; taste as "viewers with similar
# taste", never "because you liked"; no spoilers (no ending type, no twist); no algorithm talk.
SPOILER_VIBES = {"happy_ending", "twist", "tragedy", "non_hollywood"}
SPOILER_WORDS = ["ending", "twist", "dies", "die", "death", "killed", "tragic", "tragedy", "happily ever after"]
LIGHT_WORDS = {1: "heavy", 2: "fairly heavy", 3: "", 4: "fairly light", 5: "light"}      # Light/Heavy label, 5 = light
PACE_WORDS = {1: "slow", 2: "unhurried", 3: "", 4: "brisk", 5: "fast-paced"}             # Pacing label, 5 = fast


def part_contributions(parts, row, k=60):
    """Each fused part's share of one movie's relevance: 1 / (k + the movie's rank in that part), as in rrf."""
    out = {}
    for name, s in parts.items():
        s = np.asarray(s, float)
        out[name] = 1.0 / (k + 1 + np.sum(s > s[row]))
    return pd.Series(out).sort_values(ascending=False)


def words_in_text(phrases, text, stop=frozenset({"a", "an", "the", "and", "or", "of", "in", "on", "with", "to", "for",
                                                   "about", "like", "movie", "film", "some", "something"})):
    """Words of the request's literal phrases that appear in the movie's text."""
    have = set(re.findall(r"[a-z0-9']+", text.lower()))
    wanted = [w for p in phrases for w in re.findall(r"[a-z0-9']+", p.lower()) if w not in stop and len(w) > 2]
    return list(dict.fromkeys(w for w in wanted if w in have))


def top_feel(z_row, attrs, k=3):
    """The movie's strongest vibes (largest standardised scores), spoiler vibes left out."""
    s = pd.Series(z_row, index=attrs).drop(labels=[a for a in SPOILER_VIBES if a in attrs])
    return [VIBE_WORDS[a] for a in s[s > 0].sort_values(ascending=False).index[:k]]


def premise_text(overview, max_words=60):
    """A short premise from a TMDB overview: whole sentences up to about max_words words."""
    if not isinstance(overview, str) or not overview.strip():
        return ""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", overview.strip()):
        if out and len(" ".join(out + [sent]).split()) > max_words:
            break
        out.append(sent)
    return " ".join(out)


def search_evidence(engine, parsed, request, movie_id, parts, sub, three=None, sample_titles=None, premise=""):
    """Everything the "why" may say about one search result, reasons ordered by their real share of the score."""
    row = int(engine.pos[movie_id])
    contrib = part_contributions(parts, row)
    attrs, z = engine.attrs, engine.z[row]
    genres = [g for g in engine.genres[row] if g and g != "(no genres listed)"]
    ev = {"request": request, "movie": display_title(engine.tiers.at[movie_id, "title"]),
          "year": int(engine.year[row]) if np.isfinite(engine.year[row]) else None, "genres": genres,
          "feel": top_feel(z, attrs), "premise": premise, "reasons": []}
    if three is not None and movie_id in three.index:
        lh, pace = three.at[movie_id, "light_heavy"], three.at[movie_id, "pacing"]
        if pd.notna(lh) and LIGHT_WORDS[int(round(lh))]:
            ev["feel"] = [LIGHT_WORDS[int(round(lh))]] + ev["feel"]
        if pd.notna(pace) and PACE_WORDS[int(round(pace))] and PACE_WORDS[int(round(pace))] not in ev["feel"]:
            ev["feel"].append(PACE_WORDS[int(round(pace))])
    for name in contrib.index:
        if name == "vibe":
            prof = pd.Series([(a in parsed["vibe_up"]) - (a in parsed["vibe_down"]) for a in attrs], index=attrs, dtype=float)
            part = prof * pd.Series(z, index=attrs)
            more = [VIBE_WORDS[a] for a in part[(part > 0) & (prof > 0)].sort_values(ascending=False).index if a not in SPOILER_VIBES][:3]
            less = [VIBE_WORDS[a] for a in part[(part > 0) & (prof < 0)].sort_values(ascending=False).index if a not in SPOILER_VIBES][:2]
            if more or less:
                ev["reasons"].append({"reason": "matches the feel you asked for", "has": more, "is_not": less})
        elif name == "keywords":
            found = words_in_text(parsed["literal_phrases"], engine.texts[row])
            if found:
                ev["reasons"].append({"reason": "its story has what you described", "words": found})
        elif name.startswith("ref:"):
            ref = int(name[4:])
            r = int(engine.pos[ref])
            pieces = part_contributions(sub[name], row)
            shared = vibe_contributions(pd.Series(z, index=attrs), pd.Series(engine.z[r], index=attrs))
            shared_words = [VIBE_WORDS[a] for a in top_positive(shared.drop(labels=list(SPOILER_VIBES & set(attrs))), 3)]
            for piece in pieces.index:
                if piece == "similar_taste":
                    ev["reasons"].append({"reason": "viewers with similar taste rate it highly"})
                elif piece == "same_feel" and shared_words:
                    ev["reasons"].append({"reason": "same feel as the movie you gave",
                                          "sample_movie": display_title(engine.tiers.at[ref, "title"]), "shared_feel": shared_words})
    recent = int(round(np.expm1(engine.recent[row])))
    ev["reasons"].append({"reason": "well known", "recent_ratings": recent})
    return ev


def search_template(ev):
    """Free baseline wording, from the evidence only."""
    kind = " ".join(x for x in [", ".join(ev["feel"][:3]), "/".join(g.lower() for g in ev["genres"][:2])] if x) or "film"
    out = [f"A {kind} film." if not kind.endswith("film") else f"A {kind}."]
    if ev.get("premise"):
        out.append(re.split(r"(?<=[.!?])\s+", ev["premise"])[0])
    for r in ev["reasons"]:
        if r["reason"] == "matches the feel you asked for":
            s = "It has the " + ", ".join(r["has"]) + " feel you asked for" if r["has"] else "It fits the feel you asked for"
            out.append(s + (", without being " + " or ".join(r["is_not"]) if r["is_not"] else "") + ".")
        elif r["reason"] == "its story has what you described":
            out.append("Its story involves " + ", ".join(r["words"]) + ".")
        elif r["reason"] == "same feel as the movie you gave":
            out.append(f"It shares the {', '.join(r['shared_feel'])} feel of {r['sample_movie']}.")
        elif r["reason"] == "viewers with similar taste rate it highly":
            out.append("Viewers with similar taste rate it highly.")
        elif r["reason"] == "it has the feel of the films you like" and r.get("shared_feel"):
            out.append(f"It has the {', '.join(r['shared_feel'])} feel of the films you like.")
    if len(out) < 3:
        out.append("It is also well known.")
    return " ".join(out)


SEARCH_SYSTEM = """You write the short "why" shown under one movie in a search result.
The viewer typed a request (a sample movie and the feel they want). Explain why THIS movie fits their request.
Use ONLY the evidence given: the movie's genres, feel words, premise, and the reasons (listed most important first).
Rules:
- Start by saying what kind of film it is, using the feel words and genres given; you may mention its setup, but only
  from the premise, and never go beyond the premise.
- Then connect it to the request, following the order of the reasons.
- Speak of taste as "viewers with similar taste"; never say "because you liked" or "your watch history".
- You may compare it to the sample movie, but only through the shared feel listed.
- No spoilers: never mention the ending, twists, deaths or how the story turns out.
- No algorithms, scores or numbers. Plain text, two sentences, at most 50 words. Do not start with the title."""


def search_llm_wording(client, guard, model_id, evidence, cache_dir, max_tokens=150):
    """Claude's wording of one search result's evidence (cached by evidence + instructions)."""
    import hashlib
    import json
    from pathlib import Path

    from .llm_budget import guarded_converse
    text = json.dumps(evidence, sort_keys=True)
    path = Path(cache_dir) / f"{hashlib.sha1((text + SEARCH_SYSTEM).encode()).hexdigest()[:16]}.json"
    if path.exists():
        return clean_text(json.loads(path.read_text())["text"])
    resp = guarded_converse(client, guard, "search why", model_id, [{"role": "user", "content": [{"text": text}]}],
                            {"maxTokens": max_tokens, "temperature": 0}, system=[{"text": SEARCH_SYSTEM}])
    out = clean_text("".join(b.get("text", "") for b in resp["output"]["message"]["content"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"evidence": evidence, "text": out, "usage": resp["usage"]}, indent=1))
    return out


def spoiler_problems(text, words=SPOILER_WORDS, premise=""):
    """Ending words in the text. A word the premise itself uses ("death row") is the setup, not the ending."""
    low, setup = _norm_text(text), _norm_text(premise)
    return [f"spoiler: {w}" for w in words if _norm_text(w) in low and _norm_text(w) not in setup]
