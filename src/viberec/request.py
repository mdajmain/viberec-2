"""Step 9: split a free-text request into parts (PROJECT.md §8): reference movie, genres, vibe phrases and
literal phrases. Each part is later searched with the method that tested best for it.
"""
import re

import numpy as np

from .labels import movielens_keys, norm

# synonyms -> MovieLens genres (vocabulary, not a tuned number)
GENRE_WORDS = {
    "rom com": ["Romance", "Comedy"], "romcom": ["Romance", "Comedy"], "romantic comedy": ["Romance", "Comedy"],
    "comedy": ["Comedy"], "comedies": ["Comedy"], "drama": ["Drama"], "dramas": ["Drama"],
    "romance": ["Romance"], "romantic": ["Romance"], "horror": ["Horror"], "thriller": ["Thriller"],
    "sci fi": ["Sci-Fi"], "scifi": ["Sci-Fi"], "science fiction": ["Sci-Fi"], "action": ["Action"],
    "adventure": ["Adventure"], "animated": ["Animation"], "animation": ["Animation"], "cartoon": ["Animation"],
    "documentary": ["Documentary"], "musical": ["Musical"], "western": ["Western"], "war movie": ["War"],
    "crime": ["Crime"], "mystery": ["Mystery"], "fantasy": ["Fantasy"], "kids": ["Children"], "children": ["Children"],
}
NOT_REQUIRED_BEFORE = {"less", "no", "not", "without", "non", "fewer"}   # genre named but not required
FILLER = re.compile(r"\b(i|i'm|im|wanna|want|to|watch|a|an|the|some|something|movie|movies|film|films|show|me|"
                    r"please|looking|for|that|is|are|based|on|like|similar|kind|of|with|about|and|or|in|set)\b")
SPLIT = re.compile(r",|;|\band\b|\bwith\b|\babout\b|\bbased on\b|\bset in\b")


def build_title_index(ml_movies, allowed_ids, popularity):
    """normalised title -> movieId (most popular movie when several share a title), main and alternate titles."""
    index = {}
    for mid, title in zip(ml_movies["movieId"], ml_movies["title"]):
        if mid not in allowed_ids:
            continue
        for t in movielens_keys(title)[1]:
            key = norm(t)
            if key and (key not in index or popularity.get(mid, 0) > popularity.get(index[key], 0)):
                index[key] = mid
    return index


def find_reference(text, title_index, max_words=10):
    """The catalog title the request refers to. A title right after 'like' / 'similar to' wins (so "a superhero
    movie like Iron Man" means Iron Man, not the film called Superhero Movie); otherwise the longest title in the
    request, with one-word titles only accepted after 'like'. Returns (movieId, matched words) or (None, None)."""
    words = norm(text).split()
    best = (None, None)
    starts = [i + 1 for i, w in enumerate(words) if w == "like"] + \
             [i + 2 for i in range(len(words) - 1) if words[i:i + 2] == ["similar", "to"]]
    for i in starts:                                     # longest title starting right after the cue
        for n in range(min(max_words, len(words) - i), 0, -1):
            key = " ".join(words[i:i + n])
            if key in title_index:
                return title_index[key], key
    for n in range(min(max_words, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            key = " ".join(words[i:i + n])
            if key not in title_index:
                continue
            after_like = i > 0 and (words[i - 1] == "like" or words[i - 2:i] == ["similar", "to"])
            if n >= 2 or after_like:
                return title_index[key], key
    return best


def clean(text):
    """Lowercase ASCII with punctuation turned into spaces, except commas and semicolons (they separate phrases)."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w,;\s]", " ", t)).strip()


def remove_words(text, words):
    """Remove a word sequence (as produced by norm) from cleaned text, allowing punctuation between the words."""
    pattern = r"\b" + r"[\s,;]+".join(map(re.escape, words.split())) + r"\b"
    return re.sub(pattern, " ", text, count=1)


def find_genres(text):
    """MovieLens genres named in the request, and the request (cleaned, commas kept) with those words removed."""
    t = clean(text)
    genres = []
    for phrase in sorted(GENRE_WORDS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(phrase)}\b", t)
        if not m:
            continue
        before = t[:m.start()].split()[-1:]                # "less action", "no horror", "without romance"
        after = t[m.end():].split()[:2]                    # "animation is fine", "romance is ok"
        optional = (before and before[0] in NOT_REQUIRED_BEFORE) or after[:2] in (["is", "fine"], ["is", "ok"], ["is", "okay"])
        if not optional:
            genres += GENRE_WORDS[phrase]
        t = remove_words(t, phrase)
    return sorted(set(genres)), re.sub(r"\s+", " ", t).strip()


def phrases(text):
    """Split the remaining request into short phrases and drop filler words."""
    out = []
    for part in SPLIT.split(text.lower()):
        p = re.sub(r"\s+", " ", FILLER.sub(" ", norm(part))).strip()
        if p:
            out.append(p)
    return out


class Router:
    """Decides whether a phrase describes a feeling (vibe) or something literal (topic, name, place), from the
    phrase's embedding. Trained on labelled example phrases with logistic regression."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.model = None

    def fit(self, texts, is_vibe, balanced=True):
        """balanced=True weights the two classes equally (vibe examples are the minority), which trades some
        literal accuracy for catching more vibe phrases."""
        from sklearn.linear_model import LogisticRegressionCV
        x = self.encoder.encode(list(texts), normalize_embeddings=True)
        self.model = LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, class_weight="balanced" if balanced else None)
        self.model.fit(x, np.asarray(is_vibe, int))
        return self

    def p_vibe(self, texts):
        return self.model.predict_proba(self.encoder.encode(list(texts), normalize_embeddings=True))[:, 1]


def parse_request(text, title_index, router):
    """Split a request into {reference, genres, vibe_phrases, literal_phrases}."""
    ref, ref_words = find_reference(text, title_index)
    rest = clean(text)
    if ref_words:
        rest = remove_words(rest, ref_words)
    genres, rest = find_genres(rest)
    parts = phrases(rest)
    p = router.p_vibe(parts) if parts else np.array([])
    return {"reference": ref, "genres": genres,
            "vibe_phrases": [x for x, pv in zip(parts, p) if pv >= 0.5],
            "literal_phrases": [x for x, pv in zip(parts, p) if pv < 0.5]}
