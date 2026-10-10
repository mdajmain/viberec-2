"""Adil's hand labels (data/labels/vibe_labels_v1.xlsx): read them and match each row to a MovieLens movieId.

Matching is by normalised title + year. MovieLens writes "Proposal, The (2009)" and adds alternate titles
in trailing brackets ("Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008)"), so each
MovieLens movie gets a key for its main title and one per alternate title. If several movies share a key,
the one whose punctuation matches the label exactly wins ("Mother!" vs "Mother").
"""
import re
import unicodedata

import pandas as pd

COLUMNS = {"v1": ["row", "title", "year", "easy_watch", "comfort", "notes"],
           "v2": ["row", "title", "year", "light_heavy", "ending", "pacing", "notes"]}


def read_labels(path, version="v2"):
    labels = pd.read_excel(path, sheet_name="Labels")
    labels.columns = COLUMNS[version]
    return labels.dropna(subset=["title"]).reset_index(drop=True)


QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


def _ascii_lower(text):
    """Lowercase ASCII; curly quotes become straight first (We\u2019re -> We're), so they are not dropped."""
    return unicodedata.normalize("NFKD", str(text).translate(QUOTES)).encode("ascii", "ignore").decode().lower().strip()


def _move_article(title):
    m = re.match(r"^(.*), (the|a|an)$", title)
    return f"{m.group(2)} {m.group(1)}" if m else title


def norm(title):
    """Lowercase, ASCII, leading article restored, punctuation removed."""
    return re.sub(r"[^a-z0-9]+", " ", _move_article(_ascii_lower(title))).strip()


def movielens_keys(ml_title):
    """(year, [exact titles]) for a MovieLens title: the main title and any alternate titles in trailing brackets."""
    t = ml_title.strip()
    year = None
    m = re.search(r"\((\d{4})\)\s*$", t)
    if m:
        year, t = int(m.group(1)), t[:m.start()].strip()
    titles = []
    while True:                                           # peel trailing "(alternate title)" groups
        m = re.search(r"\s*\(([^()]*)\)\s*$", t)
        if not m or not t[:m.start()].strip():
            break
        titles.append(m.group(1))
        t = t[:m.start()].strip()
    return year, [t] + titles


def match(labels, ml_movies):
    """Add movieId (and the MovieLens title) to each label row; NaN where nothing matches."""
    index = {}
    for mid, title in zip(ml_movies["movieId"], ml_movies["title"]):
        year, titles = movielens_keys(title)
        for t in titles:
            index.setdefault((norm(t), year), []).append((mid, _move_article(_ascii_lower(t)), title))
    ids, ml_titles = [], []
    for title, year in zip(labels["title"], labels["year"]):
        hits = index.get((norm(title), int(year)), [])
        exact = [h for h in hits if h[1] == _ascii_lower(title)]
        pick = (exact or hits)[:1]
        ids.append(pick[0][0] if pick else None)
        ml_titles.append(pick[0][2] if pick else None)
    return labels.assign(movieId=pd.array(ids, dtype="Int64"), ml_title=ml_titles)


# v2 as Adil finally filled it (sheet "Movie Ratings", checked by him on 2026-10-08):
# Light vs heavy runs 1 = light ... 5 = heavy, and Ending has 7 values.
ENDING_TO_FOUR = {"happy": "happy", "hopeful": "happy", "bittersweet": "bittersweet",
                  "sad": "sad", "tragic": "sad", "bleak": "sad", "ambiguous": "open"}


def read_labels_v2(path, title_fixes=None):
    """v2 labels on the LLM's scales: light_heavy flipped to 1 = heavy ... 5 = light (6 - value),
    ending mapped to happy / bittersweet / sad / open (original kept as ending_detail)."""
    raw = pd.read_excel(path, sheet_name="Movie Ratings")
    raw.columns = ["row", "title", "year", "heavy_light_raw", "ending_detail", "pacing", "notes"]
    raw = raw.dropna(subset=["title"]).reset_index(drop=True)
    for old, new in (title_fixes or {}).items():
        raw.loc[raw["title"] == old, "title"] = new
    raw["ending_detail"] = raw["ending_detail"].str.strip().str.lower()
    unknown = set(raw["ending_detail"].dropna()) - set(ENDING_TO_FOUR)
    if unknown:
        raise ValueError(f"unknown ending labels: {sorted(unknown)}")
    return raw.assign(light_heavy=6 - raw["heavy_light_raw"], ending=raw["ending_detail"].map(ENDING_TO_FOUR))
