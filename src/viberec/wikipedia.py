"""Fetch English Wikipedia article text one movie at a time, resumably, and pull out the plot (Step 5).

The whole plain-text article is cached (one JSON file per movie), so the plot extraction can be
changed later without fetching again. Files are written to a temp name and renamed, so a crash
never leaves a half-written file, and a rerun only fetches movies that have no file yet.
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests

from .wikidata import RETRY_STATUS, USER_AGENT, RetryableHTTPError, _write_json_atomic

API_URL = "https://en.wikipedia.org/w/api.php"
PLOT_HEADINGS = re.compile(r"^(plot|plot summary|plot synopsis|synopsis|summary|premise|story|storyline)$", re.I)
HEADING = re.compile(r"^(={2,})\s*(.*?)\s*\1\s*$", re.M)


def title_from_url(url):
    return unquote(url.rsplit("/wiki/", 1)[1]).replace("_", " ")


def fetch_extract(title, session=None, timeout=30):
    """Plain text of one article (redirects followed). Returns {"title", "text"}; text is None if missing."""
    session = session or requests.Session()
    resp = session.get(API_URL, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                        "titles": title, "format": "json", "formatversion": 2},
                       headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if resp.status_code in RETRY_STATUS:
        retry_after = resp.headers.get("Retry-After")
        raise RetryableHTTPError(resp.status_code, float(retry_after) if retry_after and retry_after.isdigit() else None)
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][0]
    return {"title": page.get("title", title), "text": page.get("extract") if not page.get("missing") else None}


def fetch_each(items, out_dir, fetch, pause=0.5, retries=5, backoff=5.0):
    """items: {key: argument}. Calls fetch(argument) for every key without a saved file. Returns (fetched, skipped)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = skipped = 0
    for key, arg in items.items():
        key = key.item() if hasattr(key, "item") else key        # NumPy numbers -> plain Python (JSON-safe)
        arg = arg.item() if hasattr(arg, "item") else arg
        path = out_dir / f"{key}.json"
        if path.exists():
            skipped += 1
            continue
        for attempt in range(retries):
            try:
                result = fetch(arg)
                break
            except Exception as err:
                if attempt == retries - 1:
                    raise RuntimeError(f"{key} failed {retries} times; rerun to resume from it") from err
                time.sleep(getattr(err, "retry_after", None) or backoff * 2 ** attempt)
        _write_json_atomic(path, {"key": key, "request": arg, **result})
        fetched += 1
        time.sleep(pause)
    return fetched, skipped


RECEPTION_HEADINGS = re.compile(r"^(reception|critical reception|critical response|critical reaction|"
                                r"release and reception|reception and legacy|critical reception and legacy)$", re.I)


def section(text, headings, max_level=2):
    """Text of the first section (heading level 2..max_level) whose name matches `headings`, or None.

    The section runs to the next heading of the same or a higher level; its sub-headings are dropped.
    """
    if not text:
        return None
    heads = [(m.start(), m.end(), len(m.group(1)), m.group(2)) for m in HEADING.finditer(text)]
    for i, (_, end, level, name) in enumerate(heads):
        if 2 <= level <= max_level and headings.match(name):
            stop = next((s for s, _, lvl, _ in heads[i + 1:] if lvl <= level), len(text))
            body = HEADING.sub("", text[end:stop]).strip()
            return body or None
    return None


def plot_section(text):
    """Text of the first level-2 section whose heading looks like a plot heading, or None."""
    return section(text, PLOT_HEADINGS)


def load_extracts(out_dir):
    """All saved articles: key, title, text."""
    rows = []
    for path in Path(out_dir).glob("*.json"):
        d = json.loads(path.read_text())
        rows.append({"key": d["key"], "title": d.get("title"), "text": d.get("text")})
    return rows


# --- batched wikitext (50 pages per request), used for the full fetch -------------------------

TEXT_TEMPLATES = {"nowrap", "lang", "transl", "small", "nobr", "ill", "noitalic", "visible anchor"}


def fetch_wikitext_batch(titles, session=None, timeout=60):
    """Raw wikitext for up to 50 titles in one request (redirects followed, continuation handled).

    Returns one row per requested title: {"request", "title", "wikitext"} (wikitext None if missing).
    """
    session = session or requests.Session()
    params = {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
              "titles": "|".join(titles), "redirects": 1, "format": "json", "formatversion": 2}
    pages, renames = {}, {}
    while True:
        resp = session.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code in RETRY_STATUS:
            retry_after = resp.headers.get("Retry-After")
            raise RetryableHTTPError(resp.status_code, float(retry_after) if retry_after and retry_after.isdigit() else None)
        resp.raise_for_status()
        data = resp.json()
        query = data.get("query", {})
        for step in query.get("normalized", []) + query.get("redirects", []):
            renames[step["from"]] = step["to"]
        for page in query.get("pages", []):
            revs = page.get("revisions")
            if revs:                                              # a continued page arrives with its content once
                pages[page["title"]] = revs[0]["slots"]["main"]["content"]
            else:
                pages.setdefault(page["title"], None)
        if "continue" not in data:
            break
        params = {**params, **data["continue"]}
    rows = []
    for title in titles:
        final = title
        for _ in range(5):                                        # follow normalisation, then redirect
            final = renames.get(final, final)
        rows.append({"request": title, "title": final, "wikitext": pages.get(final)})
    return rows


def plot_from_wikitext(wikitext):
    """Plain text of the plot section of an article's wikitext, or None."""
    return clean_wikitext(plot_section(wikitext))


def reception_from_wikitext(wikitext):
    """Plain text of the critical-reception section, or None. It is often a level-3 section under "Release"."""
    return clean_wikitext(section(wikitext, RECEPTION_HEADINGS, max_level=3))


def clean_wikitext(text):
    """Wikitext -> plain prose: links, citations and templates removed, small inline templates kept as text."""
    import mwparserfromhell
    if not text:
        return None
    code = mwparserfromhell.parse(text)
    for tag in code.filter_tags(recursive=False):
        if str(tag.tag).strip().lower() == "ref":                  # citations, including plain-text ones
            code.remove(tag)
    for tpl in code.filter_templates(recursive=False):
        if str(tpl.name).strip().lower() in TEXT_TEMPLATES:
            unnamed = [p for p in tpl.params if not p.showkey]
            code.replace(tpl, str(unnamed[-1].value) if unnamed else "")
    plain = code.strip_code(normalize=True, collapse=True)
    plain = re.sub(r"[ \t]+", " ", plain)
    plain = re.sub(r" +([.,;:])", r"\1", plain).strip()
    return plain or None
