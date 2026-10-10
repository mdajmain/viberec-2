"""TMDB movie details + keywords, one cached JSON file per movie (Step 5).

The API key is read from the project's .env file (TMDB_API_KEY) or the environment, never from code.
Fetching uses wikipedia.fetch_each, so a rerun skips movies already on disk.
"""
import os
from pathlib import Path

import requests

from .wikidata import RETRY_STATUS, RetryableHTTPError

API = "https://api.themoviedb.org/3"


def api_key(env_file=Path(__file__).resolve().parents[2] / ".env"):
    if os.environ.get("TMDB_API_KEY"):
        return os.environ["TMDB_API_KEY"]
    for line in Path(env_file).read_text().splitlines():
        if line.startswith("TMDB_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise KeyError("TMDB_API_KEY not found in the environment or .env")


def fetch_movie(tmdb_id, key=None, session=None, timeout=30):
    """Overview, tagline and keywords of one movie. Missing movies return found=False."""
    session = session or requests.Session()
    resp = session.get(f"{API}/movie/{int(tmdb_id)}", params={"api_key": key or api_key(),
                                                              "append_to_response": "keywords"}, timeout=timeout)
    if resp.status_code == 404:
        return {"found": False}
    if resp.status_code in RETRY_STATUS:
        retry_after = resp.headers.get("Retry-After")
        raise RetryableHTTPError(resp.status_code, float(retry_after) if retry_after and retry_after.isdigit() else None)
    resp.raise_for_status()
    d = resp.json()
    return {"found": True, "title": d.get("title"), "overview": d.get("overview"), "tagline": d.get("tagline"),
            "keywords": [k["name"] for k in d.get("keywords", {}).get("keywords", [])],
            "production_countries": [c["iso_3166_1"] for c in d.get("production_countries", [])]}


def fetch_poster(tmdb_id, key=None, session=None, timeout=30, language="en-US"):
    """Poster and backdrop image paths of one movie (to show as https://image.tmdb.org/t/p/w342<path>).
    language: without it TMDB may return another language's poster (Forrest Gump came back in Chinese).
    Missing movies return found=False."""
    session = session or requests.Session()
    params = {"api_key": key or api_key(), **({"language": language} if language else {})}
    resp = session.get(f"{API}/movie/{int(tmdb_id)}", params=params, timeout=timeout)
    if resp.status_code == 404:
        return {"found": False}
    if resp.status_code in RETRY_STATUS:
        retry_after = resp.headers.get("Retry-After")
        raise RetryableHTTPError(resp.status_code, float(retry_after) if retry_after and retry_after.isdigit() else None)
    resp.raise_for_status()
    d = resp.json()
    return {"found": True, "poster_path": d.get("poster_path"), "backdrop_path": d.get("backdrop_path"),
            "release_date": d.get("release_date")}
