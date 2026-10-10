"""IMDb ID -> Wikidata item -> English Wikipedia article, fetched in resumable batches (Step 5).

Every batch is saved to its own JSON file as soon as it arrives (written to a temp name, then
renamed), so an interrupted run resumes at the first missing batch and nothing is fetched twice.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "VibeRec/2.0 (personal movie-recommender research project; python-requests)"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _write_json_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)                       # a file either exists complete or not at all


def _check_manifest(out_dir, ids, batch_size):
    """Batches only line up if the id list and batch size are unchanged since the first run."""
    manifest = {"batch_size": batch_size, "n_ids": len(ids),
                "ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest()}
    path = out_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise ValueError(f"{out_dir} holds batches for a different id list or batch size; "
                             "use a new folder instead of mixing them")
    else:
        _write_json_atomic(path, manifest)


def fetch_in_batches(ids, out_dir, fetch, batch_size=300, pause=1.0, retries=5, backoff=5.0, max_batches=None):
    """Call fetch(batch) for each batch of ids not already saved in out_dir. Returns (fetched, skipped).

    max_batches stops after that many newly fetched batches (for a trial run); a rerun continues.

    fetch returns a JSON-serialisable list of rows. A failed batch is retried with growing waits;
    if it keeps failing the error is raised, and a rerun continues from that batch.
    """
    ids = sorted(set(ids))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_manifest(out_dir, ids, batch_size)
    fetched = skipped = 0
    for start in range(0, len(ids), batch_size):
        path = out_dir / f"batch_{start // batch_size:05d}.json"
        if path.exists():
            skipped += 1
            continue
        if max_batches is not None and fetched >= max_batches:
            break
        batch = ids[start:start + batch_size]
        for attempt in range(retries):
            try:
                rows = fetch(batch)
                break
            except Exception as err:                        # network error, timeout, 429/5xx
                if attempt == retries - 1:
                    raise RuntimeError(f"{path.name} failed {retries} times; rerun to resume from it") from err
                wait = getattr(err, "retry_after", None) or backoff * 2 ** attempt
                time.sleep(wait)
        _write_json_atomic(path, {"ids": batch, "rows": rows})
        fetched += 1
        time.sleep(pause)
    return fetched, skipped


class RetryableHTTPError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.retry_after = retry_after


def sparql_imdb_to_enwiki(imdb_ids, session=None, timeout=60):
    """One Wikidata query: for each IMDb id, the item(s) carrying it and their English Wikipedia article."""
    session = session or requests.Session()
    values = " ".join(f'"{i}"' for i in imdb_ids)
    query = f"""
    SELECT ?imdb ?item ?article WHERE {{
      VALUES ?imdb {{ {values} }}
      ?item wdt:P345 ?imdb .
      OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}
    }}"""
    resp = session.post(SPARQL_URL, data={"query": query, "format": "json"},
                        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
                        timeout=timeout)
    if resp.status_code in RETRY_STATUS:
        retry_after = resp.headers.get("Retry-After")
        raise RetryableHTTPError(resp.status_code, float(retry_after) if retry_after and retry_after.isdigit() else None)
    resp.raise_for_status()
    return [{k: b[k]["value"] for k in ("imdb", "item", "article") if k in b}
            for b in resp.json()["results"]["bindings"]]


def load_batches(out_dir):
    """All saved batches as one table: imdb, item, article (one row per match; ids with no match absent)."""
    rows = []
    for path in sorted(Path(out_dir).glob("batch_*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    return pd.DataFrame(rows, columns=["imdb", "item", "article"])
