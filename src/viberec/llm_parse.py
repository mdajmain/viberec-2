"""Step 9: parse a search request with an LLM (Claude Haiku on Bedrock), PROJECT.md §8 step 1.

The model fills a fixed form (a forced tool call): reference movies, required genres, vibe phrases, literal
phrases, and which of the 41 vibe attributes the request pushes up or down. Every call goes through the budget
guard, and every answer is cached on disk by request text, so the same request is never paid for twice.
"""
import hashlib
import json
from pathlib import Path

from .llm_budget import guarded_converse

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

SYSTEM = """You turn a movie request into structured search fields for a movie recommender. Be literal and faithful:
- reference_movies: every movie the user names as a reference, with its usual English title and year if you know it,
  whether or not they say "like" (e.g. "Hangover, the more similar the better" names The Hangover).
- required_genres: only genres the user clearly requires. Not genres they negate or allow ("less action",
  "animation is fine"). A rom-com requires Romance and Comedy.
- vibe_phrases: short phrases about how it should feel (tone, mood, pace, emotion), in the user's words.
- literal_phrases: concrete topics, settings, places, people, eras, events.
- vibe_up / vibe_down: which of the listed vibe attributes the request clearly asks for more or less of.
Leave a field empty rather than guess."""


def tool_spec(attributes):
    schema = {
        "type": "object",
        "properties": {
            "reference_movies": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "year": {"type": "integer"}}, "required": ["title"]}},
            "required_genres": {"type": "array", "items": {"type": "string", "enum": GENRES}},
            "vibe_phrases": {"type": "array", "items": {"type": "string"}},
            "literal_phrases": {"type": "array", "items": {"type": "string"}},
            "vibe_up": {"type": "array", "items": {"type": "string", "enum": list(attributes)}},
            "vibe_down": {"type": "array", "items": {"type": "string", "enum": list(attributes)}},
        },
        "required": ["reference_movies", "required_genres", "vibe_phrases", "literal_phrases", "vibe_up", "vibe_down"],
    }
    return {"tools": [{"toolSpec": {"name": "record_search", "description": "Record the parsed movie request.",
                                    "inputSchema": {"json": schema}}}],
            "toolChoice": {"tool": {"name": "record_search"}}}


def describe_attributes(attribute_tags):
    """System-prompt list of vibe attributes with the Genome tags that define them."""
    return "\n".join(f"- {a}: {', '.join(tags)}" for a, tags in attribute_tags.items())


def parse(client, guard, request, attribute_tags, cache_dir, max_tokens=600):
    """Parsed fields for one request (cached by request text)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(request.strip().lower().encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.json"
    if path.exists():
        return clean(json.loads(path.read_text())["parsed"])
    system = SYSTEM + "\n\nVibe attributes:\n" + describe_attributes(attribute_tags)
    resp = guarded_converse(client, guard, "search parse", MODEL_ID,
                            [{"role": "user", "content": [{"text": request}]}], {"maxTokens": max_tokens, "temperature": 0},
                            system=[{"text": system}], toolConfig=tool_spec(attribute_tags.keys()))
    blocks = [b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b]
    parsed = blocks[0] if blocks else {}
    path.write_text(json.dumps({"request": request, "parsed": parsed, "usage": resp["usage"]}, indent=1))
    return clean(parsed)


def clean(parsed):
    """Bring a parse into the expected shape. The schema is a request, not a guarantee: the model sometimes returns a
    reference as a plain string, or a list field as one string; those are converted, anything else is dropped."""
    def as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                return [v] if v.strip() else []
        return v if isinstance(v, list) else [v]
    refs = []
    for r in as_list(parsed.get("reference_movies")):
        if isinstance(r, str) and r.strip():
            refs.append({"title": r.strip()})
        elif isinstance(r, dict) and r.get("title"):
            refs.append({"title": str(r["title"]), **({"year": r["year"]} if isinstance(r.get("year"), int) else {})})
    out = {"reference_movies": refs}
    for k in ("required_genres", "vibe_phrases", "literal_phrases", "vibe_up", "vibe_down"):
        out[k] = [str(x) for x in as_list(parsed.get(k)) if isinstance(x, (str, int)) and str(x).strip()]
    out["required_genres"] = [g for g in out["required_genres"] if g in GENRES]
    return out
