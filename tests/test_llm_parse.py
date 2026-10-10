from src.viberec import llm_budget as lb
from src.viberec import llm_parse as lp


class FakeClient:
    def __init__(self):
        self.calls = 0

    def converse(self, **kw):
        self.calls += 1
        assert kw["toolConfig"]["toolChoice"]["tool"]["name"] == "record_search"
        return {"usage": {"inputTokens": 700, "outputTokens": 80},
                "output": {"message": {"content": [{"toolUse": {"input": {
                    "reference_movies": [{"title": "The Hangover", "year": 2009}], "required_genres": ["Comedy"],
                    "vibe_phrases": ["extremely funny"], "literal_phrases": [], "vibe_up": ["funny"], "vibe_down": []}}}]}}}


def test_parse_is_cached_and_guarded(tmp_path):
    g = lb.BudgetGuard(tmp_path / "ledger.json", cap_usd=1.0)
    client = FakeClient()
    attrs = {"funny": ["funny", "hilarious"], "tense": ["tense"]}
    p1 = lp.parse(client, g, "Hangover, the more similar the better", attrs, tmp_path / "cache")
    p2 = lp.parse(client, g, "  hangover, THE more similar the better ", attrs, tmp_path / "cache")
    assert p1 == p2 and client.calls == 1                       # second request served from cache, not paid
    assert p1["reference_movies"][0]["title"] == "The Hangover"
    assert g.spent() > 0


def test_tool_spec_restricts_vibes_and_genres():
    spec = lp.tool_spec(["funny", "tense"])["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]
    assert spec["vibe_up"]["items"]["enum"] == ["funny", "tense"]
    assert "Romance" in spec["required_genres"]["items"]["enum"]


def test_clean_accepts_plain_strings_and_drops_junk():
    raw = {"reference_movies": ["Titanic", {"title": "The Hangover", "year": 2009}, {"year": 1999}],
           "required_genres": ["Comedy", "Rom-Com"], "vibe_phrases": "make me cry", "literal_phrases": None,
           "vibe_up": ["funny"], "vibe_down": []}
    c = lp.clean(raw)
    assert c["reference_movies"] == [{"title": "Titanic"}, {"title": "The Hangover", "year": 2009}]
    assert c["required_genres"] == ["Comedy"] and c["vibe_phrases"] == ["make me cry"] and c["literal_phrases"] == []
