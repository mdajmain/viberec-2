import pandas as pd

from src.viberec import serve


class FakeClient:
    def __init__(self, whys):
        self.whys = whys

    def converse(self, **kw):
        return {"output": {"message": {"content": [{"toolUse": {"input": {"whys": self.whys}}}]}},
                "usage": {"inputTokens": 10, "outputTokens": 10}}


class FakeGuard:
    def check(self, est): pass
    def estimate(self, *a): return 0.0
    def record(self, *a): return 0.0


def make_service(tmp_path, whys):
    s = object.__new__(serve.VibeRecService)
    s.client, s.guard, s.model_id, s.cache, s.catalog = FakeClient(whys), FakeGuard(), "m", tmp_path, ["Titanic (1997)"]
    return s


EV = {"request": "light romance", "movie": "Hitch (2005)", "year": 2005, "genres": ["Comedy", "Romance"],
      "feel": ["light", "funny", "romantic"], "premise": "A date doctor helps men woo women.",
      "reasons": [{"reason": "matches the feel you asked for", "has": ["funny"], "is_not": []},
                  {"reason": "well known", "recent_ratings": 900}]}


def test_good_why_is_kept_and_spoiler_falls_back_to_template(tmp_path):
    s = make_service(tmp_path, ["A light, funny romance about a date doctor.", "Funny, and it has a happy ending."])
    out, src = s.whys([EV, dict(EV, movie="Hitch 2 (2006)")])
    assert src == ["claude", "template"]
    assert out[0] == "A light, funny romance about a date doctor." and "ending" not in out[1]


def test_missing_why_falls_back(tmp_path):
    s = make_service(tmp_path, ["only one"])
    out, src = s.whys([EV, EV])
    assert src[1] == "template"


def test_feedback_rejects_unknown_verdict():
    s = object.__new__(serve.VibeRecService)
    try:
        s.feedback("x", 1, "love it")
        assert False
    except ValueError:
        pass


def test_mild_extra_word_is_kept_but_unknown_title_falls_back(tmp_path):
    s = make_service(tmp_path, ["A witty, light romance.", "As good as Titanic (1997)."])
    out, src = s.whys([EV, dict(EV, movie="Hitch 3 (2007)")])
    assert src == ["claude", "template"]
