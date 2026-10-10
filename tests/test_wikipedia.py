import pytest

from src.viberec import wikipedia as wp

ARTICLE = """Heat is a 1995 crime film.

== Plot ==
A crew of thieves plans a heist.

=== Aftermath ===
The detective chases them.

== Cast ==
Al Pacino"""


def test_plot_section_includes_subsections_and_stops_at_next_section():
    assert wp.plot_section(ARTICLE) == "A crew of thieves plans a heist.\n\n\nThe detective chases them."


def test_plot_section_other_heading_and_missing():
    assert wp.plot_section("Intro\n\n== Synopsis ==\nThey fall in love.\n\n== Reception ==\nGood.") == "They fall in love."
    assert wp.plot_section("Intro only.\n\n== Cast ==\nSomeone") is None
    assert wp.plot_section(None) is None


def test_title_from_url():
    assert wp.title_from_url("https://en.wikipedia.org/wiki/Heat_(1995_film)") == "Heat (1995 film)"
    assert wp.title_from_url("https://en.wikipedia.org/wiki/Am%C3%A9lie") == "Amélie"


def test_fetch_each_resumes(tmp_path):
    items = {1: "A", 2: "B", 3: "C"}

    def fails_on_b(title):
        if title == "B":
            raise ConnectionError("down")
        return {"title": title, "text": title.lower()}

    with pytest.raises(RuntimeError):
        wp.fetch_each(items, tmp_path, fails_on_b, pause=0, retries=2, backoff=0)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["1.json"]
    assert wp.fetch_each(items, tmp_path, lambda t: {"title": t, "text": t}, pause=0) == (2, 1)
    assert {r["key"] for r in wp.load_extracts(tmp_path)} == {1, 2, 3}


def test_plot_from_wikitext_cleans_markup():
    wikitext = ("Lead.\n\n==Plot==\nA [[thief|robber]] named {{nowrap|Neil}} robs a bank.<ref>{{cite web|title=x}}</ref> "
                "He escapes.<ref name=a/>\n\n===Ending===\n''Everyone'' lives.\n\n==Cast==\n* Al")
    assert wp.plot_from_wikitext(wikitext) == "A robber named Neil robs a bank. He escapes.\n\nEveryone lives."
    assert wp.plot_from_wikitext("Lead only.\n==Reception==\nGood.") is None


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class FakeSession:
    """First call: normalised + redirected title, one page continued; second call: the rest."""
    def __init__(self):
        self.calls = 0

    def get(self, url, params, headers, timeout):
        self.calls += 1
        if self.calls == 1:
            return FakeResponse({"continue": {"rvcontinue": "x"}, "query": {
                "normalized": [{"from": "heat_(1995_film)", "to": "Heat (1995 film)"}],
                "redirects": [{"from": "Heat (1995 film)", "to": "Heat (film)"}],
                "pages": [{"title": "Heat (film)", "revisions": [{"slots": {"main": {"content": "HEAT"}}}]},
                          {"title": "Jumanji"}]}})
        return FakeResponse({"query": {"pages": [{"title": "Jumanji", "revisions": [{"slots": {"main": {"content": "JUMANJI"}}}]},
                                                  {"title": "Nope", "missing": True}]}})


def test_fetch_wikitext_batch_redirects_and_continue():
    rows = wp.fetch_wikitext_batch(["heat_(1995_film)", "Jumanji", "Nope"], session=FakeSession())
    assert rows == [{"request": "heat_(1995_film)", "title": "Heat (film)", "wikitext": "HEAT"},
                    {"request": "Jumanji", "title": "Jumanji", "wikitext": "JUMANJI"},
                    {"request": "Nope", "title": "Nope", "wikitext": None}]


def test_reception_level_2_and_level_3():
    top = "Lead.\n\n==Plot==\nStuff.\n\n==Reception==\nA ''bittersweet'' comedy.<ref>x</ref>\n\n==Legacy==\nLater."
    assert wp.reception_from_wikitext(top) == "A bittersweet comedy."
    nested = ("Lead.\n\n==Release==\nOpened in May.\n\n===Box office===\n$5m.\n\n===Critical response===\n"
              "Critics called it [[heartwarming]].\n\n==Legacy==\nLater.")
    assert wp.reception_from_wikitext(nested) == "Critics called it heartwarming."
    assert wp.reception_from_wikitext("Lead.\n\n==Plot==\nStuff.") is None


def test_fetch_each_accepts_numpy_keys(tmp_path):
    import numpy as np
    assert wp.fetch_each({np.int64(7): np.int64(7)}, tmp_path, lambda x: {"value": x}, pause=0) == (1, 0)
    assert (tmp_path / "7.json").exists()
