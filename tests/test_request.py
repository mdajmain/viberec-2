import numpy as np
import pandas as pd

from src.viberec import request as rq

ML = pd.DataFrame({"movieId": [1, 2, 3, 4], "title": [
    "How to Lose a Guy in 10 Days (2003)", "Up (2009)", "Heat (1995)", "Heat (1986)"]})


def test_reference_found_and_one_word_titles_guarded():
    idx = rq.build_title_index(ML, {1, 2, 3, 4}, {3: 100, 4: 5})
    assert rq.find_reference("a rom-com like How to Lose a Guy in 10 Days, urban", idx)[0] == 1
    assert rq.find_reference("something to cheer me up", idx)[0] is None       # 'up' not after 'like'
    assert rq.find_reference("something like Up", idx)[0] == 2
    assert rq.find_reference("a movie like heat", idx)[0] == 3                 # most popular of two 'Heat's


def test_genres_and_phrases():
    g, rest = rq.find_genres("soft soothing rom-com urban working people")
    assert g == ["Comedy", "Romance"] and "rom" not in rest
    assert rq.phrases("I wanna watch something soft, soothing and about working people") == ["soft", "soothing", "working people"]


class FakeRouter:
    def p_vibe(self, texts):
        return np.array([1.0 if t in ("soft", "soothing") else 0.0 for t in texts])


def test_parse_request_splits_all_parts():
    idx = rq.build_title_index(ML, {1, 2, 3, 4}, {})
    out = rq.parse_request("I wanna watch a soft, soothing rom-com like How to Lose a Guy in 10 Days, based on urban life, working people",
                           idx, FakeRouter())
    assert out["reference"] == 1 and out["genres"] == ["Comedy", "Romance"]
    assert out["vibe_phrases"] == ["soft", "soothing"] and "working people" in out["literal_phrases"]


def test_feel_good_survives_filler_removal():
    assert rq.phrases("a cozy, feel-good movie") == ["cozy", "feel good"]


def test_reference_after_like_beats_an_earlier_title():
    ml = pd.DataFrame({"movieId": [1, 2, 3], "title": ["Superhero Movie (2008)", "Iron Man (2008)", "We're the Millers (2013)"]})
    idx = rq.build_title_index(ml, {1, 2, 3}, {})
    assert rq.find_reference("A superhero movie like Iron Man, fun and fast", idx)[0] == 2
    assert rq.find_reference("Something like We’re the Millers", idx)[0] == 3


def test_genres_negated_or_optional_are_not_required():
    assert rq.find_genres("like inception but less action and more emotion, a complicated romance")[0] == ["Romance"]
    assert rq.find_genres("youthful and funny. animation is fine")[0] == []
