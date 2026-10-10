import numpy as np
import pandas as pd
import pytest

from src.viberec import genome as gn


def test_attribute_scores_mean_and_any_of():
    m = pd.DataFrame({"dark": [0.9, 0.1], "bleak": [0.7, 0.3], "london": [0.0, 0.8], "paris": [0.9, 0.1]}, index=[1, 2])
    vibe_map = {"tone": {"dark_bleak": ["dark", "bleak"]}, "setting": {"big_city": ["london", "paris"]}}
    s = gn.attribute_scores(m, vibe_map)
    assert s.loc[1, "dark_bleak"] == pytest.approx(0.8)          # synonyms: mean
    assert list(s["big_city"]) == pytest.approx([0.9, 0.8])     # alternatives: highest tag


def test_attribute_scores_rejects_unknown_tags():
    with pytest.raises(KeyError):
        gn.attribute_scores(pd.DataFrame({"dark": [0.5]}), {"tone": {"x": ["dark", "nope"]}})


def test_every_mapped_tag_is_in_the_genome_vocabulary():
    tags = set(pd.read_csv("data/raw/movielens/ml-25m/genome-tags.csv")["tag"])
    assert set(gn.map_table()["tag"]) <= tags


def test_within_vs_between_flags():
    rng = np.random.default_rng(0)
    base = rng.random(200)
    m = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.01, 200), "c": rng.random(200), "d": rng.random(200)})
    rows, random_r = gn.within_vs_between(m, {"x": {"easy": ["a", "b"], "big_city": ["c", "d"]}}, n_random=50)
    rows = rows.set_index("attribute")
    assert rows.loc["easy", "mean_within_r"] > 0.9 and not rows.loc["easy", "genome_checkable"]
    assert rows.loc["big_city", "scoring"] == "max (any of)"
    assert len(random_r) > 0
