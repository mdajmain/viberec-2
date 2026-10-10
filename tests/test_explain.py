import numpy as np
import pandas as pd

from src.viberec import cf
from src.viberec import explain as ex


def test_liked_contributions_sum_to_fold_in_score():
    rng = np.random.default_rng(0)
    items = rng.normal(size=(50, 8)).astype(np.float64)
    yty = items.T @ items
    liked = [3, 7, 11, 20]
    u = cf.fold_in_user(items, yty, liked, alpha=10.0, reg=0.1)
    parts = ex.liked_contributions(items, yty, liked, target_col=30, alpha=10.0, reg=0.1)
    assert len(parts) == 4
    assert abs(parts.sum() - items[30] @ u) < 1e-3


def test_vibe_contributions_sum_to_cosine():
    m = pd.Series([1.0, -2.0, 0.5], index=["a", "b", "c"])
    t = pd.Series([1.0, -2.0, 0.0], index=["a", "b", "c"])
    parts = ex.vibe_contributions(m, t)
    cos = m @ t / (np.linalg.norm(m) * np.linalg.norm(t))
    assert abs(parts.sum() - cos) < 1e-12
    assert ex.top_positive(parts, k=3) == ["b", "a"]          # b: (-2)(-2) = 4 beats a: (1)(1) = 1


def test_top_drivers_only_positive_and_skips_base():
    row = pd.Series({"als_z": 0.5, "vibe_z": -0.2, "log_recent": 0.9, "base": 3.0})
    assert ex.top_drivers(row, k=3) == ["log_recent", "als_z"]


def test_template_uses_only_evidence():
    ev = {"because_liked": ["Hitch (2005)"], "shared_vibes": ["romantic", "funny"], "recent_ratings": 1200,
          "fair_score": 3.9, "year": 2004, "all_ratings": 20000}
    s = ex.template_sentence(["als_z", "vibe_z"], ev)
    assert "Hitch (2005)" in s and "romantic, funny" in s


def test_grounding_flags_unlisted_titles_and_vibes():
    titles = ["Hitch (2005)", "Titanic (1997)"]
    ok = ex.grounding_problems("Because you liked Hitch (2005); romantic and funny.", ["Hitch (2005)"],
                               ["romantic", "funny"], titles)
    assert ok == []
    bad = ex.grounding_problems("Like Titanic (1997), it is dark.", ["Hitch (2005)"], ["romantic"], titles)
    assert sorted(bad) == ["title: Titanic (1997)", "vibe: dark"]
    # titles are also caught without the year and with MovieLens's "Name, The" turned around
    assert ex.grounding_problems("Like The Hangover, it is funny.", [], ["funny"], ["Hangover, The (2009)"]) == \
        ["title: Hangover, The (2009)"]
    assert ex.grounding_problems("It lifts you up.", [], [], ["Up (2009)"]) == []        # short names only with year


def test_llm_evidence_has_only_driver_facts():
    ev = {"because_liked": ["Hitch (2005)"], "shared_vibes": ["romantic"], "recent_ratings": 1200,
          "fair_score": 3.94, "year": 2004, "all_ratings": 20000}
    e = ex.evidence_for_llm("The Proposal (2009)", ["log_recent", "als_z"], ev)
    assert [r["reason"] for r in e["reasons"]] == ["popular right now", "people with your taste liked it"]
    assert "romantic" not in str(e)                            # vibe was not a top driver, so it is not given


def test_grounding_ignores_words_inside_evidence_titles():
    cat = ["Dark Knight, The (2008)", "Shawshank Redemption, The (1994)", "Sunshine (2007)",
           "Eternal Sunshine of the Spotless Mind (2004)", "Watch, The (2012)", "Magnificent Seven, The (2016)",
           "Magnificent Seven, The (1960)", "Titanic (1997)"]
    allowed = ["Dark Knight, The (2008)", "Shawshank Redemption, The (1994)",
               "Eternal Sunshine of the Spotless Mind (2004)", "Magnificent Seven, The (1960)"]
    text = ("You should watch the film: people who loved The Dark Knight, The Shawshank Redemption, "
            "Eternal Sunshine of the Spotless Mind and The Magnificent Seven liked it.")
    assert ex.grounding_problems(text, allowed, [], cat) == []
    assert ex.grounding_problems(text + " It is dark, like Titanic (1997).", allowed, [], cat) == ["title: Titanic (1997)", "vibe: dark"]


def test_display_title_and_clean_text():
    assert ex.display_title("Dark Knight, The (2008)") == "The Dark Knight (2008)"
    assert ex.display_title("Hitch (2005)") == "Hitch (2005)"
    assert ex.clean_text("**Good Will Hunting**\n\nPeople who loved *Memento* liked it") == \
        "Good Will Hunting People who loved Memento liked it"


def test_grounding_flags_genre_words():
    assert ex.grounding_problems("This thriller is popular right now.", [], [], []) == ["genre: thriller"]
    assert ex.grounding_problems("One of the best comedies around.", [], [], []) == ["genre: comedy"]


def test_part_contributions_follow_rank():
    parts = {"vibe": np.array([3.0, 2.0, 1.0]), "keywords": np.array([1.0, 2.0, 3.0])}
    c = ex.part_contributions(parts, row=0)
    assert c.index[0] == "vibe" and abs(c["vibe"] - 1 / 61) < 1e-12 and abs(c["keywords"] - 1 / 63) < 1e-12


def test_words_in_text_and_spoilers():
    assert ex.words_in_text(["urban life", "working people"], "Two working people meet in an urban office.") == \
        ["urban", "working", "people"]
    assert ex.spoiler_problems("A warm film with a happy ending.") == ["spoiler: ending"]
    assert ex.spoiler_problems("A warm, funny film.") == []


def test_top_feel_skips_spoiler_vibes():
    attrs = ["happy_ending", "funny", "romantic", "dark_bleak"]
    assert ex.top_feel(np.array([3.0, 2.0, 1.0, -1.0]), attrs) == ["funny", "romantic"]


def test_grounding_allows_evidence_genres():
    assert ex.grounding_problems("A funny romance.", [], ["funny"], [], allowed_genres=["Romance"]) == []
    assert ex.grounding_problems("A funny romance.", [], ["funny"], []) == ["genre: romance"]


def test_grounding_titles_only_when_written_as_titles():
    cat = ["Following (1998)", "Ordinary People (1980)", "Inception (2010)", "Vacation (2015)"]
    text = "Following a gangster's rise, ordinary people face a vacation gone wrong, like Inception."
    assert ex.grounding_problems(text, [], [], cat) == ["title: Inception (2010)"]


def test_grounding_accepts_words_from_the_premise_and_genre_synonyms():
    assert ex.grounding_problems("A tense animated thriller.", [], [], [], allowed_genres=["Animation"],
                                 evidence_text="A tense thriller about a heist.") == []


def test_premise_text_keeps_whole_sentences():
    ov = "One. " + "word " * 70 + "end. Three."
    assert ex.premise_text(ov, max_words=60) == "One."
    assert ex.premise_text("Short one. Short two.") == "Short one. Short two."


def test_spoiler_words_from_the_premise_are_setup():
    assert ex.spoiler_problems("A guard on death row meets a gentle giant.", premise="Set on death row in a prison.") == []
    assert ex.spoiler_problems("He dies at the end.", premise="Set on death row.") == ["spoiler: dies"]
