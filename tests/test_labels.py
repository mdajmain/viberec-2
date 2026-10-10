import pandas as pd

from src.viberec import labels as lb

ML = pd.DataFrame({"movieId": [1, 2, 3, 4, 5, 6], "title": [
    "Proposal, The (2009)",
    "Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008)",
    "(500) Days of Summer (2009)",
    "Mother! (2017)",
    "Mother (2017)",
    "Proposal, The (1999)"]})


def test_match_handles_articles_alt_titles_numbers_and_punctuation():
    labels = pd.DataFrame({"title": ["THE PROPOSAL", "THE BOY IN THE STRIPED PYJAMAS", "500 DAYS OF SUMMER",
                                     "MOTHER!", "MOTHER", "NOT A MOVIE"],
                           "year": [2009, 2008, 2009, 2017, 2017, 2001]})
    out = lb.match(labels, ML)
    assert out["movieId"].tolist()[:5] == [1, 2, 3, 4, 5]     # year picks 2009, exact "!" picks movie 4
    assert pd.isna(out["movieId"].iloc[5])


def test_movielens_keys():
    assert lb.movielens_keys("Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008)") == \
        (2008, ["Boy in the Striped Pajamas, The", "Boy in the Striped Pyjamas, The"])
    assert lb.movielens_keys("(500) Days of Summer (2009)") == (2009, ["(500) Days of Summer"])


def test_read_labels_v2_flips_and_maps(tmp_path):
    path = tmp_path / "v2.xlsx"
    pd.DataFrame({"#": [1, 2, 3], "Movie title": ["A", "B", "THE HOLMES?"], "Year": [2000, 2001, 2011],
                  "Light vs heavy (1-5)": [1, 5, 4], "Ending": ["Hopeful", "Bleak", "Ambiguous"],
                  "Pacing (1-5)": [4, 2, 3], "Notes (optional)": ["", "", ""]}).to_excel(path, sheet_name="Movie Ratings", index=False)
    out = lb.read_labels_v2(path, title_fixes={"THE HOLMES?": "THE HELP"})
    assert out["light_heavy"].tolist() == [5, 1, 2]             # 1 = light on the sheet -> 5 = light here
    assert out["ending"].tolist() == ["happy", "sad", "open"]
    assert out["title"].tolist()[2] == "THE HELP"
