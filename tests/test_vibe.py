import numpy as np
import pandas as pd

from src.viberec import vibe_eval as ve
from src.viberec import vibe_llm as vl


def test_tool_spec_covers_checkable_attributes_only():
    props = vl.tool_spec()["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]["attributes"]["properties"]
    assert set(props) == set(vl.ATTRIBUTES)
    assert "easy" not in props and "comfort" not in props
    assert len(props) == 41


def test_build_prompt_cuts_long_text():
    p = vl.build_prompt("X", 2000, "Drama|Comedy", "word " * 2000, reception="good " * 10, keywords=["a", "b"])
    assert "[...]" in p and "Genres: Drama, Comedy" in p and "Keywords: a, b" in p
    assert "Keywords" not in vl.build_prompt("X", 2000, "Drama", "short plot")


def test_check_answer():
    good = {"attributes": {a: 1 for a in vl.ATTRIBUTES}, "light_heavy": 3, "pacing": 4, "ending_type": "bittersweet",
            "vibe_paragraph": "word " * 120}
    assert vl.check_answer(good) == []
    bad = {**good, "light_heavy": 9, "ending_type": "twist", "attributes": {**good["attributes"], "tense": 7}}
    assert len(vl.check_answer(bad)) == 3


def test_agreement_on_simulated_llm():
    """A noisy copy of the truth should agree and beat a baseline that knows nothing."""
    rng = np.random.default_rng(1)
    idx = pd.Index(range(300), name="movieId")
    truth = pd.DataFrame({"dark": rng.random(300), "funny": rng.random(300)}, index=idx)
    llm = (truth * 3 + rng.normal(0, 0.4, truth.shape)).round().clip(0, 3)
    base = pd.DataFrame(rng.random(truth.shape), index=idx, columns=truth.columns)
    res = ve.agreement(llm, truth, baseline=base, n_boot=200).set_index("attribute")
    assert (res["llm_rho"] > 0.7).all() and res["beats_baseline"].all()


def test_genre_baseline_uses_only_other_movies():
    attr = pd.DataFrame({"scary": [0.9, 0.8, 0.1, 0.5]}, index=pd.Index([1, 2, 3, 4], name="movieId"))
    genres = pd.Series({1: "Horror", 2: "Horror", 3: "Comedy", 4: "Horror"})
    pred = ve.genre_baseline(attr, genres, [4])
    assert abs(pred.loc[4, "scary"] - 0.85) < 1e-9   # mean of movies 1 and 2, not movie 4 itself


def test_cohen_kappa():
    assert ve.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1
    assert abs(ve.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9     # chance-level


def test_label_agreement_perfect_llm_beats_baseline():
    rng = np.random.default_rng(0)
    idx = pd.Index(range(80))
    human = pd.DataFrame({"light_heavy": rng.integers(1, 6, 80), "pacing": rng.integers(1, 6, 80),
                          "ending": rng.choice(["happy", "sad", "open"], 80)}, index=idx)
    genres = pd.Series(rng.choice(["Drama", "Comedy", "Horror"], 80), index=idx)
    res = ve.label_agreement(human.copy(), human, genres, n_boot=100).set_index("label")
    assert abs(res.loc["light_heavy", "llm"] - 1) < 1e-9 and res.loc["light_heavy", "beats_baseline"]
    assert res.loc["ending", "llm"] == 1


def test_loo_predict_recovers_signal():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.random((60, 5)), columns=list("abcde"))
    y = (X["a"] * 4 + 1).round()                                   # label driven by feature a only
    pred = ve.loo_predict(X, y, kind="ordinal")
    from scipy.stats import spearmanr
    assert spearmanr(pred, y)[0] > 0.8
    cats = np.where(X["b"] > 0.5, "happy", "sad")
    assert (ve.loo_predict(X, cats, kind="category") == cats).mean() > 0.8


def test_compact_tool_spec_and_check():
    props = vl.tool_spec(compact=True)["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]
    assert set(props) == {"attributes", "light_heavy", "pacing", "ending_type"}
    ok = {"attributes": {a: 2 for a in vl.ATTRIBUTES}, "light_heavy": 4, "pacing": 3, "ending_type": "happy"}
    assert vl.check_compact(ok) == []
    assert vl.check_compact({**ok, "pacing": 0}) == ["pacing=0 not in 1-5"]


def test_agreement_skips_missing_scores_per_attribute():
    """A movie the LLM left blank on one vibe is dropped for that vibe only, not turned into NaN for the whole vibe."""
    rng = np.random.default_rng(2)
    idx = pd.Index(range(200), name="movieId")
    truth = pd.DataFrame({"dark": rng.random(200), "funny": rng.random(200)}, index=idx)
    llm = truth.copy()
    llm.iloc[:5, 0] = np.nan
    base = pd.DataFrame(rng.random(truth.shape), index=idx, columns=truth.columns)
    res = ve.agreement(llm, truth, baseline=base, n_boot=100).set_index("attribute")
    assert res.loc["dark", "movies"] == 195 and res.loc["funny", "movies"] == 200
    assert res["llm_rho"].notna().all() and res["beats_baseline"].all()


def test_three_vibe_form_has_only_three_fields():
    schema = vl.tool_spec(keep=vl.THREE)["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert sorted(schema["properties"]) == sorted(vl.THREE) and sorted(schema["required"]) == sorted(vl.THREE)
    assert vl.check_three({"light_heavy": 4, "pacing": 2, "ending_type": "bittersweet"}) == []
    assert len(vl.check_three({"light_heavy": 0, "pacing": 2, "ending_type": "twist"})) == 2
