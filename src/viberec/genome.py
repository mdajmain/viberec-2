"""MovieLens Tag Genome as a benchmark for vibe attributes (Step 5).

VIBE_MAP lines up the vibe dimensions of PROJECT.md §7 with Genome tags. Each attribute's Genome
score is the mean relevance of its tags. The grouping is a hand-made judgment, so it is checked
with data (tags in one attribute should correlate across movies) and reviewed by Adil.
"""
import numpy as np
import pandas as pd

# dimension -> attribute -> Genome tags that indicate it
VIBE_MAP = {
    "tone": {
        "dark_bleak": ["dark", "bleak", "grim", "depressing", "disturbing", "gritty"],
        "heartwarming": ["heartwarming", "feel-good", "feel good movie", "touching", "sweet", "cute"],
        "tense": ["tense", "suspense", "suspenseful", "intense"],
        "melancholic": ["melancholic", "melancholy", "bittersweet", "sad", "poignant", "heartbreaking"],
        "whimsical": ["whimsical", "quirky", "surreal", "dreamlike", "weird", "bizarre"],
        "scary": ["scary", "creepy"],
        "funny": ["funny", "hilarious", "humorous", "very funny"],
        "atmospheric": ["atmospheric", "moody", "stylish"],
    },
    "pacing": {
        "slow": ["slow", "slow paced", "meditative", "reflective"],
        "fast": ["fast paced", "action packed"],
    },
    "watch_effort": {
        "demanding": ["thought-provoking", "cerebral", "complex", "intellectual", "philosophical", "mindfuck", "confusing"],
        "easy": ["predictable", "simple", "entertaining", "dumb but funny"],
    },
    "comfort": {
        "comfort": ["feel-good", "fun", "fun movie", "light", "silly fun", "nostalgic"],       # indirect only
    },
    "emotional_arc": {
        "underdog_rise": ["underdog", "rags to riches", "inspirational", "inspiring"],
        "redemption": ["redemption"],
        "tragedy": ["tragedy"],
        "revenge": ["revenge"],
        "survival": ["survival"],
    },
    "ending": {
        "happy_ending": ["happy ending"],
        "twist": ["twist ending", "plot twist", "surprise ending", "twists & turns"],
        "non_hollywood": ["non-hollywood ending"],
    },
    "setting": {
        "big_city": ["new york", "new york city", "los angeles", "london", "paris"],
        "small_town": ["small town"],
        "suburbia": ["suburbia"],
        "workplace": ["office", "workplace", "corporate america"],
        "school": ["high school", "school", "boarding school", "college"],
        "period": ["period piece", "historical"],
        "future": ["future", "dystopic future"],
        "space": ["space", "space travel"],
    },
    "characters": {
        "teen_coming_of_age": ["teen", "teenager", "teenagers", "teens", "teen movie", "coming of age", "coming-of-age"],
        "family_parents": ["family", "parenthood", "single father", "family drama", "dysfunctional family"],
        "working_people": ["working class", "workplace", "lawyers", "doctors", "journalism", "teacher"],
        "midlife": ["midlife crisis"],
        "marriage": ["marriage", "divorce", "wedding"],
    },
    "humor_style": {
        "dark_humor": ["dark humor", "dark comedy", "black comedy"],
        "satire": ["satire", "parody"],
        "slapstick": ["slapstick"],
        "deadpan_witty": ["deadpan", "witty", "clever"],
    },
    "violence": {
        "violent": ["violent", "violence", "gore", "bloody", "brutal", "brutality", "gratuitous violence"],
    },
    "romance": {
        "romantic": ["romantic", "romance", "love story"],
        "steamy": ["sexy", "sex", "sexual"],
    },
    "aesthetic": {
        "christmas": ["christmas"],
        "visually_striking": ["visually stunning", "beautiful scenery", "colourful", "visually appealing"],
    },
}


# attributes whose tags are alternatives (one city, one job), not synonyms: scored as the highest tag,
# and not expected to correlate with each other
ANY_OF = {"big_city", "workplace", "working_people", "family_parents"}

# attributes whose tags failed the agreement check (2026-10-08): the Genome cannot check them,
# so they are validated with a hand-labeled set instead
NOT_GENOME_CHECKABLE = {"easy", "comfort"}


def map_table(vibe_map=VIBE_MAP):
    """VIBE_MAP as a flat table: dimension, attribute, tag."""
    return pd.DataFrame([(d, a, t) for d, attrs in vibe_map.items() for a, tags in attrs.items() for t in tags],
                        columns=["dimension", "attribute", "tag"])


def genome_matrix(scores_csv, tags_csv):
    """Movies x tags relevance matrix (float32), columns named by tag."""
    tags = pd.read_csv(tags_csv).set_index("tagId")["tag"]
    scores = pd.read_csv(scores_csv, dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"})
    m = scores.pivot(index="movieId", columns="tagId", values="relevance")
    m.columns = tags.loc[m.columns].to_numpy()
    return m


def attribute_scores(matrix, vibe_map=VIBE_MAP):
    """Movies x attributes: mean relevance of each attribute's tags."""
    table = map_table(vibe_map)
    missing = sorted(set(table["tag"]) - set(matrix.columns))
    if missing:
        raise KeyError(f"tags not in the Genome: {missing}")
    return pd.DataFrame({a: (matrix[g["tag"].tolist()].max(axis=1) if a in ANY_OF else matrix[g["tag"].tolist()].mean(axis=1))
                         for a, g in table.groupby("attribute", sort=False)})


def within_vs_between(matrix, vibe_map=VIBE_MAP, n_random=2000, seed=0):
    """Per attribute: mean Spearman correlation among its tags, against random tag pairs (baseline)."""
    ranks = matrix.rank()
    table = map_table(vibe_map)
    rows = []
    for attr, g in table.groupby("attribute", sort=False):
        tags = g["tag"].tolist()
        if len(tags) < 2:
            rows.append({"attribute": attr, "tags": 1, "mean_within_r": np.nan, "min_pair_r": np.nan,
                         "scoring": "single tag", "genome_checkable": attr not in NOT_GENOME_CHECKABLE})
            continue
        c = ranks[tags].corr().to_numpy()
        pairs = c[np.triu_indices(len(tags), 1)]
        rows.append({"attribute": attr, "tags": len(tags), "mean_within_r": pairs.mean(), "min_pair_r": pairs.min(),
                     "scoring": "max (any of)" if attr in ANY_OF else "mean",
                     "genome_checkable": attr not in NOT_GENOME_CHECKABLE})
    rng = np.random.default_rng(seed)
    cols = ranks.columns.to_numpy()
    pick = rng.integers(0, len(cols), (n_random, 2))
    pick = pick[pick[:, 0] != pick[:, 1]]
    a, b = ranks[cols[pick[:, 0]]].to_numpy(), ranks[cols[pick[:, 1]]].to_numpy()
    a, b = a - a.mean(0), b - b.mean(0)
    random_r = (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return pd.DataFrame(rows), random_r
