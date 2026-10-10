import numpy as np
import pandas as pd

from src.viberec import retrieval as rt
from src.viberec.search_engine import neighbour_features, rank_two_step


class FakeEngine:
    """Six films; film 0 is the sample. Films 1-2 are close to it, 3-5 far, but 5 matches the requested vibe best."""
    def __init__(self):
        self.ids = np.array([10, 11, 12, 13, 14, 15])
        self.pos = pd.Series(np.arange(6), index=self.ids)
        self.z = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [-1, 0], [-0.9, 0.1], [0, 1.0]], float)
        self.als = self.z.copy()
        self.year = np.array([2003, 2004, 2001, 1970, 2010, 2005], float)
        self.recent = np.zeros(6)
        self.genres = np.array([["Comedy"]] * 6, dtype=object)

    def relevance_parts(self, parsed, vibe_mode, rows):
        return {"vibe": self.z[rows][:, 1]}, {}

    def rank(self, parsed, **kw):
        return []


def test_two_step_only_returns_neighbours():
    e = FakeEngine()
    parsed = {"reference_ids": [10], "genres": [], "vibe_phrases": [], "literal_phrases": [], "vibe_up": ["x"], "vibe_down": []}
    out = rank_two_step(e, parsed, k=5, n_neighbours=2)
    assert sorted(out) == [11, 12]                 # film 15 fits the vibe best but is not near the sample


def test_neighbour_features_years_older_only():
    f = neighbour_features(FakeEngine(), 10)
    assert list(f["years_older"]) == [0, 0, 2, 33, 0, 0]


def test_blend_takes_split_and_skips_duplicates():
    from src.viberec.search_engine import blend
    a, b = [1, 2, 3, 4, 5], [2, 6, 7, 8, 9]
    assert blend(a, b, 3) == [1, 2, 3, 6, 7]
    assert blend(a, b, 5) == a and blend(a, b, 0) == b
    assert blend([1], [2], 1, k=3) == [1, 2]                  # short lists: as many as there are
