import numpy as np

from src.viberec import retrieval as rt


def test_bm25_prefers_rare_matching_words():
    docs = ["a heist in the city", "a quiet city story", "the city the city the city", "a ghost haunts a house"]
    b = rt.BM25(docs)
    s = b.scores("ghost house")
    assert np.argmax(s) == 3 and s[0] == 0
    s2 = b.scores("heist city")
    assert np.argmax(s2) == 0                           # rare "heist" beats repeated common "city"
    assert (b.scores("unknownword") == 0).all()


def test_cosine_rank_and_rrf():
    movies = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    s = rt.cosine_rank_scores(movies, [1.0, 0.0])
    assert np.argmax(s) == 0
    fused = rt.rrf([np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])])
    assert fused[1] > 0 and np.isclose(fused[0], fused[2])   # symmetric disagreement -> ties at the ends


class TinyEncoder:
    """Embeds by keyword: 'calm' words point one way, 'violent' words the other."""
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.array([sum(w in t for w in ("calm", "gentle", "soothing")), sum(w in t for w in ("violent", "brutal"))], float)
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)


def test_embedding_vibe_profile_leans_the_right_way():
    prof = rt.embedding_vibe_profile(TinyEncoder(), "something soothing", ["calm gentle", "violent brutal"])
    assert prof[0] > 0 > prof[1]


def test_precision_at_k():
    assert rt.precision_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == 0.5


def test_rrf_tied_scores_share_a_rank():
    fused = rt.rrf([np.array([1.0, 0.0, 0.0, 0.0])])
    assert fused[1] == fused[2] == fused[3] == 1 / 62
