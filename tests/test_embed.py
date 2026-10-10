import numpy as np

from src.viberec import embed


class FakeModel:
    """Embeds a chunk as a 3-d vector of word counts for 'dark', 'happy', 'other' (unit length)."""
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, chunks, batch_size, normalize_embeddings, show_progress_bar):
        out = []
        for c in chunks:
            w = c.split()
            v = np.array([w.count("dark"), w.count("happy"), sum(x not in ("dark", "happy") for x in w)], float)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_chunk_text():
    assert embed.chunk_text("a b c d e", max_words=2) == ["a b", "c d", "e"]
    assert embed.chunk_text(None) == [] and embed.chunk_text("   ") == []


def test_embed_sections_averages_chunks_and_handles_missing():
    vecs, has = embed.embed_sections(FakeModel(), ["dark dark", None, "happy happy happy"], max_words=1)
    assert list(has) == [True, False, True]
    assert np.allclose(vecs[0], [1, 0, 0]) and np.allclose(vecs[1], 0) and np.allclose(vecs[2], [0, 1, 0])
    assert np.allclose(np.linalg.norm(vecs[[0, 2]], axis=1), 1)


def test_similarity_scores():
    vecs = np.array([[1, 0, 0], [0, 1, 0]], float)
    q = np.array([[1, 0, 0]], float)
    assert embed.similarity_scores(vecs, q).ravel().tolist() == [1, 0]


def test_embed_sections_resumable_skips_finished_blocks(tmp_path):
    texts = ["dark", "happy", None, "dark happy", "happy"]
    calls = {"n": 0}

    class Counting(FakeModel):
        def encode(self, chunks, **kw):
            calls["n"] += 1
            return super().encode(chunks, **kw)

    v1, h1 = embed.embed_sections_resumable(Counting(), texts, tmp_path, block_size=2)
    first = calls["n"]
    v2, h2 = embed.embed_sections_resumable(Counting(), texts, tmp_path, block_size=2)
    assert calls["n"] == first                                  # second run loads every block from disk
    assert np.allclose(v1, v2) and list(h1) == [True, True, False, True, True]
    (tmp_path / "block_00001.npz").unlink()                    # simulate a crash that lost one block
    embed.embed_sections_resumable(Counting(), texts, tmp_path, block_size=2)
    assert calls["n"] == first + 1                              # only the missing block is redone
    import pytest
    with pytest.raises(ValueError):
        embed.embed_sections_resumable(Counting(), texts[:3], tmp_path, block_size=2)
