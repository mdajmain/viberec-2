"""Text embeddings for movies (Step 5): one vector per text section (plot, reception), PROJECT.md §8.

An embedding model reads a passage and returns a fixed-length vector that represents its meaning: passages
that mean similar things get vectors pointing the same way, even with different words. The model reads at
most ~512 tokens (~380 words), so a long section is split into chunks, each chunk is embedded, and the
chunk vectors are averaged (then re-normalised) into one vector for the section.
"""
import numpy as np


def chunk_text(text, max_words=300):
    """Split a text into consecutive chunks of at most max_words words (empty text -> no chunks)."""
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def embed_sections(model, texts, max_words=300, batch_size=64):
    """texts: list of section texts (None/empty allowed). Returns (vectors, has_text): one unit-length row per
    text, the mean of its chunk embeddings; rows without text are all zeros and has_text is False."""
    chunks, owner = [], []
    for i, text in enumerate(texts):
        for c in chunk_text(text, max_words):
            chunks.append(c)
            owner.append(i)
    dim = model.get_sentence_embedding_dimension()
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    if chunks:
        emb = model.encode(chunks, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        np.add.at(vectors, np.array(owner), emb)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    has_text = norms[:, 0] > 0
    vectors[has_text] /= norms[has_text]
    return vectors, has_text


def similarity_scores(vectors, query_vectors):
    """Cosine similarity of each movie vector with each query vector (both unit length): movies x queries."""
    return vectors @ query_vectors.T


def embed_sections_resumable(model, texts, out_dir, block_size=2000, max_words=300, batch_size=64):
    """embed_sections in blocks of block_size texts; each finished block is saved to out_dir at once (temp file,
    then rename), so a crash, hang or sleep loses at most one block and a rerun skips finished blocks.
    A manifest stops blocks for a different text list or block size from being mixed in."""
    import hashlib
    import json
    import os
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1("\x1e".join(t if isinstance(t, str) else "" for t in texts).encode()).hexdigest()
    manifest = {"n_texts": len(texts), "block_size": block_size, "max_words": max_words, "texts_sha1": digest}
    mpath = out_dir / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise ValueError(f"{out_dir} holds blocks for different texts or settings; use a new folder")
    mpath.write_text(json.dumps(manifest))
    parts_v, parts_h = [], []
    for start in range(0, len(texts), block_size):
        path = out_dir / f"block_{start // block_size:05d}.npz"
        if not path.exists():
            vec, has = embed_sections(model, texts[start:start + block_size], max_words, batch_size)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, vectors=vec, has_text=has)
            os.replace(tmp, path)
        block = np.load(path)
        parts_v.append(block["vectors"])
        parts_h.append(block["has_text"])
    return np.concatenate(parts_v), np.concatenate(parts_h)
