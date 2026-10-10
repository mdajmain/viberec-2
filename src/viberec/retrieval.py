"""Step 9: retrieval for free-text requests (PROJECT.md §8). A request is either matched literally (BM25 keyword
search) or first translated into the 41 vibe dimensions and matched against each movie's vibe vector.
Step 5 showed raw document similarity cannot find "feel", so the vibe route parses the request first.
"""
import numpy as np
import pandas as pd


class BM25:
    """Okapi BM25 keyword search over movie texts (word unigrams, standard k1 = 1.5, b = 0.75)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(lowercase=True, strip_accents="unicode", token_pattern=r"(?u)\b\w+\b")
        tf = self.vec.fit_transform(texts).tocsc().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        tf = tf.tocsr()
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows])      # saturating term frequency
        self.weights = tf.tocsc()

    def scores(self, query):
        cols = [self.vec.vocabulary_[w] for w in self.vec.build_analyzer()(query) if w in self.vec.vocabulary_]
        if not cols:
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, cols] @ self.idf[cols]).ravel()


def standardise_rows(x, mean, std):
    return (np.asarray(x, float) - mean) / std


def cosine_rank_scores(movie_z, query_z):
    """Cosine similarity between each movie's standardised vibe vector and the query's."""
    m = movie_z / np.linalg.norm(movie_z, axis=1, keepdims=True)
    q = np.asarray(query_z, float)
    return m @ (q / (np.linalg.norm(q) or 1))


def embedding_vibe_profile(encoder, query, attribute_texts, attribute_vectors=None):
    """Query -> vibe profile via meaning: similarity of the query to each vibe's description, standardised across
    vibes (so the profile says which vibes the request leans towards and away from). attribute_vectors can be
    precomputed with the same encoder."""
    if attribute_vectors is None:
        attribute_vectors = encoder.encode(attribute_texts, normalize_embeddings=True)
    q = encoder.encode([query], normalize_embeddings=True)[0]
    sim = attribute_vectors @ q
    return (sim - sim.mean()) / (sim.std() or 1)


def rrf(rankings, k=60):
    """Reciprocal rank fusion of several score arrays over the same movies: sum of 1 / (k + rank).
    k = 60 is the constant from the original method (Cormack et al., 2009).

    Tied scores share the same rank ("competition ranking": 1, 1, 3, ...). Until 2026-10-10 ties were broken by row
    order, so e.g. the many films with a zero keyword score got arbitrary different ranks; that changed about a quarter
    of the search top 10 (found when checking the database version, notebook 16)."""
    from scipy.stats import rankdata
    total = np.zeros(len(rankings[0]))
    for s in rankings:
        total += 1.0 / (k + rankdata(-np.nan_to_num(np.asarray(s, float), nan=-np.inf), method="min"))
    return total


def precision_at_k(ranked_ids, relevant, k):
    return len(set(ranked_ids[:k]) & relevant) / k
