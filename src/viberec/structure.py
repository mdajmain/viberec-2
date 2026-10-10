"""Step 6: structure in the vibe space. PCA (how many independent vibe directions are there?) with parallel
analysis to decide which directions are real, and clustering helpers.
"""
import numpy as np
import pandas as pd


def fit_pca(z, seed=42):
    """PCA on already-standardised vibe scores (movies x vibes)."""
    from sklearn.decomposition import PCA
    return PCA(random_state=seed).fit(z)


def parallel_analysis(z, n_iter=20, quantile=95, seed=42):
    """For each component, the given percentile of its explained variance when every column is shuffled
    independently (all real structure destroyed). A real component must beat this noise level."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    x = np.asarray(z, float)
    noise = []
    for _ in range(n_iter):
        shuffled = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        noise.append(PCA().fit(shuffled).explained_variance_)
    return np.percentile(np.array(noise), quantile, axis=0)


def component_labels(pca, columns, n_components, top=3):
    """For each component, the vibes with the largest positive and negative loadings (what it means)."""
    rows = []
    for i in range(n_components):
        load = pd.Series(pca.components_[i], index=columns).sort_values()
        rows.append({"component": i + 1, "explained": pca.explained_variance_ratio_[i],
                     "high_end": ", ".join(load.index[::-1][:top]), "low_end": ", ".join(load.index[:top])})
    return pd.DataFrame(rows)


def kmeans_cosine(z, k, seed=42, n_init=4):
    """k-means on unit-length vectors (so distance follows cosine similarity, like the recommender).
    Returns (labels, unit-length centroids)."""
    from sklearn.cluster import KMeans
    x = np.asarray(z, float)
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(u)
    c = km.cluster_centers_
    return km.labels_, c / np.linalg.norm(c, axis=1, keepdims=True)


def silhouette_cosine(z, labels, sample=5000, seed=42):
    """Silhouette score (-1..1, higher = cleaner groups) with cosine distance, on a random sample of movies."""
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(np.asarray(z, float), labels, metric="cosine",
                                  sample_size=min(sample, len(labels)), random_state=seed))
