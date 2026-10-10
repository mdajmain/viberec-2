import numpy as np

from src.viberec import structure as st


def test_parallel_analysis_keeps_only_real_components():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2000, 2))                                  # 2 real directions
    x = np.hstack([latent[:, [0]] + 0.3 * rng.normal(size=(2000, 3)),    # 3 columns driven by direction 1
                   latent[:, [1]] + 0.3 * rng.normal(size=(2000, 3)),    # 3 columns driven by direction 2
                   rng.normal(size=(2000, 4))])                          # 4 pure-noise columns
    x = (x - x.mean(0)) / x.std(0)
    pca = st.fit_pca(x)
    noise = st.parallel_analysis(x, n_iter=10)
    assert (pca.explained_variance_ > noise).sum() == 2


def test_kmeans_and_cluster_search():
    import pandas as pd
    from src.viberec import recommenders as rc
    rng = np.random.default_rng(1)
    a = rng.normal([5, 0], 0.3, (100, 2)); b = rng.normal([0, 5], 0.3, (100, 2))
    x = np.vstack([a, b])
    labels, cents = st.kmeans_cosine(x, 2)
    assert len(set(labels[:100])) == 1 and labels[0] != labels[150]       # the two blobs are separated
    assert st.silhouette_cosine(x, labels) > 0.8
    vibes = pd.DataFrame(x, index=range(200), columns=["p", "q"])
    z = rc.standardise(vibes).to_numpy(); zc = []
    for c in range(2):
        m = z[labels == c].mean(0); zc.append(m / np.linalg.norm(m))
    rec = rc.make_cluster_vibe_match(vibes, list(range(200)), pd.Series({9: 0.0}), pd.Series(labels, index=range(200)),
                                     np.array(zc), n_probe=1, k=5)
    recs, scanned = rec(9, pd.DataFrame({"movieId": [0], "rating": [5.0]}))     # liked a movie from blob a
    assert scanned == 100 and all(r < 100 for r in recs) and 0 not in recs
