"""Check LLM vibe scores against the Tag Genome, and against a genre-only baseline (Step 5 pass/fail check).

Agreement = Spearman correlation across movies between the LLM's 0-3 score and the Genome attribute
score. Baseline = predict each attribute from the movie's genres only (mean Genome score of other movies
with the same genres). The LLM earns its place on an attribute only if it beats that baseline.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def genre_baseline(genome_attr, genres, target_ids):
    """For each target movie: mean attribute score over its genres, learned from non-target movies.

    genome_attr: movies x attributes (index movieId). genres: Series movieId -> "A|B|C".
    """
    train = genome_attr.index.difference(target_ids)
    rows = (genres.reindex(train).dropna().str.split("|").explode().rename("genre").reset_index())
    rows.columns = ["movieId", "genre"]
    per_genre = rows.join(genome_attr, on="movieId").groupby("genre")[list(genome_attr.columns)].mean()
    overall = genome_attr.loc[train].mean()
    preds = {}
    for mid in target_ids:
        g = [x for x in str(genres.get(mid, "")).split("|") if x in per_genre.index]
        preds[mid] = per_genre.loc[g].mean() if g else overall
    return pd.DataFrame(preds).T


def agreement(pred, truth, n_boot=1000, seed=0, baseline=None):
    """Per attribute: Spearman(pred, truth), and if baseline is given, Spearman(baseline, truth) and a
    paired bootstrap interval (resampling movies) for the difference."""
    ids = pred.index.intersection(truth.index)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ids), (n_boot, len(ids)))
    rows = []
    for a in pred.columns:
        p, t = pred.loc[ids, a].to_numpy(float), truth.loc[ids, a].to_numpy(float)
        b = baseline.loc[ids, a].to_numpy(float) if baseline is not None else np.zeros(len(ids))
        ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(b)      # a movie missing a score is left out of that attribute
        k_picks = picks
        if not ok.all():
            p, t, b = p[ok], t[ok], b[ok]
            k_picks = rng.integers(0, ok.sum(), (n_boot, ok.sum()))
        row = {"attribute": a, "movies": int(ok.sum()), "llm_rho": spearmanr(p, t)[0]}
        if baseline is not None:
            row["baseline_rho"] = spearmanr(b, t)[0]
            diffs = [spearmanr(p[k], t[k])[0] - spearmanr(b[k], t[k])[0] for k in k_picks]
            row["diff_lo"], row["diff_hi"] = np.nanpercentile(diffs, [2.5, 97.5])
            row["beats_baseline"] = row["diff_lo"] > 0
        rows.append(row)
    return pd.DataFrame(rows)


def loo_genre_baseline(values, genres):
    """Leave-one-out genre baseline for a hand label: each movie gets the mean label of the OTHER labeled
    movies sharing its genres (overall mean if none). values: Series movieId -> number."""
    g = genres.reindex(values.index).fillna("").str.split("|")
    preds = {}
    for mid in values.index:
        others = [o for o in values.index if o != mid and set(g[o]) & set(g[mid])]
        preds[mid] = values[others].mean() if others else values.drop(mid).mean()
    return pd.Series(preds)


def cohen_kappa(a, b):
    """Agreement between two label lists beyond what their label frequencies give by chance."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    observed = (a == b).mean()
    labels = set(a) | set(b)
    expected = sum((a == x).mean() * (b == x).mean() for x in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def label_agreement(llm, human, genres, n_boot=1000, seed=0):
    """LLM vs hand labels. Ordinal columns (light_heavy, pacing): Spearman, against a leave-one-out genre
    baseline with a paired bootstrap interval. Ending: accuracy and Cohen's kappa, against always
    predicting the most common label."""
    ids = llm.index.intersection(human.index)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ids), (n_boot, len(ids)))
    rows = []
    for col in ("light_heavy", "pacing"):
        h = human.loc[ids, col].astype(float)
        p, b = llm.loc[ids, col].astype(float).to_numpy(), loo_genre_baseline(h, genres).loc[ids].to_numpy()
        h = h.to_numpy()
        diffs = [spearmanr(p[k], h[k])[0] - spearmanr(b[k], h[k])[0] for k in picks]
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        rows.append({"label": col, "movies": len(ids), "llm": spearmanr(p, h)[0], "baseline": spearmanr(b, h)[0],
                     "metric": "Spearman", "diff_lo": lo, "diff_hi": hi, "beats_baseline": lo > 0})
    h, p = human.loc[ids, "ending"], llm.loc[ids, "ending"]
    majority = h.value_counts().idxmax()
    rows.append({"label": "ending", "movies": len(ids), "llm": (p == h).mean(), "baseline": (h == majority).mean(),
                 "metric": f"accuracy (kappa {cohen_kappa(h, p):.2f}); baseline = always '{majority}'"})
    return pd.DataFrame(rows)


def loo_predict(features, target, kind="ordinal"):
    """Leave-one-out predictions of a hand label from vibe scores: each movie is predicted by a model trained on
    all the other labeled movies. Features are standardised; the penalty is chosen inside each training fold.

    kind="ordinal": ridge regression (RidgeCV); kind="category": multinomial logistic regression (LogisticRegressionCV).
    """
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = np.asarray(features, float), np.asarray(target)
    preds = []
    for train, test in LeaveOneOut().split(X):
        if kind == "ordinal":
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        else:
            model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000))
        model.fit(X[train], y[train])
        preds.append(model.predict(X[test])[0])
    return pd.Series(preds, index=getattr(features, "index", None))


def accuracy_gain_interval(pred, truth, baseline_label, n_boot=2000, seed=0):
    """Paired bootstrap 95% interval for accuracy(pred) minus accuracy(always baseline_label)."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    hit = (pred == truth).astype(float) - (truth == baseline_label).astype(float)
    picks = np.random.default_rng(seed).integers(0, len(hit), (n_boot, len(hit)))
    return tuple(np.percentile(hit[picks].mean(axis=1), [2.5, 97.5]))
