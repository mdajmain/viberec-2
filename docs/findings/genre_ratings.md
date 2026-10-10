> **Note:** the numbers in this write-up come from the earlier genre analysis and may not match the current `notebooks/03_eda.ipynb` (for example, after the genre comma fix). To be updated in Step 4. The figures and the bootstrap file it cites are archived (see ARTIFACTS.md).

# Do movie ratings vary by genre?

**Status:** Analysis complete. Decision made — Genre-specific (Adil, 2026-10-06).
**Notebook:** `notebooks/04_eda.ipynb`

## Question

Do `mean_rating` distributions differ meaningfully across genres?

## Why it matters

Decides whether Step 4's quality-tier rating/count cutoffs (PROJECT.md §4) should be genre-specific or a single global threshold. A global threshold applied to a genre that structurally rates lower (for reasons unrelated to quality) would systematically exclude or down-tier that genre's movies regardless of how good they are relative to their own genre's norms.

## Hypotheses

- **H0:** `mean_rating` distributions are the same across genres.
- **H1:** at least one genre differs.

## Practical significance (chosen before seeing results)

A genre gap of **≥0.5 stars** counts as practically significant. Chosen because it equals MovieLens's own rating granularity (individual ratings are given in 0.5-star increments) — a gap smaller than the scale's own resolution isn't a meaningful design input regardless of statistical significance.

## Data and population

- **Source:** `data/processed/movies_full.parquet`.
- **Population: n = 50,046** (movies with at least one rating, within the final post-titleType-filter catalog of 52,974) — **not** the 59,047 used in notebook 03's rating-count/mean-rating analysis.
- **Why the population changed:** Step 2's `titleType == "movie"` strict filter dropped 9,449 non-movie IMDb-type rows. Of those, **9,001 had at least one MovieLens rating** (the other 448 were already unrated). Breakdown of the dropped-but-rated rows by IMDb type: `tvMovie` 3,351 · `short` 2,752 · `video` 1,441 · `tvSpecial` 470 · `tvMiniSeries` 450 · `tvEpisode` 366 · `tvShort` 88 · `tvSeries` 83. 59,047 − 9,001 = 50,046, confirming the filter is the entire explanation.
- **Genre explosion:** `genres` mixes two delimiter conventions — MovieLens uses `|`, IMDb (used wherever Step 2 backfilled a null MovieLens genre) uses `,`. Splitting on `[|,]` handles both; splitting on `,` alone (an earlier mistake, caught before this write-up) silently failed to split any MovieLens-native row.
- **Non-independence:** a movie belonging to multiple genres (e.g. Action+Comedy) is counted in every one of those groups. This is a real limitation of every test below, not just a footnote — it inflates the effective sample size per group and violates the independence assumption Kruskal-Wallis/Dunn's technically require. We proceed anyway per the plan, flagged here and in Limitations.
- **`(no genres listed)` fallback:** 26 movies have no genre at all post-IMDb-backfill. Excluded from every per-genre group; counted only in the global/overall statistics.
- **Small-genre flag:** genres with n < 200 are flagged as too small to trust for the inferential tests (Kruskal-Wallis, Dunn's, bootstrap CIs used only the n≥200 set below). The cutoff comes from a visible natural gap in the count distribution (Film-Noir at 354 vs. IMAX at 176) — not picked a priori. Flagged genres: `IMAX` (176), `Biography` (164), `History` (150), `Family` (125), `Music` (115), `Sport` (43), `Adult` (5), `News` (7). Several of these (`Biography`, `History`, `Family`, `Music`, `Sport`, `News`, `Adult`) exist only in IMDb's genre vocabulary, not MovieLens's own 20-genre set — a taxonomy mismatch worth separate attention later, not resolved here.

## Method

| Step | Test/approach | Why it fits |
|---|---|---|
| Descriptives | n, mean, median, IQR, p10/p25/p75/p90 per genre | Percentiles (especially p10/p25) are where a genre-specific floor would actually sit |
| Sensitivity | Medians recomputed at rating_count ≥1/10/50/100 | Checks whether genre gaps are real signal or low-count noise artifacts; cutoffs are diagnostic only, not candidate thresholds |
| Omnibus test | Kruskal-Wallis + epsilon-squared | Ratings aren't normal (confirmed in notebook 03); Kruskal-Wallis doesn't assume it. Effect size reported because p-values are uninformative at n≈94k (exploded rows) |
| Pairwise | Dunn's test, Holm-corrected | Standard nonparametric post-hoc for Kruskal-Wallis, Holm correction controls family-wise error across all pairs |
| Uncertainty | Bootstrap 95% CI per genre median (2,000 resamples) | Nonparametric, makes no distributional assumption about the median's sampling distribution |
| Confound | Popularity-quartile bucketing + OLS (`mean_rating ~ genre + log(rating_count)`) | Tests whether genre differences are actually a popularity effect in disguise, given notebook 03's weak positive rating_count/rating correlation |

## Results

**Descriptives (robust genres, n≥200), sorted by median:**

| Genre | n | median | p10 | p25 | p75 | p90 |
|---|---|---|---|---|---|---|
| Documentary | 4,705 | 3.500 | 2.500 | — | 3.780 | 4.000 |
| Film-Noir | 354 | 3.359 | 2.655 | — | 3.635 | 3.860 |
| War | 1,706 | 3.323 | 2.500 | — | 3.643 | 3.905 |
| Drama | 24,206 | 3.250 | 2.400 | — | 3.567 | 3.847 |
| Musical | 1,036 | 3.250 | 2.500 | — | 3.541 | 3.750 |
| Romance | 7,179 | 3.200 | 2.350 | — | 3.500 | 3.821 |
| ... | | | | | | |
| Comedy | 14,947 | 3.081 | 2.159 | — | 3.500 | 3.772 |
| Thriller | 7,846 | 3.000 | 2.000 | — | 3.419 | 3.731 |
| Action | 6,440 | 3.000 | 2.000 | — | 3.435 | 3.746 |
| Sci-Fi | 2,881 | 2.863 | 1.833 | — | 3.300 | 3.655 |
| Horror | 5,177 | 2.750 | 1.750 | — | 3.151 | 3.500 |

Overall (global) median: **3.146**. Full table with p25/p75 in the notebook output.

**Documentary − Horror gap: 0.75 stars** — exceeds the 0.5-star practical threshold on its own.

**Sensitivity:** every genre's median rises somewhat as low-count movies are excluded (consistent with the weak positive popularity/rating correlation from notebook 03), but **relative genre order is stable across all four thresholds** (Documentary/Film-Noir/War stay near the top, Horror/Sci-Fi stay at the bottom at every cutoff) — evidence the genre effect is real, not low-count noise.

**Kruskal-Wallis:** H=5002.83, df=17, p<0.001, n=94,122 (exploded rows), k=18 groups. **Epsilon-squared = 0.053** — a modest effect (genre explains roughly 5% of rank-variance in ratings). Real, not huge.

**Dunn's test (Holm-corrected):** 126 of 153 pairs significant at p<0.05. Non-significant pairs are mostly genres that are plausibly similar in tone (Action≈Thriller, Children≈Fantasy, Fantasy≈Mystery≈Western, Comedy≈Mystery≈Western, Adventure≈Children≈Comedy≈Fantasy≈Mystery≈Western) — the structure of what's *not* distinguishable looks sensible, not random.

**Bootstrap 95% CIs:** saved in `artifacts/archive/genre_median_bootstrap_cis.parquet`. Horror's CI (median 2.750) and Documentary's CI (median 3.500) don't overlap — the gap is not a sampling artifact.

**Confound check:**
- *Popularity-quartile bucketing:* Documentary stays at or near the top and Horror stays at the bottom within every quartile (Q1 through Q4) of popularity.
- *Regression* (`mean_rating ~ genre + log(rating_count)`, Drama as reference, R²=0.097): Horror coefficient **−0.479**, Documentary coefficient **+0.278** relative to Drama. Horror−Documentary gap controlling for popularity ≈ **0.757 stars** — still above the 0.5-star threshold, and log(rating_count)'s own coefficient (+0.149) confirms popularity has an effect too, but genre's effect doesn't disappear once it's accounted for.

## Figures

- `figures/archive/genre_boxplots.png` — sorted boxplots, overall median marked.
- `figures/archive/genre_forest_plot.png` — genre medians with bootstrap 95% CIs, overall median marked.
- `figures/archive/genre_small_multiples.png` — histograms for the 9 largest genres.

## Interpretation (only what the numbers support)

- Genre differences in `mean_rating` are statistically real (Kruskal-Wallis, Dunn's) and **practically real by the chosen 0.5-star threshold** for at least the Documentary/Film-Noir/War vs. Horror/Sci-Fi/Action contrast — that gap is ~0.5–0.75 stars and survives both the sensitivity check and the popularity confound check.
- The effect size is modest (ε²=0.053) — genre is a real factor, not the dominant one. Most of the variance in ratings is still within-genre, not between-genre.
- Non-significant Dunn's pairs cluster among genres that are plausibly tonally similar, which is a weak sanity check that the signal isn't noise.
- This does **not** by itself say genre-specific *tiers* are the right design — it says a genre-blind global cutoff would treat Horror and Documentary inconsistently relative to their own genre's distribution.

## Limitations

- **Multi-genre overlap:** every test here treats genre-exploded rows as independent, which they are not. True effective sample sizes per genre are somewhat smaller than reported n. This likely doesn't change the qualitative conclusion (effects are large relative to typical overlap-induced inflation) but hasn't been formally corrected for (e.g. via a mixed-effects model with movie as a random effect).
- **Low-count noise:** addressed via the sensitivity check, not eliminated — individual low-count movies within any genre remain noisy.
- **MovieLens user bias:** MovieLens's rater population isn't representative of all viewers; genre rating patterns here reflect that population's tastes, not an objective quality signal.
- **Genre taxonomy mismatch:** 7 of the 26 genres (`Biography`, `History`, `Family`, `Music`, `Sport`, `News`, `Adult`) exist only because of IMDb backfill and aren't part of MovieLens's native vocabulary — and all 7 are also below the n=200 trust threshold, so this mismatch didn't end up affecting the robust-genre conclusions, but it's a data-quality issue worth revisiting.
- **No correction for genre co-occurrence patterns:** e.g. Horror-Comedy movies get counted in both genres' stats even though that combination might behave differently from pure Horror or pure Comedy.

## Decision

**Genre-specific — Adil, 2026-10-06.**

Step 4 thresholds are genre-specific, as already built in `03_eda.ipynb`. A movie with several genres is counted in each of its genres (not assigned to one primary genre) and gets the most lenient value among them: lowest floor, highest ceiling, lowest q1/q3. The multi-genre overlap only matters for significance tests between genres, not for each genre's own distribution.

## Next steps

- Decision made (genre-specific, multi-genre movies counted in every genre they belong to). The independence limitation above applies to the between-genre tests only; revisit it only if a new between-genre test is added.
- Numbers in this write-up are from the earlier analysis; update them in Step 4 (see the note at the top).
