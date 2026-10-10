# Step 4: quality tiers

**Status:** Done. Tiers saved to `data/processed/movie_tiers_v2.parquet` (hybrid rule).
**Notebook:** `notebooks/05_quality_tiers.ipynb` (reusable code in `src/viberec/credibility.py`, tests in `tests/`)

Step 4 is worked through one sub-question at a time. Each section below follows the same order: question, why it matters, method, results, interpretation, limitations, decision.

**Decisions already made**
- Thresholds are genre-specific. A movie with several genres is counted in each of them and gets the most lenient value among them.
- No age buckets for now. The existing time test stays (`DATA_YEAR` from the data, `MARGIN_YEARS = 8`).
- New movies use content/vibe features plus CF, with the CF weight growing as ratings arrive.

| Sub-question | Status |
|---|---|
| Q4.1 Credibility: how many ratings until a movie's mean is trustworthy? | Done: reliability 0.8 (about 27 ratings), one global count |
| Q4.2 Age boundaries | Deferred (future work, PROJECT.md §14) |
| Q4.3 Trusted rating measure | Done: shrunk mean, k = 7.3 |
| Q4.4 Rating floor | Done: keep 04's μ − 1σ rule; too low on the raw mean, too high on the shrunk score |
| Q4.5 New-movie "good range" | Deferred: no post-2019 data |
| Q4.6 Sensitivity | Done |
| Q4.7 Validation on a time-based holdout | Done: rule and tiers validated |

---

## Q4.1 Credibility: how many ratings until a movie's mean is trustworthy?

### Question
How many ratings does a movie need before its mean rating mostly reflects the movie and not chance?

### Why it matters
The tier rules need a minimum count: enough ratings to trust a low rating enough to exclude a movie (N), and enough to trust a high one (M). Without evidence for "enough", those counts would be guesses. In this catalog 16.4% of movies have a single rating and 29.0% have two or fewer, so most movies sit in the noisy zone.

### Method
Population: the 50,046 rated movies in `movies_full.parquet` (after the titleType filter), 24,744,432 ratings.

The measure is **reliability** R(n): the share of the spread between movies' observed means that is real difference between movies rather than sampling noise. R is 0 when the means are all noise and 1 when none is.

A movie's ratings take 10 values (0.5 to 5.0), so the 25M raw ratings are reduced to one row of 10 counts per movie, saved once. Resampling or splitting that row is equivalent to doing it on the raw ratings.

Movies are grouped into 13 strata by number of ratings (quantile bins; stratum 0 is the single-rating movies). Three methods estimate reliability in each stratum:

| Method | How | Weakness |
|---|---|---|
| Standard error | R = 1 − average(σ²/n) / variance of observed means, with σ² pooled within the stratum | Assumes the mean is roughly normal; σ² is not measurable for 1 rating |
| Bootstrap | Resample each movie's own ratings 1,000 times and use the variance of the resampled means as noise | Cannot resample what is not there: wrong for very few ratings |
| Split-half (reference) | Split each movie's ratings at random into two halves, correlate the half-means across the movies in a stratum, apply the Spearman-Brown correction; 80 random splits averaged | Needs 2+ ratings; assumes nothing about distributions |

Assumption checks came first (figure `q41_variance_checks.png`). The number of repeats and resamples was chosen from convergence checks, not by hand: 1,000 bootstrap resamples differ from 250 by at most 1.6%, and with 80 splits the Monte Carlo error is at most 9% of the ordinary sampling error of each correlation.

For genres, the same split-half curve is computed per genre group (the 10 largest genres plus Other, as in `03_eda.ipynb`). Movies are resampled 4,000 times; in each resample both the genre's and the all-movies number are recomputed, so the interval for the difference carries the uncertainty of both. A genre is flagged when that interval excludes 0, at 95% and with a Bonferroni correction for 11 groups. The practical size is the share of the genre's movies whose "enough ratings" status would change if it used its own number.

### Results

**Assumptions.** Ratings are lopsided (4.0 is the most common value, 26.6%; median skewness of a movie's ratings −0.59). The within-movie variance changes with the number of ratings (1.18 at 5–6 ratings, 0.91 at 500+) and with the movie's mean (1.24 for the lowest decile, 0.80 for the highest). A single global σ would misstate noise by as much as a third, so the standard-error method uses a pooled σ² per stratum.

**Method comparison** (figure `q41_reliability_curves.png`):

| Ratings (harmonic mean) | Standard error | Bootstrap | Split-half |
|---|---|---|---|
| 1 | 0.117 | 1.000 (meaningless) | not possible |
| 2.3 | 0.209 | 0.571 | 0.201 |
| 4.0 | 0.327 | 0.496 | 0.328 |
| 6.4 | 0.471 | 0.553 | 0.454 |
| 9.2 | 0.596 | 0.637 | 0.581 |
| 14.1 | 0.689 | 0.712 | 0.683 |
| 22.3 | 0.774 | 0.785 | 0.772 |
| 40.0 | 0.855 | 0.859 | 0.852 |
| 84.6 | 0.938 | 0.939 | 0.938 |
| 215.3 | 0.980 | 0.980 | 0.980 |

The standard error tracks split-half to within 0.016 from about 6 ratings up (largest gap 0.052, at 5 ratings). The bootstrap is too optimistic below about 20 ratings.

**Ratings needed** (split-half curve; all movies):

| Reliability | Ratings needed | Share of the catalog with at least that many |
|---|---|---|
| 0.5 | 7.3 | 47.7% |
| 0.7 | 15.4 | 36.7% |
| 0.8 | 27.4 | 30.5% |
| 0.9 | 60.7 | 23.5% |

The curve fits R = n / (n + k) with k = 7.3 to within 0.038. Each extra rating adds about 0.034 reliability at 7 ratings, 0.006 at 27 and 0.002 at 61.

**By genre** (ratings needed at reliability 0.8; figure `q41_genre_reliability.png`):

| Genre group | Ratings needed | Difference vs all (95% interval) | Flagged | Movies whose status would change |
|---|---|---|---|---|
| Documentary | 57.0 | +29.7 (7.8 to 58.7) | 95% only | 5.9% |
| Horror | 40.8 | +13.5 (6.8 to 20.3) | yes | 3.9% |
| Drama | 39.6 | +12.2 (9.0 to 17.0) | yes | 3.6% |
| Romance | 35.2 | +7.9 (3.7 to 15.2) | yes | 2.5% |
| Crime | 35.2 | +7.8 (3.2 to 14.1) | yes | 2.6% |
| Thriller | 33.2 | +5.8 (1.9 to 10.8) | 95% only | 2.2% |
| Other | 30.8 | +3.5 (−0.0 to 6.8) | no | 1.1% |
| Adventure | 26.4 | −1.0 (−10.1 to 5.3) | no | 0.4% |
| Comedy | 24.7 | −2.7 (−5.5 to 0.4) | no | 1.0% |
| Action | 22.5 | −4.8 (−7.5 to 0.8) | no | 1.8% |
| Sci-Fi | 21.6 | −5.8 (−9.5 to 3.6) | no | 2.2% |

"yes" means the interval excludes 0 after the Bonferroni correction.

### Interpretation
- The reliability curve is smooth with no sharp elbow, so any cutoff is a choice of level, not something the data announces.
- Two independent methods agree, so the curve is not an artifact of one method.
- More than half of the catalog (52.3%) has fewer than 7.3 ratings; for those movies less than half of the spread in means is real.
- Genres with a narrower spread of true quality need more ratings. The between-movie variance is 0.113 for Documentary and 0.140 for Drama, against 0.23 to 0.25 for Action, Adventure and Sci-Fi. Within-movie noise is similar across genres (0.82 to 1.02).
- Horror, Drama, Romance and Crime clearly need more ratings than the all-movies number; Documentary and Thriller do at 95%. But even for Documentary only 5.9% of its movies would change status if it used its own number, because most movies sit far below or far above the cutoff.
- k = 7.3 is the shrinkage strength a Bayesian average would use, which feeds Q4.3.

### Limitations
- This measures sampling noise, not bias. People who watch obscure movies are not a random sample, so a movie with 200 ratings can still be consistently over- or under-rated.
- The raters are MovieLens users, not all viewers.
- The 16.4% of movies with one rating cannot be split or bootstrapped; their reliability (0.12) assumes the same σ² as movies with 2–3 ratings.
- Reliability is estimated per rating-count stratum, and the spread of true quality differs between strata, so it describes the movies at that count.
- Multi-genre movies are counted in each genre (as decided), so the genre groups overlap and the 11 comparisons are not independent. The Bonferroni correction is conservative for that reason.
- Genre groups follow `03_eda.ipynb` (10 largest plus Other); the 7 IMDb-only genres sit inside Other. Documentary's interval is wide.
- Only the 24,744,432 ratings of the 50,046 catalog movies are used (255,663 ratings of filtered-out title types are excluded).
- The 13 strata are for reporting; the numbers of ratings at each level are interpolated between them.

### Decision
**Decided 2026-10-08.** A movie's mean is trustworthy at split-half reliability 0.8, about 27 ratings (30.5% of the catalog has at least that many). One global count for now. Genre-specific counts are tested against it in Q4.6/Q4.7 and adopted only if they change tier assignment materially. 0.8 is a convention, so it is sensitivity-tested at 0.7 and 0.9 in Q4.6. See `docs/DECISIONS.md` for the evidence.

Saved outputs: see `docs/ARTIFACTS.md` (rows for `movie_rating_histogram_v1`, the `q41_*` files in `artifacts/`, and the three `q41_*` figures).

---

## Q4.3 Trusted rating measure: raw mean or shrunk mean?

**Question.** Which score should stand for a movie's quality when ratings are few? The score is a priority weight for ranking, not a filter: nothing is dropped.

**Method.** Build each score from a movie's *early* ratings and compare it with the mean of its *later* ratings, on the approved movies. Cutoffs are the 50th/70th/80th/90th percentiles of rating timestamps (2007-12-28, 2015-01-22, 2016-06-25, 2018-01-04), so no date is hand-picked. A movie is tested only if it has at least 1 early and at least L late ratings (L = 28 main, 8 and 61 as sensitivity). Candidates: raw mean; shrunk mean with k = 7.3 from the Q4.1 curve; shrunk mean with k tuned on one random half and scored on the other; hard weight (raw mean if 28+ early ratings, else the prior). The prior is the mean of the early per-movie means, so nothing from after the cutoff is used. Gap to the raw mean has a paired bootstrap interval (1,000 resamples).

**Results** (L = 28, squared error against the later mean; lower is better)

| Cutoff | Raw | Shrunk k=7.3 | Gap (95% interval) | Hard weight |
|---|---|---|---|---|
| 2007-12 | 0.050 | 0.045 | -0.005 (-0.007 to -0.004) | 0.052 |
| 2015-01 | 0.063 | 0.047 | -0.016 (-0.021 to -0.011) | 0.057 |
| 2016-06 | 0.058 | 0.044 | -0.014 (-0.023 to -0.007) | 0.050 |
| 2018-01 | 0.042 | 0.034 | -0.008 (-0.014 to -0.003) | 0.036 |

- Shrunk k=7.3 beats the raw mean at all 12 combinations of cutoff and L; the interval excludes 0 in 11 (the exception: L=61, 2018 cutoff, upper end 0.000).
- The gain is concentrated where ratings are few. Movies with 1-7 early ratings: error 0.31-1.08 raw, 0.12-0.22 shrunk. With 61+ ratings the two are nearly identical (0.031-0.044 vs 0.031-0.042).
- Ranking (Spearman with the later mean) is almost unchanged at L=28 (0.907 vs 0.907, 0.870 vs 0.875, 0.888 vs 0.887, 0.912 vs 0.914), because 61+-rating movies make up 82% to 95% of the test set. Shrinkage fixes the low-count tail; it does not reorder the popular movies.
- Tuned k: 6.9 to 8.4 at three cutoffs, so tuning adds nothing there. At the 2007 cutoff k=32.8 lowers overall error (0.042 vs 0.045) but is worse for 1-7 early ratings (0.157 vs 0.115) and ranks worse (0.899 vs 0.907).
- Hard weight is worse than shrinkage everywhere and at 16-27 early ratings is 1.7 to 3 times worse than the raw mean (0.14-0.22 vs 0.06-0.09): falling back to the prior throws away usable information. Its ranking is also below raw (0.845 vs 0.870 at the 2015 cutoff).

**Interpretation.** Shrinkage earns its place against the raw-mean baseline; the hard 28-rating weight and the tuned k do not earn theirs. The smooth form of "trust the mean more as ratings grow" predicts later ratings better than the on/off version.

**Limitations**
- Requiring 28+ late ratings selects movies that kept being rated, so the test favours popular movies. Bands below 28 early ratings hold only 14 to 183 movies each, so those comparisons are noisy.
- Only approved movies were tested, and the prior is one global number; per-genre priors were not tested.
- k = 7.3 comes from the all-movie Q4.1 curve.
- It predicts the later MovieLens mean, which is a proxy for quality, not for any one user's taste.

### Decision
**Decided 2026-10-08.** The trusted score is the shrunk mean with k = 7.3 and the global early-mean prior. The `enough_r*` flags stay as separate columns. See `docs/DECISIONS.md`.

Saved outputs: `artifacts/q43_early_counts_v1.parquet`, `artifacts/q43_score_comparison_v1.csv`, `data/processed/approved_movies_credibility_v1.parquet` (see `docs/ARTIFACTS.md`).

---

## Q4.4 Rating floor

**Question.** What counts as a "low rating" for the exclusion rule?

**Answer.** The floor already exists in `04_eda.ipynb`: below μ − 1σ of the movie's genre (most forgiving genre for multi-genre movies), combined with per-genre count levels (Q1 / Q3) and the time test. It is kept. It is rebuilt as `approve_rule` in `src/viberec/tiers.py` and reproduces 04's 33,812 approved movies exactly. **Revised by Adil.** The all-shrunk version (both sides on the shrunk score) did slightly better in Q4.7, but it rescued 3,629 mid-count movies with a low average on few ratings, which Adil wants rejected. The final rule is a hybrid:
- "Too low" is judged on the raw mean, as in 04.
- "Too high" is judged on the shrunk score, so a too-good-to-be-true average from a handful of ratings is no longer rejected.

The hybrid's AUC is equal to or better than 04's raw rule in all 40 comparisons, and at most 0.008 below the all-shrunk rule.

## Q4.5 New-movie "good range"

Deferred. MovieLens ends in 2019, and only 17 rated movies are from 2020 on.

## Q4.6 Sensitivity

**Question.** Do the hand-set multipliers `LOWER_SD = 1` and `UPPER_SD = 2` change the result?

**Method.** Rerun the rule on all 50,046 rated movies with `LOWER_SD` from 0.5 to 1.5 and `UPPER_SD` 1.5, 2, 2.5, 3 or none, on both scores. Count the movies that flip compared with 1 / 2.

**Results.**
- `UPPER_SD` flips at most 599 movies on the raw mean and at most 1 on the shrunk score. It only caught 5.0 averages from one or two ratings, and shrinkage already handles those.
- `LOWER_SD` matters. Each 0.25 step flips 1,200–2,700 movies on the raw mean (26%–42% rejected) and 1,400–6,200 on the shrunk score (19%–36%), with no flat region. Its value is decided by Q4.7.

## Q4.7 Validation

**Question.** Does the rule remove movies that really are worse, and do the three tiers differ?

**Method.** At each Q4.3 cutoff, the rule is run on the early ratings only. The cutoff year stands in for "now" in the time test. Each movie's later mean is the outcome, for movies with at least 8 or 28 later ratings. Approved vs rejected is measured by AUC, the chance that an approved movie ends up rated higher than a rejected one. Tiers are compared with Kruskal-Wallis and pairwise Mann-Whitney tests. The tiers are:
- Tier 1: rejected with a too-low rating.
- Tier 2: rejected for any other reason, which in practice means few ratings with a normal or high rating.
- Tier 3: approved.

**Results.**
- **The rule works.** AUC is 0.70–0.98 with 28+ later ratings and 0.71–0.87 with 8+, at `LOWER_SD = 1`. Approved movies' later median is 3.33–3.43, against 2.06–2.97 for rejected ones.
- **The shrunk score is at least as good as the raw mean.** It is equal or better in 37 of 40 comparisons, and it rejects 24.0% of movies against 32.4%.
- **`LOWER_SD` = 1 stays.** The best value moves with the cutoff (0.5, 0.75, 1.25 and 1.5 at the four cutoffs), and 1 is within 0.04 AUC of the best everywhere.
- **The tiers differ.** Kruskal-Wallis p < 0.001 wherever testable (ε² 0.02–0.14).
  - Tier 1's later mean is about 2.5, below both other tiers everywhere.
  - Tier 2 is slightly below tier 3 with 8+ later ratings (3.11–3.22 against 3.33). With 28+ it is indistinguishable (p 0.57–0.81).

**Interpretation.**
- Excluding tier 1 is justified.
- Tier 2 are the low-count but normally rated movies that 04 rejected. They are about as good as approved movies, so they get reduced weight, not exclusion. That matches PROJECT.md §4.1.

**Final tiers** (hybrid rule, `LOWER_SD = 1`, `UPPER_SD = 2`, all 50,046 rated movies):

| Tier | Movies | Share | Meaning |
|---|---|---|---|
| 1 | 10,452 | 20.9% | Excluded (too-low raw average, not a high count) |
| 2 | 5,191 | 10.4% | Reduced weight (few ratings, normal or high average); the weight is learned by the ranker (Step 10) |
| 3 | 34,403 | 68.7% | Approved: 04's 33,812 plus the 591 "too high on few ratings" movies |

All of 04's 33,802 approved movies stay in tier 3. The earlier all-shrunk tiers (4,001 / 8,026 / 38,019) are kept in `movie_tiers_v1.parquet`.

**Limitations.**
- Only movies that kept being rated have later ratings, so the test favours popular movies. With 28+ later ratings, tier 2 has only 4–421 movies, and none at the 2018 cutoff.
- The later mean is MovieLens's average, a proxy for quality and not any one user's taste.
- The outcome is not adjusted for genre.
- `MARGIN_YEARS = 8` cannot be tested on this data.

Saved outputs: `artifacts/q46_sensitivity_v1.csv`, `artifacts/q47_validation_v2.csv`, `artifacts/q47_tier_validation_v2.csv` and `data/processed/movie_tiers_v2.parquet` (see `docs/ARTIFACTS.md`).
