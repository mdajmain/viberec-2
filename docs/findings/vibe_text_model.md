# Step 5: predicting vibes from text without an LLM

**Notebook:** `notebooks/06_vibe_features.ipynb`, section 5. **Code:** `src/viberec/text_model.py`, `src/viberec/vibe_eval.py`.

## Question
Only 12,877 of 34,403 approved movies have MovieLens Tag Genome vibe scores. Can the rest get vibe scores from their text, for free, without paying an LLM to read every plot?

## Why it matters
Vibe is the core of VibeRec. Without vibe scores, two-thirds of the approved catalog can only be recommended by genre and ratings.

## Method
- **Data:** 12,356 movies with both a Wikipedia plot and Genome scores, split 80/20 (seed 42): 9,845 train, 2,511 test.
- **Target:** the 41 vibe attributes of the taxonomy (`artifacts/genome_vibe_map_v1.csv`) that the Genome can check.
- **Model:** TF-IDF features (sublinear term frequency, single words and two-word phrases) + ridge regression, one weight per word per vibe.
- **Tuning:** TF-IDF options and the ridge penalty were chosen by 3-fold cross-validation on the training movies, scored by mean Spearman.
- **Contenders:**
  - genre-only guess (baseline);
  - plot only;
  - plot + Reception;
  - plot + Reception with the review-score sentences removed.
- **Measure:** per-vibe Spearman on the test movies, with paired bootstrap intervals for differences.
- **Quality-leak check:** correlation of each model's scores with the movie's average rating, against the same correlation for the true Genome scores.

## Results
| Model | Median Spearman (41 vibes) |
|---|---|
| Genre-only | 0.370 |
| Plot | 0.547 (beats genre on 36/41, worse on 1) |
| **Plot + Reception** | **0.609** (beats plot-only on 41/41) |
| Plot + Reception, no score sentences | 0.600 (worse than plot + Reception on 33/41) |

- **Best vibes:** dark_bleak 0.77, melancholic 0.77, slow 0.76, atmospheric 0.74.
- **Weakest vibes:** christmas 0.29, non-hollywood ending 0.39, space and future 0.41.
- **Quality leak:** the model's words for *slow* and *melancholic* include "universal acclaim" and "Cannes". The Genome itself links these vibes to acclaimed films; atmospheric correlates 0.54 with rating in the true scores. The model follows rating no more than the truth does (median excess 0.05). Removing the score sentences moved the link to other words ("festival") without reducing it.

## Interpretation
- A free text model recovers a large part of the Genome's vibe signal and beats genre on every vibe.
- Critics' Reception text adds signal on every vibe, which supports using reviews as a "how it feels" source.
- The chosen model is plot + Reception.

## Limitations
- **It learns the Genome's view,** including the tie between some vibes and acclaim, so vibe scores carry some quality signal; Step 10 should account for it.
- **Tested on better-known movies** (all with Genome data); obscure movies have shorter text, so expect lower accuracy there.
- **Rare vibes are weak.**
- **No written vibe paragraph,** unlike the LLM route.
- **No scores yet** for 7,539 tier 2 + 3 movies with no Wikipedia text.

## Decision
Adopted (2026-10-08). Saved:
- `data/processed/vibe_scores_v1.parquet`: Genome scores for 12,877 movies, model predictions for 19,178.
- `models/tfidf_ridge_plot_reception_v1.joblib`.

## Follow-up tests (2026-10-09)
- **Adil's labels (99 movies, leave-one-out):**
  - Light↔Heavy 0.89 vs genre 0.80;
  - Pacing as `fast − slow` 0.69 vs 0.27;
  - Ending 70% vs always-happy 58%; bittersweet is never recognised.
  - Text-model scores on unseen movies do as well on Light↔Heavy, and weaker on Ending (not clearly above baseline).
- **Embeddings (bge-small):**
  - as ridge features 0.556, worse than TF-IDF on 41/41;
  - zero-shot vector search 0.22, below genre.
  - TF-IDF stays. Raw document similarity does not capture vibe, so Step 9 should parse vibe requests into vibe dimensions.
- **TMDB overview for movies without Wikipedia text:** overview only 0.372 (≈ genre); overview + keywords 0.450 (beats genre on 22/41). Used to fill gaps, with its own `source` label.
