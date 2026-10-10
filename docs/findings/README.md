# Findings index

Analysis questions explored outside the main pipeline notebooks, each with its own write-up.

| Question | Status | Write-up |
|---|---|---|
| Do movie ratings vary by genre? | Analysis complete, decision: genre-specific — Adil, 2026-10-06 | [genre_ratings.md](genre_ratings.md) |
| Step 4, Q4.1: how many ratings until a movie's mean is trustworthy? | Decided: reliability 0.8 (about 27 ratings), one global count | [tiers.md](tiers.md#q41-credibility-how-many-ratings-until-a-movies-mean-is-trustworthy) |
| Step 4, Q4.2: age boundaries | Deferred (future work) | [tiers.md](tiers.md) |
| Step 4, Q4.3: raw mean or shrunk mean as the trusted score? | Decided: shrunk mean, k = 7.3 | [tiers.md](tiers.md#q43-trusted-rating-measure-raw-mean-or-shrunk-mean) |
| Step 4, Q4.4–Q4.7: rating floor, new-movie range, sensitivity, validation | Done: tiers 1/2/3 = 10,452 / 5,191 / 34,403 (hybrid rule); Q4.5 deferred | [tiers.md](tiers.md) |
| Step 5: predicting vibes from text without an LLM | Done: TF-IDF + ridge on plot + Reception, median Spearman 0.61 vs genre 0.37 | [vibe_text_model.md](vibe_text_model.md) |
