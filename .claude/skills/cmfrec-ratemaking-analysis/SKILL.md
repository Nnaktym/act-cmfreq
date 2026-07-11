---
name: cmfrec-ratemaking-analysis
description: Use when conducting, running, extending, or reproducing a matrix-factorization insurance ratemaking analysis with the cmfrec library (R or Python) — fitting CMF/MF models to pure-premium data, tuning k and lambda, comparing against GLM/GLMM, adding side information (CMF), or writing up the analysis into a paper. Work as an actuary AND data scientist.
---

# cmfrec Ratemaking Analysis & Write-up

## Overview

End-to-end workflow for estimating class rates with (Collective) Matrix Factorization via
`cmfrec`, then writing the result up to paper standard. Work as **both** an actuary (exposure,
credibility, defensible rates) and a data scientist (valid estimation, honest validation, reproducibility).

**Core principle: the analysis must survive its own review.** This repo has a companion review skill
(`review-actuarial-paper`) that flags the failure modes below. Build the analysis so it passes that
review the first time — do not reproduce the flaws it catches. Every number that goes in the paper
must come from code you actually ran, on the same test cells, with the same metric.

**cmfrec API details:** see `cmfrec-reference.md` in this skill directory (R `CMF` vs Python `CMF`,
side-info CMF, weights, gotchas). **Reusable code already exists** in `src/cmf.R` — prefer extending
those helpers over rewriting the pipeline.

## When to Use

- Running or reproducing the Brazilian auto-insurance MF analysis (`src/brazil_data_analysis_*.R/.py`)
- Fitting a new `cmfrec` model, tuning `k`/`lambda`, or adding side information (vehicle group, pop density)
- Comparing MF against GLM/GLMM baselines, or producing the figures/tables for the paper
- Drafting or revising the paper's Applications section from analysis output

Not for: reviewing/critiquing an existing paper (use `review-actuarial-paper` instead).

## Analysis pipeline

1. **Build the matrix.** Aggregate claims and exposure to a `model × region` matrix
   (`get_total` in `src/cmf.R`). Target = pure premium = `claim_total / exposure_total`.
   Apply the exposure filters (cell exposure ≥ 100; row/model total exposure ≥ 10). Low-exposure
   cells become `NA` (missing, to be imputed) — **never 0**.
2. **Split first, then tune.** Hold out the test set (`train_test_split`, cell-level) **before**
   the CV grid search, so tuned `k`/`lambda` never see test cells. `optimize_params` runs the
   k-fold CV grid — run it on the *training* matrix only.
3. **Fit with `cmfrec`.** `CMF(X, k, lambda, method="als", nonneg=TRUE, center=FALSE, weight=…)`.
   See reference file. Refit on the full matrix only for the final all-cells heatmap.
4. **Evaluate honestly.** Predict held-out cells; report metrics. Then re-fit and compare GLM,
   GLMM, and MF **on the identical test cells** and put the numbers in one table.
5. **Visualize.** Predicted-vs-true scatter (`visualize_scatter_plot`) and actual/estimated
   heatmaps (`visualize_heatmap`) — produce the same diagnostics for every model, not just MF.
6. **(Optional) Collective MF.** Add side information (`U`, `I`) to help sparse/cold cells — vehicle
   group and urban/rural population density (Python path). Report whether it actually improves hold-out error.

## Correctness guardrails (bake these in)

- **Weight by exposure.** GLM/GLMM here use `offset(log(exposure))`; the MF loss must match that
  evidence weighting via `weight=` on `CMF` (the project's own `cmf.r:31` did this, the main analysis
  dropped it). Unweighted MF gives a cell of exposure 100 the same say as one of exposure 50,000 —
  actuarially wrong and makes the comparison unfair.
- **One comparison table, comparable metrics.** Report exposure-weighted RMSE **and** Poisson/Tweedie
  deviance for GLM vs GLMM vs MF on the same held-out cells. RMSE in raw currency alone favors big cells
  and isn't the loss the GLMs optimize.
- **No leakage.** Filtering and the CV grid must not touch test cells. Fix seeds (`123`) and record them.
- **Right target family.** Pure premium is non-negative and heavy-tailed; consider Tweedie or a
  frequency-severity split rather than Gaussian L2 on raw pure premiums. Justify the choice in text.
- **Non-negativity caveat.** `nonneg=TRUE` constrains factors but not biases, so predictions can go
  negative and the interaction can only push rates up; note/clip as appropriate.
- **Validate what you claim.** If you claim MF imputes *truly missing* cells better, test it — mask a
  block of observed high-exposure cells and score recovery. Do not sell an unfalsifiable heatmap.
- **R/Python parity.** Keep the two implementations agreeing on split, weighting, metric, and
  hyperparameters; declare which one produced the reported figures.

## Writing up

Match the paper's existing structure: Introduction → Existing Methods (GLM, GLMM) → MF/CMF Method →
Applications (dataset, GLM, GLMM, MF results) → Conclusion & Future Work. For each results subsection:

- State the model equation, the estimation settings actually used, and the hyperparameters from CV.
- Attach every claim to a number or figure. "MF captures interactions / handles missing cells /
  is interpretable" each need explicit evidence — a table, a masked-cell recovery test, or a stated
  meaning for the latent factors — not just a heatmap's appearance.
- Cite the method honestly: biased MF = Koren et al. (2009); the tool = `cmfrec` (Cortes); the data =
  CASdatasets brvehins1 (Dutang & Charpentier). Keep figure/citation numbering consistent.
- Regenerate figures from the final code so paper and repo agree.

## Common mistakes

- Filling missing pure-premium cells with 0 before fitting (0 is a real cheap rate, not "unknown").
- Tuning `k`/`lambda` on data that includes the test cells, then reporting an optimistic hold-out RMSE.
- Comparing an unweighted MF against exposure-weighted GLM/GLMM and calling MF better.
- Reporting a scatter plot for MF only, with no head-to-head error table.
- Choosing `k` near the smaller matrix dimension (over-parameterizing a small grid) without a k-vs-error curve.
- Letting the R and Python pipelines drift so the paper's numbers can't be reproduced.
