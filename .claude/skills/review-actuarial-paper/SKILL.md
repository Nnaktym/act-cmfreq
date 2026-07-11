---
name: review-actuarial-paper
description: Use when reviewing, critiquing, refereeing, or giving feedback on this repo's actuarial ratemaking paper (matrix factorization / CMF for class ratemaking, ICA2026) or a similar paper that applies ML methods (MF, GLM, GLMM, neural nets, credibility) to insurance pricing. Review as an actuary AND data scientist.
---

# Reviewing an Actuarial ML Ratemaking Paper

## Overview

Review as **both** an actuary (pricing soundness, exposure/credibility, regulatory accountability) **and** a data scientist (estimation, validation, leakage, metrics, reproducibility). A finding is only worth writing if it would change a referee's accept/revise decision or materially improve the paper.

**Core principle: every claim must be traced to evidence.** The paper's prose says the method wins; the figures, the math, and the code in this repo say whether it actually does. When they disagree, the code wins. Do not accept a superiority claim that has no quantitative comparison behind it.

**This repo's paper:** `paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking.md` (source) and the `.pdf` (rendered equations — read the PDF when the markdown math looks mangled). Method code: `src/brazil_data_analysis_R.R`, `src/brazil_data_analysis_python.py`, `src/cmf.R`, `cmf.r`. Data: `data/`.

## When to Use

- Asked to review / referee / critique / "give feedback on" the paper
- Asked whether the claims hold, or how to strengthen it before submission
- A similar paper applying MF/CMF/GLM/GLMM/credibility/neural methods to ratemaking

Not for: writing new paper content, editing the model code, or copy-editing only (unless asked).

## Method

1. **Orient.** Read the paper end to end (PDF for equations). Build a claims→evidence map: for each headline claim (esp. "MF beats GLM/GLMM", "handles missing cells", "captures non-linear interactions", "interpretable"), note exactly which figure/table/number is offered as proof. Flag every claim with no evidence attached.
2. **Run the four lenses** below. For each, produce concrete findings, not vibes.
3. **Verify against the code and data.** Claims about weighting, splitting, CV, metrics, and hyperparameters are checkable in `src/`. Open the file and confirm before asserting a methodology flaw — quote the line.
4. **Write the report** in the output format below. Severity-rank. Separate blocking issues from polish.

## Lens A — Actuarial soundness

- **Exposure weighting / credibility.** Ratemaking's core: a cell built from exposure 100 is not equal evidence to a cell from exposure 50,000. Check whether the loss actually weights by exposure. In this repo the GLM/GLMM use `offset(log(exposure))` (correctly exposure-weighted Poisson), but the MF fits `pure_premium` with an **unweighted** squared-error loss — no `weight=` on `CMF`. That asymmetry alone can explain apparent differences and is the highest-value finding. cmfrec supports per-observation weights; ask why they weren't used.
- **Target & distribution.** Pure premium (claim cost) is non-negative and heavy-tailed. An L2/Gaussian loss on raw pure premiums is dominated by large cells and mis-models the tail. Poisson on *total claim amount* (not counts) is also questionable — flag whether Tweedie / frequency-severity would be the right family.
- **Uncertainty & accountability.** Ratemaking needs credibility/CIs and an auditable rationale per rate. Point-estimate heatmaps with no uncertainty are a regulatory gap. Is "interpretability" actually demonstrated (what do the latent factors *mean*?) or only asserted?
- **The imputation claim is unfalsifiable as tested.** The paper's key selling point is estimating *truly missing* cells better than GLMM. But validation is on held-out *observed* cells — there is no ground truth for the missing cells, so the central claim is not empirically supported. Say so.

## Lens B — Statistical / ML methodology

- **Fair baselines.** The GLM is deliberately main-effects-only (no interactions), then faulted for missing interactions. That is a straw baseline. A fair comparison needs a regularized-interaction GLM or the fused-lasso the paper itself cites (Takahashi & Nomura, 2023).
- **The missing comparison table.** Superiority is claimed but only MF gets a predicted-vs-true scatter (Fig 4.5.1). There is no single table of hold-out RMSE (or Poisson deviance / MAE) for GLM vs GLMM vs MF on the *same* test cells. Without it, "MF wins" is unsupported. This is usually the #1 revision.
- **Metric comparability.** RMSE in currency on raw pure premiums favors whichever model the large cells like, and isn't the loss GLM/GLMM optimize. Recommend exposure-weighted and deviance-based metrics so all three are judged on comparable footing.
- **Validation integrity.** Confirm in code: is filtering (exposure ≥ 100, model exposure ≥ 10) applied *before* the split (leakage risk)? Is the 25% hold-out disjoint from the 4-fold CV grid search (nested vs reused)? Are seeds fixed (they are, `123`)? Is `k` (e.g. 22) sensibly bounded by the matrix dimensions?

## Lens C — Reproducibility & data

- Can the pipeline be re-run? `CASdatasets` needs a `git clone`; `data/*.csv` is provided. Check R and Python paths agree, and that reported numbers (k, λ, RMSE) match what the code produces.
- Check for hidden inconsistencies between the R and Python implementations (they should agree on split, weighting, metric).

## Lens D — Exposition & scholarship

- **Math correctness/rendering:** verify equations in the PDF; the biased-MF objective, the "Frobenius norm over observed elements" phrasing, and the SVD-on-sparse claim need care.
- **Missing citations:** biased matrix factorization (Koren et al., 2009), the `cmfrec` library (Cortes), and Tweedie/GLM-for-pricing references.
- **Internal consistency:** citation-year mismatches (e.g. "Xie et al. (2024)" in text vs "2025" in references), stray "Figure 4" references, references listed but never cited (Norberg 1993).

## Output format

```
## Review: <paper title>
**Recommendation:** Accept / Minor revision / Major revision / Reject — one line why.

**Summary** (2–3 sentences): what the paper does and its core contribution, stated fairly.

**Major issues** (blocking; severity-ranked)
1. <issue> — why it matters — where (§/fig/file:line) — suggested fix.

**Minor issues** (non-blocking): bullet list.

**Strengths**: what genuinely works — keep the review honest and usable.

**Concrete revision checklist**: the smallest set of changes that would move this to acceptance.
```

## Common mistakes (reviewer failure modes)

- Accepting "MF captured interactions" from a heatmap's appearance instead of demanding a head-to-head error table.
- Reviewing only the prose and skipping `src/` — the weighting/leakage/metric truth is in the code, not the paper.
- Copy-editing while missing the structural flaw (asymmetric exposure weighting, straw baseline).
- Listing everything flat — no severity ranking makes the review unusable. Lead with what blocks acceptance.
- Being either a cheerleader or a demolisher. Give a fair summary and real strengths alongside the blocking issues.
