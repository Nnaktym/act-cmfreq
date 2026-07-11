# Matrix Factorization for Class Ratemaking

Applying (Collective) Matrix Factorization to insurance class ratemaking, and
benchmarking it honestly against a GLM and a GLMM. Companion code for the ICA2026
paper *Matrix Factorization for Modeling High-Dimensional Interactions in Class
Ratemaking* (Kato, Fujita, Nomura).

The analysis is now maintained in **Python** (the R scripts are kept for
reference). Data: `brvehins1` (Brazilian auto insurance) from CASdatasets,
filtered to Honda, aggregated to a 48 vehicle-model × 40 region matrix of pure
premiums (468 observed cells with exposure ≥ 100; the rest are treated as
missing and imputed).

## Repository layout

```text
paper/          ICA2026 paper (Markdown source + figures fig_4_2_1 .. fig_4_5_2 + PDF)
data/           brvehins_org.csv (Honda raw export), all_data.csv, population density
src/
  ratemaking.py               shared library: data load/aggregation, split/CV,
                              metrics, visualisation (ported from cmf.R)
  brazil_data_analysis_R.py   MAIN analysis (Python): MF vs GLM vs GLMM comparison
                              + regenerates the paper's MF/GLM figures
  glmm_pymc.py                fully-converged Bayesian Poisson GLMM (pymc) +
                              GLMM figures (fig_4_4_1/2) + per-cell uncertainty
  brazil_data_analysis_R.R    original R analysis (reference)
  brazil_data_analysis_R.ipynb  original R notebook (reference)
  cmf.R                       original R helpers (reference)
  brazil_data_analysis_python.py  earlier side-info CMF experiment (reference)
docs/           analysis outputs (comparison tables, per-cell predictions, this
                doc set) — see also notebook_paper_correspondence_check.md (historical)
```

> Note on naming: `src/brazil_data_analysis_R.py` is a **Python** file (the `_R`
> suffix reflects its origin as a translation of `brazil_data_analysis_R.R`).

## Setup

```bash
pip install cmfrec statsmodels pymc pandas numpy matplotlib
```

## How to run

```bash
# 1) MF vs GLM vs GLMM on one hold-out set + regenerate MF/GLM paper figures
python3 src/brazil_data_analysis_R.py

# 2) Fully-converged Bayesian GLMM + GLMM paper figures (fig_4_4_x) + uncertainty
python3 src/glmm_pymc.py
```

Outputs land in `docs/` (comparison tables, per-cell predictions) and `paper/`
(regenerated figures). Seeds are fixed (`123`); R and Python RNGs differ, so
numbers won't match the R scripts to the decimal.

## Method & correctness notes

- **Exposure-weighted MF.** The MF loss is weighted by exposure (`W=`, normalized
  to mean 1) so it matches the `offset(log(exposure))` weighting of the GLM/GLMM —
  a cell of exposure 100 no longer counts the same as one of 50,000.
- **One comparison, same cells.** MF, GLM and GLMM are all scored on an *identical*
  held-out set of observed cells, with RMSE, exposure-weighted RMSE, and Poisson
  deviance side by side.
- **No leakage.** The test set is held out *before* the CV grid search over
  (k, λ). Tuning is exposure-weighted **and non-centered (`center=False`) to match
  the final fit exactly** — otherwise the tuned λ would be calibrated for a
  different (mean-centered) model than the one deployed.
- **R scripts are historical/unweighted.** `cmf.R` / `brazil_data_analysis_R.R`
  fit the MF *without* exposure weighting (no `W=`); the exposure-weighted
  comparison the paper reports is the **Python** pipeline only. Treat the R files
  as the earlier, unweighted reference implementation.
- **GLMM.** `glmm_pymc.py` fits `Y_ij ~ Poisson(E_ij·exp(b0+α_i+τ_j+z_ij))`,
  `z_ij ~ N(0, σ²)` with pymc (non-centered, sum-to-zero main effects). The
  interaction variance σ is only weakly identified — one observation per
  interaction cell — which is itself why the GLMM cannot exploit interactions
  out-of-sample. `statsmodels`' saturated OLRE fit does not converge; the main
  script uses it only for the test-set comparison column (where the GLMM reverts
  to main effects anyway).

## Key finding

On the identical held-out cells, with exposure weighting applied consistently:

| Model | RMSE | Exposure-weighted RMSE | Poisson deviance |
|---|---:|---:|---:|
| GLM (main effects) | 328.2 | 271.0 | 4.74×10⁶ |
| GLMM (random interaction) | 328.2 | 271.0 | 4.74×10⁶ |
| Matrix Factorization | 354.4 | 426.6 | 1.14×10⁷ |

Selected hyperparameters: `k=3`, `λ=100` (strong regularization; the CV surface is
nearly flat in `k`). The GLMM ties the GLM because every held-out cell is an
unobserved interaction, so its random effect reverts to zero (this is the paper's
GLMM limitation, shown numerically). Stratifying by exposure, MF is near-parity
with the GLM in the **sparse** stratum (wRMSE 316.4 vs 310.3) but well behind on
**dense** cells (441.5 vs 264.2).

The main-effects GLM is most accurate on **observed** cells; MF is *competitive
with*, not uniformly superior to, the conventional models there — and most
competitive where exposure is thin. Matrix factorization's distinctive value is
**structural** — it produces interaction-aware estimates for the sparse and
missing cells that the GLMM cannot resolve (it reverts to main effects) — rather
than a uniform accuracy gain. See the paper's Section 4.6 and Conclusion for the
full discussion.
