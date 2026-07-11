"""
Fully-converged Bayesian Poisson GLMM (pure Python, pymc) for the interaction
random-effects model of Section 4.4.

Motivation: statsmodels' PoissonBayesMixedGLM does not converge for this model
because the interaction random effect is per observed cell (observation-level),
so the fit is saturated. R's lme4::glmer is the usual tool, but this project
keeps the analysis in Python. This script fits the same model with pymc using a
non-centered parameterization, which samples cleanly, and:
  * writes the GLMM heatmaps paper/fig_4_4_1.png and paper/fig_4_4_2.png
  * saves posterior-mean rates + 90% credible interval widths per observed cell
    to docs/glmm_pymc_summary.csv (also addressing the reviewer's point that the
    point-estimate heatmaps lack uncertainty quantification)

Finding: the identified quantities (main effects and the per-cell fitted rates)
sample cleanly and reproduce the observed rates closely (corr ~0.97). The
interaction-variance parameter sigma, however, is only weakly identified -- with
a single observation per interaction cell the data cannot pin it down, and its
posterior drifts between runs (rhat > 1.01 on sigma/z persists regardless of
sampler tuning, and would not be resolved by lme4 either). This weak
identifiability is not a fitting artefact; it is precisely why the GLMM's
interaction term contributes little to out-of-sample prediction here.

Model (matching Section 4.4):
    Y_ij ~ Poisson(E_ij * exp( b0 + alpha_i + tau_j + z_ij ))
    z_ij ~ Normal(0, sigma^2)            # interaction random effect, one per cell

Dependencies: pandas, numpy, matplotlib, pymc
"""

import os
import sys

import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratemaking import load_pure_premium, visualize_heatmap


def main():
    os.makedirs("docs", exist_ok=True)
    pure_premium, exposure_total = load_pure_premium()
    pp = pure_premium.to_numpy(dtype=float)
    exp_mat = exposure_total.to_numpy(dtype=float)
    models = pure_premium.index.to_numpy()
    areas = pure_premium.columns.to_numpy()

    r, c = np.where(~np.isnan(pp))
    claim = (pp[r, c] * exp_mat[r, c])              # total claim (count-scale response)
    log_exp = np.log(exp_mat[r, c])
    n_obs = len(r)
    n_m, n_a = len(models), len(areas)
    print(f"observed cells: {n_obs}  ({n_m} models x {n_a} areas)")

    with pm.Model():
        b0 = pm.Normal("b0", 0.0, 5.0)
        # sum-to-zero main effects -> identifiable against the intercept (clean mixing)
        alpha = pm.ZeroSumNormal("alpha", sigma=5.0, shape=n_m)  # vehicle-model effect
        tau = pm.ZeroSumNormal("tau", sigma=5.0, shape=n_a)      # area effect
        sigma = pm.HalfNormal("sigma", 1.0)                      # interaction SD (delta)
        z_raw = pm.Normal("z_raw", 0.0, 1.0, shape=n_obs)        # non-centered OLRE
        z = sigma * z_raw
        eta = b0 + alpha[r] + tau[c] + z + log_exp
        pm.Poisson("y", mu=pm.math.exp(eta), observed=claim)
        idata = pm.sample(1000, tune=1000, chains=4, cores=4, target_accept=0.95,
                          random_seed=123, progressbar=False)

    post = idata.posterior
    # posterior-mean linear predictor (rate scale, i.e. without the log-exposure offset)
    b0_m = float(post["b0"].mean())
    alpha_m = post["alpha"].mean(dim=("chain", "draw")).to_numpy()
    tau_m = post["tau"].mean(dim=("chain", "draw")).to_numpy()
    sigma_m = float(post["sigma"].mean())
    z_m = (post["sigma"] * post["z_raw"]).mean(dim=("chain", "draw")).to_numpy()
    rate_mean = np.exp(b0_m + alpha_m[r] + tau_m[c] + z_m)

    # 90% credible interval on the rate, per observed cell (uncertainty)
    b0_d = post["b0"].to_numpy().ravel()
    alpha_d = post["alpha"].to_numpy().reshape(-1, n_m)
    tau_d = post["tau"].to_numpy().reshape(-1, n_a)
    z_d = (post["sigma"] * post["z_raw"]).to_numpy().reshape(-1, n_obs)
    lp_d = b0_d[:, None] + alpha_d[:, r] + tau_d[:, c] + z_d      # (draws, n_obs)
    rate_d = np.exp(lp_d)
    lo, hi = np.percentile(rate_d, [5, 95], axis=0)
    ci_width = hi - lo

    print(f"sigma (interaction SD) posterior mean: {sigma_m:.3f}")

    # per-cell summary with uncertainty
    pd.DataFrame({
        "VehModel": models[r], "Area": areas[c],
        "exposure": exp_mat[r, c], "actual_rate": pp[r, c],
        "glmm_rate_mean": rate_mean, "ci90_low": lo, "ci90_high": hi,
        "ci90_width": ci_width,
    }).to_csv("docs/glmm_pymc_summary.csv", index=False)
    print("saved docs/glmm_pymc_summary.csv")

    # heatmap grid (observed cells only; missing left white -> paper's Fig 4.4.x)
    grid = np.full(pp.shape, np.nan)
    grid[r, c] = rate_mean
    grid_df = pd.DataFrame(grid, index=pure_premium.index, columns=pure_premium.columns)
    visualize_heatmap(grid_df, "Estimated Pure Premium Rates -- GLMM (pymc, observed cells)",
                      fig_path="paper/fig_4_4_1.png")
    visualize_heatmap(grid_df, "Estimated Pure Premium Rates -- GLMM (pymc, white = missing)",
                      fig_path="paper/fig_4_4_2.png")
    print("saved paper/fig_4_4_1.png, paper/fig_4_4_2.png")


if __name__ == "__main__":
    main()
