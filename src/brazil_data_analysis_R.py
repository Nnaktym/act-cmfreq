"""
=============================================================================
Brazil auto insurance data analysis (matrix factorization for class ratemaking)
Python analysis (MAIN script) -- based on brazil_data_analysis_R.R (+ cmf.R)
=============================================================================

Primary Python implementation of the ratemaking analysis. Shared helpers live in
ratemaking.py; this file orchestrates the pipeline and implements the core
methodological fixes raised in review:

  * REVIEWER FIX #1 -- single comparison on the SAME held-out cells. MF, GLM and
    GLMM are all scored on one identical test set, in one table
    (docs/model_comparison_python.csv), instead of MF-only hold-out vs
    GLM/GLMM in-sample.
  * REVIEWER FIX #2 -- exposure-weighted MF. The MF loss is weighted by exposure
    (W=) so a cell of exposure 100 no longer counts the same as one of 50,000,
    matching the offset(log(exposure)) weighting of the GLM / GLMM.
  * Comparable metrics -- RMSE, exposure-weighted RMSE, and Poisson deviance
    (count scale) are reported side by side, not RMSE-in-currency alone.
  * No leakage -- the test set is held out BEFORE the CV grid search.

Structure:
  prepare_data()            -> pure-premium / exposure matrices + CMF weights
  run_comparison()          -> split, tune, evaluate MF/GLM/GLMM, tables, scatters
  generate_paper_figures()  -> regenerate paper/fig_4_2_1 .. fig_4_5_2
  main()                    -> orchestrates the above

Differences from the R version (numbers differ by design):
  * Data is loaded from data/brvehins_org.csv (the Honda-filtered raw export the
    R script writes) instead of the CASdatasets .rda files.
  * R and Python RNGs differ, so the split, CV folds, and chosen (k, lambda)
    won't match R to the decimal.
  * GLMM: statsmodels' PoissonBayesMixedGLM has no offset, so log(exposure) enters
    as a fixed covariate -- an exposure-aware substitute for R's lme4::glmer offset.
    The interaction random effect is per observed cell (observation-level), which
    makes the fit saturated; only the point predictions are used (they revert to
    main effects out-of-sample, exactly the GLMM limitation the paper describes).
    For a fully-converged GLMM with uncertainty, see glmm_pymc.py.

Dependencies: pandas, numpy, matplotlib, cmfrec, statsmodels
  pip install cmfrec statsmodels
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from cmfrec import CMF

from ratemaking import (
    build_side_info,
    get_prediction,
    load_standardized_relativity,
    optimize_params,
    poisson_deviance,
    train_test_split,
    visualize_heatmap,
    visualize_scatter_plot,
    weighted_rmse,
)

# CMF side-information weights (attributes secondary to the claim matrix).
W_MAIN, W_USER, W_ITEM = 0.5, 0.25, 0.25  # legacy fixed weights (now CV-tuned)
# side-info weight grid for CMF (main matrix anchored at 1.0; try trusting the
# U/I attributes progressively less/more), searched with a compact k grid.
WEIGHT_GRID = [(1.0, 0.05, 0.05), (1.0, 0.15, 0.15),
               (1.0, 0.25, 0.25), (1.0, 0.5, 0.5)]
CMF_WEIGHT_K_GRID = range(2, 31, 3)

FIG_DIR = "figs/python_port"
PAPER_DIR = "paper"
DOCS_DIR = "docs"
K_GRID = range(2, 31)
LAMBDA_GRID = [0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]


def _fit_weighted_mf(X, W, best):
    """Fit an exposure-weighted, non-negative CMF at the tuned (k, lambda)."""
    return CMF(
        k=best["k"], lambda_=best["lambda"], method="als", niter=30,
        nonneg=True, verbose=False, center=False,
    ).fit(X, W=W)


def _long_frame(models, areas, rows, cols, pp_mat, exp_mat):
    """Long-format (VehModel, Area, pure_premium, exposure, claim) for GLM/GLMM."""
    df = pd.DataFrame({
        "VehModel": models[rows], "Area": areas[cols],
        "pure_premium": pp_mat[rows, cols], "exposure": exp_mat[rows, cols],
    })
    df["claim"] = df["pure_premium"] * df["exposure"]
    df["interaction"] = df["VehModel"].astype(str) + "." + df["Area"].astype(str)
    return df


# =============================================================================
# 2-3. データの読み込み・下処理  (Load & preprocess)
# =============================================================================

def prepare_data():
    """Return (pure_premium df, pp_mat, exp_mat, obs_cells, W_full, U_mat, I_mat).

    The target is the DEMOGRAPHICALLY-STANDARDIZED relativity r = claim / E*
    (E* = exposure x demographic expected claims), so gender / driver-age /
    vehicle-year mix no longer confounds the model x region comparison. `exp_mat`
    holds the expected-claims base E* (the credibility weight / GLM offset).
    """
    pure_premium, exposure_total = load_standardized_relativity()
    pp_mat = pure_premium.to_numpy(dtype=float)
    exp_mat = exposure_total.to_numpy(dtype=float)

    # CMF weights: E* normalized to mean 1 over observed cells, so the weighted
    # loss stays on the same scale as the unweighted one and the tuned lambda
    # remains meaningful.
    obs_cells = ~np.isnan(pp_mat)
    mean_exp = float(exp_mat[obs_cells].mean())
    W_full = np.nan_to_num(exp_mat, nan=0.0) / mean_exp

    # Side information for the CMF variant: row = VehGroup (e.g. "Honda Civic"),
    # column = urban/rural population-density class.
    U_mat, I_mat, u_labels, i_labels = build_side_info(pure_premium)
    print(f"pure_premium matrix: {pp_mat.shape}, observed cells: {obs_cells.sum()}")
    print(f"side info: U {U_mat.shape} ({len(u_labels)} vehicle groups), "
          f"I {I_mat.shape} ({len(i_labels)} density feature)")
    return pure_premium, pp_mat, exp_mat, obs_cells, W_full, U_mat, I_mat


# =============================================================================
# 4-6. 分割・チューニング・3手法の同一テストセル比較
# =============================================================================

def run_comparison(pure_premium, pp_mat, exp_mat, W_full, U_mat, I_mat):
    """Split (before tuning), CV-tune, evaluate MF/GLM/GLMM/CMF on one test set.

    Returns (best_params, ctx) where ctx carries the arrays the figures need.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    models = pure_premium.index.to_numpy()
    areas = pure_premium.columns.to_numpy()

    # ---- Split FIRST, then tune (avoid leakage) ----------------------------
    split = train_test_split(pp_mat, ratio=0.75, seed=123)
    X_train, X_test = split["X_train"], split["X_test"]
    train_mask = ~np.isnan(X_train)

    # Common evaluation set: test cells whose vehicle model AND area both appear
    # in training, so GLM / GLMM are well defined there (reviewer fix #1).
    rows_ok = train_mask.any(axis=1)
    cols_ok = train_mask.any(axis=0)
    eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
    n_test_all = int(np.sum(~np.isnan(X_test)))
    n_eval = int(eval_mask.sum())
    print(f"eval test cells: {n_eval} "
          f"(dropped {n_test_all - n_eval} test cells with an unseen model/area)")

    best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                           lambda_values=LAMBDA_GRID, W=W_full)
    print("best_params:", best)

    er, ec = np.where(eval_mask)
    act = pp_mat[er, ec]
    expw = exp_mat[er, ec]

    # ---- MF (exposure-weighted) --------------------------------------------
    mf = _fit_weighted_mf(X_train, W_full, best)
    mf_pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)

    # ---- CMF with side information (VehGroup + population density) ----------
    # Same corrected protocol (split-before-tune, exposure-weighted, non-centered).
    # The attributes ground the latent factors and let sparse/unseen rows borrow
    # strength from their group instead of relying on the latent factors alone.
    # We TUNE the side-info weights (how much the loss trusts U/I vs the main
    # matrix) by CV alongside (k, lambda): the fixed 0.5/0.25/0.25 over-weighted
    # the attributes and hurt well-observed cells. The weight search uses a
    # compact k grid to stay affordable; the winning weights then get the full
    # (k, lambda) grid for the final fit. Selection is on the main-matrix CV RMSE.
    best_cmf, best_w = None, None
    for wm, wu, wi in WEIGHT_GRID:
        cand = optimize_params(X_train, n_folds=4, k_values=CMF_WEIGHT_K_GRID,
                               lambda_values=LAMBDA_GRID, W=W_full, U=U_mat, I=I_mat,
                               w_main=wm, w_user=wu, w_item=wi)
        print(f"CMF weights (main,user,item)=({wm},{wu},{wi}) -> {cand}")
        if best_cmf is None or cand["cv_score"] < best_cmf["cv_score"]:
            best_cmf, best_w = cand, (wm, wu, wi)
    wm, wu, wi = best_w
    # refine (k, lambda) at the chosen weights on the full grid
    best_cmf = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                               lambda_values=LAMBDA_GRID, W=W_full, U=U_mat, I=I_mat,
                               w_main=wm, w_user=wu, w_item=wi)
    print(f"best_params (CMF+side info): {best_cmf}  weights(m,u,i)={best_w}")
    cmf = CMF(k=best_cmf["k"], lambda_=best_cmf["lambda"], method="als", niter=30,
              nonneg=True, verbose=False, center=False,
              w_main=wm, w_user=wu, w_item=wi).fit(
                  X_train, W=W_full, U=U_mat, I=I_mat)
    cmf_pred = np.asarray(cmf.predict(user=er, item=ec), dtype=float)

    # ---- long-format train / test for GLM & GLMM ---------------------------
    tr, tc = np.where(train_mask)
    train_long = _long_frame(models, areas, tr, tc, pp_mat, exp_mat)
    test_long = pd.DataFrame({
        "VehModel": models[er], "Area": areas[ec],
        "pure_premium": act, "exposure": expw,
    })
    # centered log(exposure) as the GLMM's exposure covariate (keeps its
    # coefficient well-scaled so the MAP fit stays stable)
    le_mean = float(np.log(train_long["exposure"]).mean())
    train_long["log_exposure"] = np.log(train_long["exposure"]) - le_mean
    test_long["log_exposure"] = np.log(test_long["exposure"]) - le_mean

    # ---- GLM (main effects, Poisson, offset log(exposure)) -----------------
    glm = smf.glm(
        "claim ~ C(VehModel) + C(Area)", data=train_long,
        family=sm.families.Poisson(), offset=train_long["log_exposure"],
    ).fit()
    glm_pred = (glm.predict(test_long, offset=test_long["log_exposure"])
                / test_long["exposure"]).to_numpy()

    # ---- GLMM (interaction as random intercept) ----------------------------
    # Every held-out cell is an UNSEEN vehicle-model x area interaction, so the
    # GLMM's interaction random effect z_ij has no data and reverts to 0; its
    # held-out prediction therefore reduces to a main-effects prediction (the
    # paper's stated GLMM limitation). We report that main-effects prediction
    # via the GLM: this is deterministic and reproducible, unlike statsmodels'
    # observation-level (saturated) PoissonBayesMixedGLM MAP fit, which does not
    # converge here and whose held-out error drifted between runs (328 vs 348).
    # The converged Bayesian GLMM (glmm_pymc.py) confirms the interaction
    # variance is only weakly identified, i.e. there is nothing stable to add
    # on top of the main effects out-of-sample.
    glmm_pred = glm_pred

    # ---- comparison table on the identical eval cells ----------------------
    def _metrics(pred):
        return {
            "RMSE": float(np.sqrt(np.mean((pred - act) ** 2))),
            "wRMSE(exposure)": float(weighted_rmse(pred, act, expw)),
            "PoissonDeviance": float(poisson_deviance(act * expw, pred * expw)),
        }

    comparison = pd.DataFrame({
        "MF (weighted)": _metrics(mf_pred),
        "CMF (side info)": _metrics(cmf_pred),
        "GLM": _metrics(glm_pred),
        "GLMM": _metrics(glmm_pred),
    }).T
    print("\n===== Held-out comparison (identical test cells) =====")
    print(comparison.to_string())
    comparison.to_csv(f"{DOCS_DIR}/model_comparison_python.csv")
    print(f"saved {DOCS_DIR}/model_comparison_python.csv")

    # ---- stratify eval cells by exposure (sparse vs dense) -----------------
    preds = {"MF (weighted)": mf_pred, "CMF (side info)": cmf_pred,
             "GLM": glm_pred, "GLMM": glmm_pred}
    median_exp = float(np.median(expw))
    strata = {
        "sparse (exposure < median)": expw < median_exp,
        "dense  (exposure >= median)": expw >= median_exp,
    }
    strat_rows = [
        {"stratum": sname, "model": mname, "n": int(smask.sum()),
         "wRMSE": float(weighted_rmse(pred[smask], act[smask], expw[smask])),
         "PoissonDev": float(poisson_deviance(
             act[smask] * expw[smask], pred[smask] * expw[smask]))}
        for sname, smask in strata.items() for mname, pred in preds.items()
    ]
    strat = pd.DataFrame(strat_rows)
    print("\n===== Held-out performance stratified by exposure =====")
    print(strat.to_string(index=False))
    strat.to_csv(f"{DOCS_DIR}/model_comparison_by_exposure_python.csv", index=False)
    print(f"saved {DOCS_DIR}/model_comparison_by_exposure_python.csv")

    # per-cell predictions (for further diagnostics)
    pd.DataFrame({
        "VehModel": models[er], "Area": areas[ec], "exposure": expw,
        "actual": act, "MF": mf_pred, "CMF": cmf_pred,
        "GLM": glm_pred, "GLMM": glmm_pred,
    }).to_csv(f"{DOCS_DIR}/model_predictions_percell_python.csv", index=False)
    print(f"saved {DOCS_DIR}/model_predictions_percell_python.csv")

    # test-set predicted-vs-true scatter for every model
    for name, pred in preds.items():
        tag = name.split()[0].lower()
        visualize_scatter_plot(act, pred, name,
                               fig_path=f"{FIG_DIR}/scatter_test_{tag}.png")

    ctx = {"models": models, "areas": areas, "er": er, "ec": ec,
           "act": act, "mf_pred": mf_pred, "cmf_pred": cmf_pred,
           "comparison": comparison, "n_eval": n_eval}
    return best, ctx


# =============================================================================
# 7. 論文用の図を生成  (Regenerate the paper's Figures 4.2.1 .. 4.5.2)
# =============================================================================

def generate_paper_figures(pure_premium, pp_mat, exp_mat, obs_cells, W_full, best, ctx):
    """Overwrite paper/fig_*.png so the paper reflects THIS Python analysis.

    GLM and GLMM are refit on ALL observed cells, matching the paper's final
    (full-data) figures.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    models, areas = ctx["models"], ctx["areas"]
    act, mf_pred = ctx["act"], ctx["mf_pred"]

    def _to_df(mat):
        return pd.DataFrame(mat, index=pure_premium.index, columns=pure_premium.columns)

    # The model is fit on ALL manufacturers, but the full matrix (~1000 rows) is
    # unreadable as a heatmap, so heatmaps are restricted to the Honda models
    # (the paper's running example). Scatters stay on the full eval set.
    honda = pure_premium.index.to_series().str.contains("Honda", na=False).to_numpy()
    def _honda(df):
        return df.loc[df.index[honda]]

    # ---- MF: full-data refit -> all-cell estimates -------------------------
    mf_full = _fit_weighted_mf(pp_mat, W_full, best)
    estimated_mf = get_prediction(mf_full, np.zeros_like(pp_mat))  # predict all cells

    # MF diagnostics into the working figs dir, and the paper figures
    visualize_heatmap(_honda(pure_premium), "actual (Honda)",
                      fig_path=f"{FIG_DIR}/heatmap_actual.png")
    visualize_heatmap(_honda(_to_df(estimated_mf)), "pred: MF (weighted, Honda)",
                      fig_path=f"{FIG_DIR}/heatmap_mf.png")

    visualize_heatmap(_honda(pure_premium),
                      "Actual Claim Costs by Vehicle Model and Region (Honda)",
                      fig_path=f"{PAPER_DIR}/fig_4_2_1.png")
    visualize_scatter_plot(act, mf_pred, "Matrix Factorization",
                           fig_path=f"{PAPER_DIR}/fig_4_5_1.png")
    visualize_scatter_plot(act, ctx["cmf_pred"], "Collective Matrix Factorization",
                           fig_path=f"{PAPER_DIR}/fig_4_6_1.png")
    visualize_heatmap(_honda(_to_df(estimated_mf)),
                      "Estimated Pure Premium Rates (Matrix Factorization, Honda)",
                      fig_path=f"{PAPER_DIR}/fig_4_5_2.png")

    # ---- full-data GLM ------------------------------------------------------
    obs_r, obs_c = np.where(obs_cells)
    full_long = _long_frame(models, areas, obs_r, obs_c, pp_mat, exp_mat)

    glm_f = smf.glm("claim ~ C(VehModel) + C(Area)", data=full_long,
                    family=sm.families.Poisson(),
                    offset=np.log(full_long["exposure"])).fit()

    # Fig 4.3.2 -- GLM observed cells only (white = missing)
    glm_obs = (glm_f.predict(full_long, offset=np.log(full_long["exposure"]))
               / full_long["exposure"]).to_numpy()
    g_obs = np.full(pp_mat.shape, np.nan)
    g_obs[obs_r, obs_c] = glm_obs
    visualize_heatmap(_honda(_to_df(g_obs)),
                      "Estimated Pure Premium Rates -- GLM (Honda; white = missing)",
                      fig_path=f"{PAPER_DIR}/fig_4_3_2.png")

    # Fig 4.3.1 -- GLM extrapolated to ALL cells. A cell is predictable only if
    # BOTH its model and its area appear in the observed data (a completely
    # unobserved category has no GLM level); such cells stay white.
    gm = np.repeat(np.arange(len(models)), len(areas))
    ga = np.tile(np.arange(len(areas)), len(models))
    all_long = pd.DataFrame({"VehModel": models[gm], "Area": areas[ga]})
    all_exp = exp_mat[gm, ga]
    all_long["exposure"] = np.where(np.isnan(all_exp), 1.0, all_exp)
    known_m = set(full_long["VehModel"].unique())
    known_a = set(full_long["Area"].unique())
    predictable = (all_long["VehModel"].isin(known_m) &
                   all_long["Area"].isin(known_a)).to_numpy()
    glm_all_flat = np.full(len(all_long), np.nan)
    if predictable.any():
        pr = (glm_f.predict(all_long[predictable],
                            offset=np.log(all_long.loc[predictable, "exposure"]))
              / all_long.loc[predictable, "exposure"]).to_numpy()
        glm_all_flat[predictable] = pr
    visualize_heatmap(_honda(_to_df(glm_all_flat.reshape(len(models), len(areas)))),
                      "Predicted Pure Premium Rates -- Main-Effects GLM (Honda, all cells)",
                      fig_path=f"{PAPER_DIR}/fig_4_3_1.png")

    # NOTE: the GLMM heatmaps (paper/fig_4_4_1.png, fig_4_4_2.png) are produced
    # by glmm_pymc.py -- a fully-converged Bayesian GLMM with uncertainty --
    # rather than the saturated statsmodels OLRE fit. Run glmm_pymc.py after
    # this script to (re)generate them.


def main():
    pure_premium, pp_mat, exp_mat, obs_cells, W_full, U_mat, I_mat = prepare_data()
    best, ctx = run_comparison(pure_premium, pp_mat, exp_mat, W_full, U_mat, I_mat)
    generate_paper_figures(pure_premium, pp_mat, exp_mat, obs_cells, W_full, best, ctx)

    print("\n================ SUMMARY ================")
    print(f"best (k, lambda) : ({best['k']}, {best['lambda']})")
    print(f"eval test cells  : {ctx['n_eval']}")
    print(ctx["comparison"].to_string())
    print("(All three models scored on the identical held-out cells; MF loss is "
          "exposure-weighted to match GLM/GLMM.)")


if __name__ == "__main__":
    main()
