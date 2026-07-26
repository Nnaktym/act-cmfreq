"""
VehModel x State THEFT variant (テスト版 paper).

Same corrected protocol as brazil_data_analysis_R.py (split-before-tune,
exposure-weighted MF loss, non-negative / non-centered, identical held-out
cells for MF vs GLM vs GLMM, exposure-stratified table) but with the row axis at
the individual-model granularity (VehModel, ~1,300 rows after filters) instead of
the coarser VehGroup families. GLMM held-out prediction reverts to the
main-effects GLM out-of-sample (every test cell is an unseen model x state
interaction), exactly as in the mainline pipeline. CMF/side-info is skipped: the
test paper only needs the MF/GLM/GLMM core comparison.

Figures are restricted to the São Paulo top-exposure vehicle models so the
4259->1277-row heatmap stays legible (see generate_test_figures).

Run:  python3 src/vehmodel_state_theft.py
"""

import os

import numpy as np
import pandas as pd

from brazil_data_analysis_R import (
    _fit_glm,
    _fit_weighted_mf,
    _long_frame,
    _predict_glm,
)
from helper import (
    get_prediction,
    load_cell_matrix,
    optimize_params,
    poisson_deviance,
    seriation_order,
    train_test_split,
    visualize_heatmap,
    visualize_scatter_plot,
    weighted_rmse,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(_ROOT, "data", "brvehins1_full.csv")
PAPER_DIR = os.path.join(_ROOT, "paper")
DOCS_DIR = os.path.join(_ROOT, "docs")
FIG_DIR = os.path.join(_ROOT, "figs", "python_port")
SFX = "_theft_vehmodel"

K_GRID = range(2, 28)
LAMBDA_GRID = [0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]


def prepare():
    rate, expo = load_cell_matrix(
        csv_path=DATA_CSV, target="pure_premium", cell_exposure_min=100,
        row_col="VehModel", col_col="State", peril="theft")
    pp_mat = rate.to_numpy(float)
    exp_mat = expo.to_numpy(float)
    obs = ~np.isnan(pp_mat)
    mean_exp = float(exp_mat[obs].mean())
    W_full = np.nan_to_num(exp_mat, nan=0.0) / mean_exp
    print(f"matrix {pp_mat.shape}, observed cells {obs.sum()}")
    return rate, pp_mat, exp_mat, obs, W_full


def run(rate, pp_mat, exp_mat, W_full):
    models = rate.index.to_numpy()
    areas = rate.columns.to_numpy()

    split = train_test_split(pp_mat, ratio=0.75, seed=123)
    X_train, X_test = split["X_train"], split["X_test"]
    train_mask = ~np.isnan(X_train)
    rows_ok = train_mask.any(axis=1)
    cols_ok = train_mask.any(axis=0)
    eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
    n_test_all = int(np.sum(~np.isnan(X_test)))
    n_eval = int(eval_mask.sum())
    print(f"eval cells {n_eval} (dropped {n_test_all - n_eval} unseen-model/state)")

    best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                           lambda_values=LAMBDA_GRID, W=W_full)
    print("best:", best)

    er, ec = np.where(eval_mask)
    act = pp_mat[er, ec]
    expw = exp_mat[er, ec]

    mf = _fit_weighted_mf(X_train, W_full, best)
    mf_pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)

    tr, tc = np.where(train_mask)
    train_long = _long_frame(models, areas, tr, tc, pp_mat, exp_mat)
    test_long = pd.DataFrame({"VehModel": models[er], "Area": areas[ec],
                              "pure_premium": act, "exposure": expw})
    glm = _fit_glm(train_long, "pure_premium")
    glm_pred = _predict_glm(glm, test_long, "pure_premium")
    glmm_pred = glm_pred  # OOS revert to main effects (see module docstring)

    def _metrics(pred):
        return {"RMSE": float(np.sqrt(np.mean((pred - act) ** 2))),
                "wRMSE(exposure)": float(weighted_rmse(pred, act, expw)),
                "PoissonDeviance": float(poisson_deviance(act * expw, pred * expw))}

    comparison = pd.DataFrame({"MF (weighted)": _metrics(mf_pred),
                               "GLM": _metrics(glm_pred),
                               "GLMM": _metrics(glmm_pred)}).T
    print("\n===== Held-out comparison (identical test cells) =====")
    print(comparison.to_string())
    comparison.to_csv(f"{DOCS_DIR}/model_comparison_python{SFX}.csv")

    preds = {"MF (weighted)": mf_pred, "GLM": glm_pred, "GLMM": glmm_pred}
    median_exp = float(np.median(expw))
    strata = {"sparse (exposure < median)": expw < median_exp,
              "dense  (exposure >= median)": expw >= median_exp}
    strat_rows = [
        {"stratum": s, "model": m, "n": int(mask.sum()),
         "wRMSE": float(weighted_rmse(p[mask], act[mask], expw[mask])),
         "PoissonDev": float(poisson_deviance(act[mask] * expw[mask],
                                              p[mask] * expw[mask]))}
        for s, mask in strata.items() for m, p in preds.items()]
    strat = pd.DataFrame(strat_rows)
    print("\n===== Stratified by exposure =====")
    print(strat.to_string(index=False))
    strat.to_csv(f"{DOCS_DIR}/model_comparison_by_exposure_python{SFX}.csv", index=False)

    ctx = {"models": models, "areas": areas, "er": er, "ec": ec,
           "act": act, "mf_pred": mf_pred, "n_eval": n_eval,
           "comparison": comparison, "strat": strat}
    return best, ctx


def _sp_top_models(rate, exp_mat, n=40):
    """Row indices of the top-`n` vehicle models by São Paulo exposure (observed
    cells only). São Paulo is the densest column (~40% of exposure), so its
    top-exposure models give a legible, business-relevant heatmap subset."""
    states = list(rate.columns)
    sp = states.index("Sao Paulo")
    sp_exp = exp_mat[:, sp].copy()
    sp_exp[np.isnan(rate.to_numpy(float)[:, sp])] = -1.0  # require observed SP cell
    order = np.argsort(sp_exp)[::-1]
    return order[:n]


def generate_test_figures(rate, pp_mat, exp_mat, obs, W_full, best, ctx):
    os.makedirs(PAPER_DIR, exist_ok=True)
    models, areas = ctx["models"], ctx["areas"]
    act, mf_pred = ctx["act"], ctx["mf_pred"]
    hmax = float(np.nanpercentile(pp_mat[obs], 99)) or 1.0
    smax = float(np.nanpercentile(act, 99)) or 1.0

    # --- full-data refits -----------------------------------------------------
    mf_full = _fit_weighted_mf(pp_mat, W_full, best)
    estimated_mf = get_prediction(mf_full, np.zeros_like(pp_mat))

    obs_r, obs_c = np.where(obs)
    full_long = _long_frame(models, areas, obs_r, obs_c, pp_mat, exp_mat)
    glm_f = _fit_glm(full_long, "pure_premium")
    gm = np.repeat(np.arange(len(models)), len(areas))
    ga = np.tile(np.arange(len(areas)), len(models))
    all_long = pd.DataFrame({"VehModel": models[gm], "Area": areas[ga]})
    all_long["exposure"] = 1.0
    glm_all = _predict_glm(glm_f, all_long, "pure_premium").reshape(
        len(models), len(areas))

    # --- restrict rows to São Paulo top-exposure models for legibility --------
    sub = _sp_top_models(rate, exp_mat, n=40)
    sub_idx = rate.index[sub]

    def _sub_df(mat):
        return pd.DataFrame(mat[sub, :], index=sub_idx, columns=rate.columns)

    # order the subset by its own São Paulo-anchored marginal cost
    sub_pp = pp_mat[sub, :]
    sub_exp = exp_mat[sub, :]
    r_ord, c_ord = seriation_order(sub_pp, sub_exp)
    hm = dict(max_limit=hmax, row_order=r_ord, col_order=c_ord,
              log=True, mark_missing=True)

    # Fig A: actual theft claim cost (SP top-40 models x 27 states)
    visualize_heatmap(_sub_df(pp_mat),
                      "Actual Theft Claim Costs — São Paulo top-40 vehicle models",
                      **hm, fig_path=f"{PAPER_DIR}/fig_vm_actual{SFX}.png")
    # Fig B: main-effects GLM (all cells) on the same subset
    visualize_heatmap(_sub_df(glm_all),
                      "Predicted Theft Rates — Main-Effects GLM (SP top-40 models)",
                      **hm, fig_path=f"{PAPER_DIR}/fig_vm_glm{SFX}.png")
    # Fig C: MF all-cell estimate on the same subset
    visualize_heatmap(_sub_df(estimated_mf),
                      "Estimated Theft Rates — Matrix Factorization (SP top-40 models)",
                      **hm, fig_path=f"{PAPER_DIR}/fig_vm_mf{SFX}.png")
    # Fig D: predicted-vs-true scatter (MF, full eval set)
    visualize_scatter_plot(act, mf_pred, "Matrix Factorization (VehModel × State, theft)",
                           max_lim=smax, fig_path=f"{PAPER_DIR}/fig_vm_scatter{SFX}.png")
    print("figures written to", PAPER_DIR)


def main():
    rate, pp_mat, exp_mat, obs, W_full = prepare()
    best, ctx = run(rate, pp_mat, exp_mat, W_full)
    generate_test_figures(rate, pp_mat, exp_mat, obs, W_full, best, ctx)
    print("\n==== SUMMARY ====")
    print(f"best (k, lambda): ({best['k']}, {best['lambda']})  eval cells: {ctx['n_eval']}")
    print(ctx["comparison"].to_string())


if __name__ == "__main__":
    main()
