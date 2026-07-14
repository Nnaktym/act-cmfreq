"""Demographic-standardized-relativity variant of the MF/CMF/GLM/GLMM comparison.

The mainline cell target (load_cell_matrix) is the RAW collision pure premium
r = sum(collision amount) / sum(exposure) over a VehGroup x State cell. That
target confounds the vehicle-group x State interaction with the demographic MIX
(Gender / DrivAge / VehYear) of whoever happens to sit in each cell.

This variant swaps in the DEMOGRAPHIC-STANDARDIZED relativity target from
load_standardized_relativity(collision_only=True): a record-level Poisson
frequency GLM on the demographics (with VehGroup + State forced to reference)
produces a demographic expected-claims base E* = exposure x exp(demographic
linear predictor). The cell target becomes r_ij = sum(collision amount) / sum(E*)
and E* REPLACES exposure everywhere (CMF weight, GLM offset, deviance weight).
The demographic mix is thereby removed identically for MF / CMF / GLM / GLMM, so
the held-out comparison reflects the VehGroup x State interaction, not the mix.

Everything else -- the split-before-tune protocol, the eval mask (test cells
whose row AND column both appear in train), the exposure(=E*)-weighted CMF loss,
the (k, lambda) CV grid, the CMF side-info weight search, and the held-out
metrics -- mirrors brazil_data_analysis_R.run_comparison, so results are directly
comparable to the raw-target VehGroup x State run.

Non-destructive: outputs go to docs/*_stdrel_vehgroup_state.csv only; the papers,
README and canonical docs/model_comparison_python.csv are never touched.

Run:  python3 src/stdrel_variant.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from cmfrec import CMF

from ratemaking import (
    build_side_info, load_standardized_relativity, optimize_params,
    poisson_deviance, train_test_split, weighted_rmse,
)
from brazil_data_analysis_R import (
    K_GRID, LAMBDA_GRID, WEIGHT_GRID, CMF_WEIGHT_K_GRID, _fit_weighted_mf,
    DOCS_DIR,
)


def run():
    sfx = "_stdrel_vehgroup_state"
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ---- standardized relativity + demographic base E* (VehGroup x State) ----
    relativity, ebase = load_standardized_relativity(
        csv_path="data/brvehins1_full.csv", brand=None,
        row_col="VehGroup", col_col="State", collision_only=True)
    groups = relativity.index.to_numpy()
    states = relativity.columns.to_numpy()
    r_mat = relativity.to_numpy(dtype=float)
    e_mat = ebase.to_numpy(dtype=float)               # E* replaces exposure

    obs_cells = ~np.isnan(r_mat)
    mean_e = float(e_mat[obs_cells].mean())
    W_full = np.nan_to_num(e_mat, nan=0.0) / mean_e   # E* normalized to mean 1
    print(f"stdrel matrix: {r_mat.shape}, observed cells: {obs_cells.sum()} "
          f"({100 * obs_cells.mean():.1f}% dense)")

    # ---- side info: company (row) + State density class (column) -------------
    U_mat, I_mat, u_labels, i_labels = build_side_info(relativity)
    print(f"side info: U {U_mat.shape} ({len(u_labels)} companies), "
          f"I {I_mat.shape} ({len(i_labels)} density classes)")

    # ---- split FIRST, then tune (no leakage) ---------------------------------
    split = train_test_split(r_mat, ratio=0.75, seed=123)
    X_train, X_test = split["X_train"], split["X_test"]
    train_mask = ~np.isnan(X_train)

    rows_ok = train_mask.any(axis=1)
    cols_ok = train_mask.any(axis=0)
    eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
    n_test_all = int(np.sum(~np.isnan(X_test)))
    n_eval = int(eval_mask.sum())
    print(f"eval test cells: {n_eval} (dropped {n_test_all - n_eval} unseen row/col)")

    best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                           lambda_values=LAMBDA_GRID, W=W_full)
    print("best_params (MF):", best)

    er, ec = np.where(eval_mask)
    act = r_mat[er, ec]
    ew = e_mat[er, ec]                                # E* weight on eval cells

    # ---- MF (E*-weighted, non-negative) --------------------------------------
    mf = _fit_weighted_mf(X_train, W_full, best)
    mf_pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)

    # ---- CMF (side info): tune weights, then refit on full (k, lambda) grid --
    best_cmf, best_w = None, None
    for wm, wu, wi in WEIGHT_GRID:
        cand = optimize_params(X_train, n_folds=4, k_values=CMF_WEIGHT_K_GRID,
                               lambda_values=LAMBDA_GRID, W=W_full, U=U_mat, I=I_mat,
                               w_main=wm, w_user=wu, w_item=wi)
        print(f"CMF weights (main,user,item)=({wm},{wu},{wi}) -> {cand}")
        if best_cmf is None or cand["cv_score"] < best_cmf["cv_score"]:
            best_cmf, best_w = cand, (wm, wu, wi)
    wm, wu, wi = best_w
    best_cmf = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                               lambda_values=LAMBDA_GRID, W=W_full, U=U_mat, I=I_mat,
                               w_main=wm, w_user=wu, w_item=wi)
    print(f"best_params (CMF+side info): {best_cmf}  weights(m,u,i)={best_w}")
    cmf = CMF(k=best_cmf["k"], lambda_=best_cmf["lambda"], method="als", niter=30,
              nonneg=True, verbose=False, center=False,
              w_main=wm, w_user=wu, w_item=wi).fit(X_train, W=W_full, U=U_mat, I=I_mat)
    cmf_pred = np.asarray(cmf.predict(user=er, item=ec), dtype=float)

    # ---- long-format train / test for GLM (offset = log E*) ------------------
    tr, tc = np.where(train_mask)
    train_long = pd.DataFrame({
        "Veh": groups[tr], "Area": states[tc],
        "relativity": r_mat[tr, tc], "ebase": e_mat[tr, tc],
    })
    train_long["claim"] = train_long["relativity"] * train_long["ebase"]
    train_long["log_ebase"] = np.log(train_long["ebase"])
    test_long = pd.DataFrame({
        "Veh": groups[er], "Area": states[ec],
        "relativity": act, "ebase": ew,
    })
    test_long["log_ebase"] = np.log(test_long["ebase"])

    # ---- GLM (main effects, Poisson, offset log E*) --------------------------
    glm = smf.glm(
        "claim ~ C(Veh) + C(Area)", data=train_long,
        family=sm.families.Poisson(), offset=train_long["log_ebase"],
    ).fit()
    glm_pred = (glm.predict(test_long, offset=test_long["log_ebase"])
                / test_long["ebase"]).to_numpy()

    # ---- GLMM: held-out interaction reverts to main effects -> = GLM ---------
    glmm_pred = glm_pred

    # ---- comparison table on the identical eval cells ------------------------
    def _metrics(pred):
        return {
            "RMSE": float(np.sqrt(np.mean((pred - act) ** 2))),
            "wRMSE(E*)": float(weighted_rmse(pred, act, ew)),
            "PoissonDeviance": float(poisson_deviance(act * ew, pred * ew)),
        }

    comparison = pd.DataFrame({
        "MF (weighted)": _metrics(mf_pred),
        "CMF (side info)": _metrics(cmf_pred),
        "GLM": _metrics(glm_pred),
        "GLMM": _metrics(glmm_pred),
    }).T
    print("\n===== stdrel held-out comparison (identical test cells) =====")
    print(comparison.to_string())
    out = f"{DOCS_DIR}/model_comparison_python{sfx}.csv"
    comparison.to_csv(out)
    print(f"saved {out}")

    # ---- stratify by E* (sparse vs dense) ------------------------------------
    preds = {"MF (weighted)": mf_pred, "CMF (side info)": cmf_pred,
             "GLM": glm_pred, "GLMM": glmm_pred}
    median_e = float(np.median(ew))
    strata = {
        "sparse (E* < median)": ew < median_e,
        "dense  (E* >= median)": ew >= median_e,
    }
    strat = pd.DataFrame([
        {"stratum": sname, "model": mname, "n": int(smask.sum()),
         "wRMSE": float(weighted_rmse(pred[smask], act[smask], ew[smask])),
         "PoissonDev": float(poisson_deviance(
             act[smask] * ew[smask], pred[smask] * ew[smask]))}
        for sname, smask in strata.items() for mname, pred in preds.items()
    ])
    print("\n===== stratified by E* =====")
    print(strat.to_string(index=False))
    strat.to_csv(f"{DOCS_DIR}/model_comparison_by_exposure_python{sfx}.csv", index=False)
    print(f"saved {DOCS_DIR}/model_comparison_by_exposure_python{sfx}.csv")

    print("\n================ SUMMARY (standardized relativity, VehGroup x State) ===")
    print(f"target           : standardized collision relativity (demographic mix removed)")
    print(f"matrix           : {r_mat.shape}, {obs_cells.sum()} observed cells")
    print(f"n_eval           : {n_eval}")
    print(f"MF   (k, lambda) : ({best['k']}, {best['lambda']})")
    print(f"CMF  (k, lambda) : ({best_cmf['k']}, {best_cmf['lambda']})  "
          f"side-info weights (main,user,item)={best_w}")
    print(comparison.to_string())
    print("\n" + strat.to_string(index=False))
    return comparison, strat


if __name__ == "__main__":
    run()
