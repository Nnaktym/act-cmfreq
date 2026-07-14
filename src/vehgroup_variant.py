"""VehGroup-row variant of the MF vs GLM vs GLMM comparison.

The canonical analysis (brazil_data_analysis_R.py) uses individual vehicle
MODELS (~4200 trim-level rows) as the matrix row axis, which is very sparse and
where MF loses to a plain GLM. This variant swaps the row axis to VehGroup
(~436 model families): a coarser, much denser matrix. Everything else -- the
collision numerator, exposure filters, the split-before-tune protocol, the
exposure-weighted CMF loss, the (k, lambda) CV grid, and the held-out metrics --
is identical to the canonical pipeline, so the two are directly comparable.

CMF (side info) is intentionally omitted here: the canonical row side-info IS
the VehGroup one-hot, which becomes an identity matrix once the rows already ARE
VehGroups (no information to add). We keep MF / GLM / GLMM, the three core
comparators.

Non-destructive: outputs go to docs/*_vehgroup{,_freq}.csv, never the canonical
files.

Run:  python3 src/vehgroup_variant.py                # pure premium
      python3 src/vehgroup_variant.py frequency      # collision frequency
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ratemaking import (
    load_cell_matrix, train_test_split, optimize_params,
    weighted_rmse, poisson_deviance,
)
from brazil_data_analysis_R import K_GRID, LAMBDA_GRID, _fit_weighted_mf, DOCS_DIR


def run(target="pure_premium", col_col="Area"):
    tsfx = "" if target == "pure_premium" else "_freq"
    csfx = "" if col_col == "Area" else "_" + col_col.lower()
    sfx = f"_vehgroup{csfx}{tsfx}"
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ---- build the VehGroup x <col_col> matrix (same filters as canonical) ---
    rate, exposure_total = load_cell_matrix(
        csv_path="data/brvehins1_full.csv", brand=None, target=target,
        cell_exposure_min=100, model_exposure_min=10,
        row_col="VehGroup", col_col=col_col)
    groups = rate.index.to_numpy()
    areas = rate.columns.to_numpy()
    pp_mat = rate.to_numpy(dtype=float)
    exp_mat = exposure_total.to_numpy(dtype=float)

    obs_cells = ~np.isnan(pp_mat)
    mean_exp = float(exp_mat[obs_cells].mean())
    W_full = np.nan_to_num(exp_mat, nan=0.0) / mean_exp
    print(f"VehGroup matrix: {pp_mat.shape}, observed cells: {obs_cells.sum()} "
          f"({100 * obs_cells.mean():.1f}% dense)")

    # ---- split FIRST, then tune (no leakage) ---------------------------------
    split = train_test_split(pp_mat, ratio=0.75, seed=123)
    X_train, X_test = split["X_train"], split["X_test"]
    train_mask = ~np.isnan(X_train)

    rows_ok = train_mask.any(axis=1)
    cols_ok = train_mask.any(axis=0)
    eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
    n_test_all = int(np.sum(~np.isnan(X_test)))
    n_eval = int(eval_mask.sum())
    print(f"eval test cells: {n_eval} (dropped {n_test_all - n_eval} unseen row/area)")

    best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                           lambda_values=LAMBDA_GRID, W=W_full)
    print("best_params:", best)

    er, ec = np.where(eval_mask)
    act = pp_mat[er, ec]
    expw = exp_mat[er, ec]

    # ---- MF (exposure-weighted, non-negative) --------------------------------
    mf = _fit_weighted_mf(X_train, W_full, best)
    mf_pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)

    # ---- long-format train / test for GLM ------------------------------------
    tr, tc = np.where(train_mask)
    train_long = pd.DataFrame({
        "Veh": groups[tr], "Area": areas[tc],
        "pure_premium": pp_mat[tr, tc], "exposure": exp_mat[tr, tc],
    })
    train_long["claim"] = train_long["pure_premium"] * train_long["exposure"]
    train_long["log_exposure"] = np.log(train_long["exposure"])
    test_long = pd.DataFrame({
        "Veh": groups[er], "Area": areas[ec],
        "pure_premium": act, "exposure": expw,
    })
    test_long["log_exposure"] = np.log(test_long["exposure"])

    # ---- GLM (main effects, Poisson, offset log(exposure)) -------------------
    glm = smf.glm(
        "claim ~ C(Veh) + C(Area)", data=train_long,
        family=sm.families.Poisson(), offset=train_long["log_exposure"],
    ).fit()
    glm_pred = (glm.predict(test_long, offset=test_long["log_exposure"])
                / test_long["exposure"]).to_numpy()

    # ---- GLMM: held-out interaction random effect reverts to main effects ----
    # (same reasoning as canonical run_comparison) -> equals the GLM prediction.
    glmm_pred = glm_pred

    # ---- comparison table on the identical eval cells ------------------------
    def _metrics(pred):
        return {
            "RMSE": float(np.sqrt(np.mean((pred - act) ** 2))),
            "wRMSE(exposure)": float(weighted_rmse(pred, act, expw)),
            "PoissonDeviance": float(poisson_deviance(act * expw, pred * expw)),
        }

    comparison = pd.DataFrame({
        "MF (weighted)": _metrics(mf_pred),
        "GLM": _metrics(glm_pred),
        "GLMM": _metrics(glmm_pred),
    }).T
    print("\n===== VehGroup held-out comparison (identical test cells) =====")
    print(comparison.to_string())
    out = f"{DOCS_DIR}/model_comparison_python{sfx}.csv"
    comparison.to_csv(out)
    print(f"saved {out}")

    # ---- stratify by exposure (sparse vs dense) ------------------------------
    preds = {"MF (weighted)": mf_pred, "GLM": glm_pred, "GLMM": glmm_pred}
    median_exp = float(np.median(expw))
    strata = {
        "sparse (exposure < median)": expw < median_exp,
        "dense  (exposure >= median)": expw >= median_exp,
    }
    strat = pd.DataFrame([
        {"stratum": sname, "model": mname, "n": int(smask.sum()),
         "wRMSE": float(weighted_rmse(pred[smask], act[smask], expw[smask])),
         "PoissonDev": float(poisson_deviance(
             act[smask] * expw[smask], pred[smask] * expw[smask]))}
        for sname, smask in strata.items() for mname, pred in preds.items()
    ])
    print("\n===== stratified by exposure =====")
    print(strat.to_string(index=False))
    strat.to_csv(f"{DOCS_DIR}/model_comparison_by_exposure_python{sfx}.csv",
                 index=False)

    print("\n================ SUMMARY (VehGroup rows) ================")
    print(f"target           : {target}")
    print(f"matrix           : {pp_mat.shape}, {obs_cells.sum()} observed cells")
    print(f"best (k, lambda) : ({best['k']}, {best['lambda']})")
    print(f"eval test cells  : {n_eval}")
    print(comparison.to_string())
    return comparison


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pure_premium"
    col_col = sys.argv[2] if len(sys.argv) > 2 else "Area"
    run(target, col_col)
