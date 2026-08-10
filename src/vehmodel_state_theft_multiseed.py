"""Multi-seed robustness check for the VehModel x State theft テスト paper.

The companion VehGroup x State experiment (docs/vehgroup_state_experiment.md)
found its single-split MF "win" was seed-fragile. This re-runs the same
split-before-tune, exposure-weighted protocol for several seeds at VehModel x
State (theft) and records held-out MF vs GLM wRMSE so the paper can state
honestly whether the MF advantage is stable. Writes
docs/validation_multiseed_vehmodel_state_theft.csv.
"""
import os

import numpy as np
import pandas as pd

from brazil_data_analysis_R import _fit_glm, _fit_weighted_mf, _long_frame, _predict_glm
from helper import (load_cell_matrix, optimize_params, train_test_split,
                    weighted_rmse)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(_ROOT, "docs")
K_GRID = range(2, 28)
LAMBDA_GRID = [0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000]
SEEDS = [123, 1, 7, 42, 2024]


def main():
    rate, expo = load_cell_matrix(
        csv_path=os.path.join(_ROOT, "data", "brvehins1_full.csv"),
        target="pure_premium", cell_exposure_min=100,
        row_col="VehModel", col_col="State", peril="theft")
    pp_mat = rate.to_numpy(float)
    exp_mat = expo.to_numpy(float)
    obs = ~np.isnan(pp_mat)
    mean_exp = float(exp_mat[obs].mean())
    W_full = np.nan_to_num(exp_mat, nan=0.0) / mean_exp
    models = rate.index.to_numpy()
    areas = rate.columns.to_numpy()

    rows = []
    for seed in SEEDS:
        split = train_test_split(pp_mat, ratio=0.75, seed=seed)
        X_train, X_test = split["X_train"], split["X_test"]
        train_mask = ~np.isnan(X_train)
        rows_ok = train_mask.any(1)
        cols_ok = train_mask.any(0)
        eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
        best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                               lambda_values=LAMBDA_GRID, W=W_full,
                               random_seed=seed)
        er, ec = np.where(eval_mask)
        act = pp_mat[er, ec]
        expw = exp_mat[er, ec]
        mf = _fit_weighted_mf(X_train, W_full, best)
        mf_pred = np.asarray(mf.predict(user=er, item=ec), float)
        tr, tc = np.where(train_mask)
        train_long = _long_frame(models, areas, tr, tc, pp_mat, exp_mat)
        test_long = pd.DataFrame({"VehModel": models[er], "Area": areas[ec],
                                  "pure_premium": act, "exposure": expw})
        glm = _fit_glm(train_long, "pure_premium")
        glm_pred = _predict_glm(glm, test_long, "pure_premium")
        mf_w = weighted_rmse(mf_pred, act, expw)
        glm_w = weighted_rmse(glm_pred, act, expw)
        rows.append({"seed": seed, "k": best["k"], "lambda": best["lambda"],
                     "n_eval": int(eval_mask.sum()),
                     "MF_wRMSE": mf_w, "GLM_wRMSE": glm_w,
                     "MF_minus_GLM": mf_w - glm_w})
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(f"{DOCS_DIR}/validation_multiseed_vehmodel_state_theft.csv", index=False)
    print("\n==== multi-seed summary ====")
    print(df.to_string(index=False))
    print(f"\nMF wins {int((df['MF_minus_GLM'] < 0).sum())}/{len(df)} seeds; "
          f"mean MF-GLM = {df['MF_minus_GLM'].mean():.1f}, "
          f"sd = {df['MF_minus_GLM'].std():.1f}")


if __name__ == "__main__":
    main()
