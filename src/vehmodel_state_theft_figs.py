"""Fast figure regen for the VehModel x State theft テスト paper.

Uses the CV-selected (k, lambda) = (8, 1000) from vehmodel_state_theft.py so it
skips the grid search: just refit MF (full data) + GLM and draw the São Paulo
top-25 heatmaps with legible, shortened model-name row labels.
"""
import os
import re

import numpy as np
import pandas as pd

from brazil_data_analysis_R import _fit_glm, _fit_weighted_mf, _long_frame, _predict_glm
from helper import (get_prediction, load_cell_matrix, seriation_order,
                    visualize_heatmap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(_ROOT, "paper")
SFX = "_theft_vehmodel"
BEST = {"k": 8, "lambda": 1000.0}   # from the CV grid in vehmodel_state_theft.py
N_SHOW = 25


def _short(name):
    """Compact a trim-level VehModel label for a heatmap tick."""
    s = re.sub(r"\s*-\s*", " ", str(name))         # drop the " - " separators
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= 44 else s[:43] + "…"


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
    hmax = float(np.nanpercentile(pp_mat[obs], 99)) or 1.0

    # full-data refits
    mf_full = _fit_weighted_mf(pp_mat, W_full, BEST)
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

    # São Paulo top-N by exposure (require an observed SP cell)
    sp = list(rate.columns).index("Sao Paulo")
    sp_exp = exp_mat[:, sp].copy()
    sp_exp[np.isnan(pp_mat[:, sp])] = -1.0
    sub = np.argsort(sp_exp)[::-1][:N_SHOW]
    sub_labels = [_short(m) for m in models[sub]]

    def _sub_df(mat):
        return pd.DataFrame(mat[sub, :], index=sub_labels, columns=rate.columns)

    r_ord, c_ord = seriation_order(pp_mat[sub, :], exp_mat[sub, :])
    hm = dict(max_limit=hmax, row_order=r_ord, col_order=c_ord,
              log=True, mark_missing=True, ylabel=f"Vehicle Model (São Paulo top-{N_SHOW})")

    visualize_heatmap(_sub_df(pp_mat),
                      f"Actual Theft Claim Costs — São Paulo top-{N_SHOW} vehicle models",
                      **hm, fig_path=f"{PAPER_DIR}/fig_vm_actual{SFX}.png")
    visualize_heatmap(_sub_df(glm_all),
                      f"Main-Effects GLM theft rates — São Paulo top-{N_SHOW} models",
                      **hm, fig_path=f"{PAPER_DIR}/fig_vm_glm{SFX}.png")
    visualize_heatmap(_sub_df(estimated_mf),
                      f"Matrix-Factorization theft rates (k=8) — São Paulo top-{N_SHOW} models",
                      **hm, fig_path=f"{PAPER_DIR}/fig_vm_mf{SFX}.png")
    print("regenerated 3 heatmaps with", N_SHOW, "labelled São Paulo models")


if __name__ == "__main__":
    main()
