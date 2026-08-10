"""
Extra GLMM validations for the VehModel x State THEFT test paper (§ 予測性能の比較).

Adds two analyses the user asked for, on top of the existing cell-level hold-out
comparison (where GLMM == GLM because every test cell is an unseen interaction):

  Task 1 -- In-sample (observed-cell) vs hold-out.
    Fit GLM / GLMM / MF on the observed cells and score on those SAME cells
    (apparent / in-sample fit). Here the interaction models (GLMM, MF) differ
    from the additive GLM because their per-cell interaction terms ARE estimated.
    Contrast with the hold-out numbers, where GLMM reverts to GLM.

  Task 2 -- Record-level split (an alternative evaluation design).
    Split the underlying policy RECORDS 75/25 (not whole cells), so a cell (i,j)
    can appear in both train and test. The GLMM's interaction random effect is
    then estimable for shared cells, so GLMM != GLM out-of-sample. This measures
    within-cell interaction recovery rather than cold-cell extrapolation.

GLMM family: to keep GLM and GLMM on the SAME footing (per the user's request),
the GLMM shares the GLM's exact Tweedie main-effects fit and adds the
vehicle-model x state interaction as a random effect implemented as
exposure-weighted Buhlmann-Straub credibility under the Tweedie variance
function (Var[rate] = phi * mu^p / exposure). This is the linear-credibility
analog of a random intercept on the Tweedie GLM, so the ONLY difference between
the GLM and GLMM rows is the interaction term (no Poisson-vs-Tweedie confound).
Python has no off-the-shelf Tweedie mixed model (statsmodels' mixed models are
Gaussian-only; pymc has no Tweedie distribution; R's glmmTMB is unavailable
here), so this credibility form is the faithful, deterministic substitute.

Reuses the CV-tuned MF hyper-parameters (k=8, lambda=1000) from the main run.

Outputs (docs/):
  insample_vs_holdout_vehmodel_theft.csv
  recordsplit_vehmodel_theft.csv
  glmm_extra_scalars_vehmodel_theft.csv   (tau, mean credibility, n's, text)
"""

import os
import sys

import numpy as np
import pandas as pd
from cmfrec import CMF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import load_bravehins, load_cell_matrix, weighted_rmse
from brazil_data_analysis_R import _fit_glm, _predict_glm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(_ROOT, "data", "brvehins1_full.csv")
DOCS_DIR = os.path.join(_ROOT, "docs")
ROW, COL = "VehModel", "State"
CELL_MIN, MODEL_MIN = 100, 10
BEST = {"k": 8, "lambda": 1000.0}      # CV-tuned in the main run
VAR_POWER = 1.5                        # Tweedie p, matching _fit_glm
SEED = 123


def _metrics(pred, act, w):
    pred = np.asarray(pred, float)
    return {
        "RMSE": float(np.sqrt(np.mean((pred - act) ** 2))),
        "wRMSE(exposure)": float(weighted_rmse(pred, act, w)),
        "corr": float(np.corrcoef(pred, act)[0, 1]),
    }


def _tweedie_credibility(glm_model, m, y, w, p=VAR_POWER):
    """Interaction random effect as Tweedie-based Buhlmann-Straub credibility.

    m = main-effects Tweedie GLM rate (prior mean), y = raw cell rate,
    w = exposure. Process variance of the exposure-weighted cell rate under the
    Tweedie model is s2 = phi * m^p / w; the between-cell interaction variance
    tau2 is estimated by (weighted) method of moments from the residuals. The
    credibility factor Z = tau2 / (tau2 + s2) shrinks each cell's raw departure
    toward the main-effects fit -- exactly a random-intercept's partial pooling.
    Returns (credibility-adjusted rate, tau2, Z).
    """
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    phi = float(glm_model.scale)                 # Tweedie dispersion
    s2 = phi * np.power(m, p) / w                 # per-cell process variance
    r = y - m
    tau2 = max(0.0, float(np.average(r ** 2 - s2, weights=w)))
    Z = tau2 / (tau2 + s2)
    return m + Z * r, tau2, Z


def _fit_mf(X, W):
    return CMF(k=BEST["k"], lambda_=BEST["lambda"], method="als", niter=30,
               nonneg=True, verbose=False, center=False).fit(X, W=W)


# ---------------------------------------------------------------------------
# Task 1: in-sample (observed-cell) fit vs hold-out
# ---------------------------------------------------------------------------
def task1_insample():
    rate, expo = load_cell_matrix(
        csv_path=DATA_CSV, target="pure_premium", cell_exposure_min=CELL_MIN,
        model_exposure_min=MODEL_MIN, row_col=ROW, col_col=COL, peril="theft")
    models = rate.index.to_numpy()
    areas = rate.columns.to_numpy()
    pp = rate.to_numpy(float)
    exp_mat = expo.to_numpy(float)
    r, c = np.where(~np.isnan(pp))
    act = pp[r, c]
    w = exp_mat[r, c]
    print(f"[task1] matrix {pp.shape}, observed cells {len(r)}")

    # GLM (additive main effects, Tweedie weighted-rate), fit & scored in-sample
    long = pd.DataFrame({"VehModel": models[r], "Area": areas[c],
                         "pure_premium": act, "exposure": w})
    long["claim"] = long["pure_premium"] * long["exposure"]
    glm = _fit_glm(long, "pure_premium")
    glm_pred = _predict_glm(glm, long, "pure_premium")

    # GLMM = same Tweedie main effects + credibility interaction (Tweedie family)
    glmm_pred, tau2, Z = _tweedie_credibility(glm, glm_pred, act, w)

    # MF fit on observed cells, predicted on the same cells
    obs = ~np.isnan(pp)
    W_full = np.nan_to_num(exp_mat, nan=0.0) / float(exp_mat[obs].mean())
    mf = _fit_mf(pp, W_full)
    mf_pred = np.asarray(mf.predict(user=r, item=c), float)

    tbl = pd.DataFrame({
        "GLM (Tweedie, additive)": _metrics(glm_pred, act, w),
        "GLMM (Tweedie + credibility interaction)": _metrics(glmm_pred, act, w),
        "MF (weighted)": _metrics(mf_pred, act, w),
    }).T
    print("\n===== Task1: IN-SAMPLE fit on observed cells =====")
    print(tbl.to_string())
    tbl.to_csv(f"{DOCS_DIR}/insample_vs_holdout_vehmodel_theft.csv")
    return {"tau2_insample": tau2, "sqrt_tau2_insample": float(np.sqrt(tau2)),
            "meanZ_insample": float(np.average(Z, weights=w)),
            "n_obs": int(len(r))}


# ---------------------------------------------------------------------------
# Task 2: record-level split
# ---------------------------------------------------------------------------
def task2_recordsplit():
    rate, expo = load_cell_matrix(
        csv_path=DATA_CSV, target="pure_premium", cell_exposure_min=CELL_MIN,
        model_exposure_min=MODEL_MIN, row_col=ROW, col_col=COL, peril="theft")
    models = list(rate.index)
    areas = list(rate.columns)
    m_index = {m: i for i, m in enumerate(models)}
    a_index = {a: j for j, a in enumerate(areas)}
    pp = rate.to_numpy(float)
    ri, ci = np.where(~np.isnan(pp))
    obs_pairs = set(zip(rate.index[ri], rate.columns[ci]))

    brv = load_bravehins(DATA_CSV)
    brv["Numerator"] = brv["ClaimAmountRob"]
    brv = brv[[ROW, COL, "Numerator", "ExposTotal"]].copy()
    brv = brv[brv[ROW].isin(m_index) & brv[COL].isin(a_index)]
    brv["pair"] = list(zip(brv[ROW], brv[COL]))
    brv = brv[brv["pair"].isin(obs_pairs)]
    print(f"[task2] records in observed-cell universe: {len(brv)}")

    rng = np.random.RandomState(SEED)
    is_train = rng.rand(len(brv)) < 0.75

    def agg(df):
        g = (df.groupby([ROW, COL], observed=True)
               .agg(claim=("Numerator", "sum"), exp=("ExposTotal", "sum"))
               .reset_index())
        g = g[g["exp"] > 0].copy()
        g["rate"] = g["claim"] / g["exp"]
        return g

    tr = agg(brv[is_train])
    te = agg(brv[~is_train])
    seen_models = set(tr[ROW])
    seen_states = set(tr[COL])
    te = te[te[ROW].isin(seen_models) & te[COL].isin(seen_states)].copy()
    shared = int(sum((m, a) in set(zip(tr[ROW], tr[COL]))
                     for m, a in zip(te[ROW], te[COL])))
    print(f"[task2] train cells {len(tr)}, test cells scored {len(te)} "
          f"(shared with train: {shared})")

    act = te["rate"].to_numpy(float)
    w = te["exp"].to_numpy(float)

    # ---- GLM (additive Tweedie) on train aggregates ----
    tr_long = tr.rename(columns={COL: "Area"}).copy()
    tr_long["pure_premium"] = tr_long["rate"]
    tr_long["exposure"] = tr_long["exp"]
    glm = _fit_glm(tr_long, "pure_premium")
    te_long = te.rename(columns={COL: "Area"}).copy()
    te_long["pure_premium"] = te_long["rate"]
    te_long["exposure"] = te_long["exp"]
    glm_pred = _predict_glm(glm, te_long, "pure_premium")

    # ---- GLMM = train Tweedie main effects + credibility interaction ----
    # credibility learned on TRAIN cells, keyed by (model,state); a shared test
    # cell inherits its train credibility-adjusted rate, so GLMM != GLM.
    tr_m = _predict_glm(glm, tr_long, "pure_premium")
    tr_adj, tau2, Z = _tweedie_credibility(
        glm, tr_m, tr["rate"].to_numpy(float), tr["exp"].to_numpy(float))
    adj_by_pair = {(m, a): v for m, a, v in zip(tr[ROW], tr[COL], tr_adj)}
    glmm_pred = np.array([adj_by_pair.get((m, a), gp)
                          for m, a, gp in zip(te[ROW], te[COL], glm_pred)])

    # ---- MF on train matrix ----
    tr_r = tr[ROW].map(m_index).to_numpy()
    tr_c = tr[COL].map(a_index).to_numpy()
    te_r = te[ROW].map(m_index).to_numpy()
    te_c = te[COL].map(a_index).to_numpy()
    Xtr = np.full((len(models), len(areas)), np.nan)
    Etr = np.zeros((len(models), len(areas)))
    Xtr[tr_r, tr_c] = tr["rate"].to_numpy(float)
    Etr[tr_r, tr_c] = tr["exp"].to_numpy(float)
    Wtr = Etr / float(Etr[Etr > 0].mean())
    mf = _fit_mf(Xtr, Wtr)
    mf_pred = np.asarray(mf.predict(user=te_r, item=te_c), float)

    tbl = pd.DataFrame({
        "GLM (Tweedie, additive)": _metrics(glm_pred, act, w),
        "GLMM (Tweedie + credibility interaction)": _metrics(glmm_pred, act, w),
        "MF (weighted)": _metrics(mf_pred, act, w),
    }).T
    print("\n===== Task2: RECORD-LEVEL split, scored on held-out records =====")
    print(tbl.to_string())
    tbl.to_csv(f"{DOCS_DIR}/recordsplit_vehmodel_theft.csv")
    return {"tau2_rec": tau2, "sqrt_tau2_rec": float(np.sqrt(tau2)),
            "meanZ_rec": float(np.average(Z, weights=tr["exp"].to_numpy(float))),
            "n_test_cells": int(len(te)), "n_train_cells": int(len(tr)),
            "glmm_ne_glm": bool(not np.allclose(glmm_pred, glm_pred))}


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    s1 = task1_insample()
    s2 = task2_recordsplit()
    scalars = {**s1, **s2}
    pd.Series(scalars).to_csv(f"{DOCS_DIR}/glmm_extra_scalars_vehmodel_theft.csv")
    print("\nscalars:", scalars)
    print("done.")
