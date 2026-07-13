"""Reinforcement checks for the VehGroup x State (pure premium) variant.

The variant (src/vehgroup_variant.py) found MF beating the main-effects GLM on
the exposure-weighted metrics for a VehGroup x State matrix. Before trusting
that, this script runs three honest robustness checks and writes the results to
docs/ (all with the _vehgroup_state tag; nothing canonical is overwritten):

  A. k-vs-error curve   -- is k=2 genuinely where held-out error bottoms out,
                           or an artefact? Held-out wRMSE vs rank k at the tuned
                           lambda, with the GLM baseline drawn in.
  B. masked-cell recovery -- mask a block of OBSERVED high-exposure cells, refit
                           MF and GLM on the rest, score recovery on the masked
                           cells. Tests the "MF imputes missing cells better"
                           claim on cells whose truth we actually know.
  C. multi-seed stability -- repeat the split-tune-fit-evaluate comparison over
                           several seeds; report mean +/- sd of the MF vs GLM
                           gap so the win isn't a single-split fluke.

Run:  python3 src/vehgroup_state_validation.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from cmfrec import CMF

from ratemaking import (
    load_cell_matrix, train_test_split, optimize_params,
    weighted_rmse, poisson_deviance,
)
from brazil_data_analysis_R import K_GRID, LAMBDA_GRID, _fit_weighted_mf, DOCS_DIR, FIG_DIR

TAG = "_vehgroup_state"
SEEDS = [123, 1, 7, 42, 2024]


def load_matrix():
    """VehGroup x State pure-premium rate + exposure (same filters as canonical)."""
    rate, exposure_total = load_cell_matrix(
        csv_path="data/brvehins1_full.csv", brand=None, target="pure_premium",
        cell_exposure_min=100, model_exposure_min=10,
        row_col="VehGroup", col_col="State")
    groups = rate.index.to_numpy()
    areas = rate.columns.to_numpy()
    pp_mat = rate.to_numpy(dtype=float)
    exp_mat = exposure_total.to_numpy(dtype=float)
    obs = ~np.isnan(pp_mat)
    mean_exp = float(exp_mat[obs].mean())
    W_full = np.nan_to_num(exp_mat, nan=0.0) / mean_exp
    print(f"VehGroup x State matrix: {pp_mat.shape}, observed cells: {obs.sum()} "
          f"({100 * obs.mean():.1f}% dense)")
    return groups, areas, pp_mat, exp_mat, W_full, obs


def _glm_predict(groups, areas, train_rows, train_cols, pp_mat, exp_mat,
                 pred_rows, pred_cols):
    """Fit a main-effects Poisson GLM on the given train cells; predict target cells."""
    train_long = pd.DataFrame({
        "Veh": groups[train_rows], "Area": areas[train_cols],
        "pp": pp_mat[train_rows, train_cols], "exposure": exp_mat[train_rows, train_cols],
    })
    train_long["claim"] = train_long["pp"] * train_long["exposure"]
    train_long["log_exposure"] = np.log(train_long["exposure"])
    test_long = pd.DataFrame({
        "Veh": groups[pred_rows], "Area": areas[pred_cols],
        "exposure": exp_mat[pred_rows, pred_cols],
    })
    test_long["log_exposure"] = np.log(test_long["exposure"])
    glm = smf.glm("claim ~ C(Veh) + C(Area)", data=train_long,
                  family=sm.families.Poisson(),
                  offset=train_long["log_exposure"]).fit()
    return (glm.predict(test_long, offset=test_long["log_exposure"])
            / test_long["exposure"]).to_numpy()


# =============================================================================
# A. k-vs-error curve
# =============================================================================
def check_k_curve(groups, areas, pp_mat, exp_mat, W_full):
    print("\n########## A. k-vs-error curve ##########")
    split = train_test_split(pp_mat, ratio=0.75, seed=123)
    X_train, X_test = split["X_train"], split["X_test"]
    train_mask = ~np.isnan(X_train)
    rows_ok, cols_ok = train_mask.any(axis=1), train_mask.any(axis=0)
    eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
    er, ec = np.where(eval_mask)
    act, expw = pp_mat[er, ec], exp_mat[er, ec]

    best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                           lambda_values=LAMBDA_GRID, W=W_full)
    lam = best["lambda"]
    print(f"tuned (k, lambda) = ({best['k']}, {lam}); sweeping k at lambda={lam}")

    tr, tc = np.where(train_mask)
    glm_pred = _glm_predict(groups, areas, tr, tc, pp_mat, exp_mat, er, ec)
    glm_wrmse = float(weighted_rmse(glm_pred, act, expw))

    rows = []
    for k in range(1, 41):
        mf = CMF(k=k, lambda_=lam, method="als", niter=30, nonneg=True,
                 verbose=False, center=False).fit(X_train, W=W_full)
        pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)
        rows.append({"k": k,
                     "MF_wRMSE": float(weighted_rmse(pred, act, expw)),
                     "GLM_wRMSE": glm_wrmse})
    curve = pd.DataFrame(rows)
    out = f"{DOCS_DIR}/validation_k_curve{TAG}.csv"
    curve.to_csv(out, index=False)
    kbest = int(curve.loc[curve.MF_wRMSE.idxmin(), "k"])
    print(f"held-out wRMSE minimized at k={kbest} "
          f"(MF={curve.MF_wRMSE.min():.2f} vs GLM={glm_wrmse:.2f})")
    print(f"saved {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve.k, curve.MF_wRMSE, marker="o", ms=3, label="MF (held-out)")
        ax.axhline(glm_wrmse, ls="--", color="crimson", label="GLM baseline")
        ax.axvline(kbest, ls=":", color="gray")
        ax.set_xlabel("latent rank k"); ax.set_ylabel("held-out wRMSE (exposure)")
        ax.set_title("VehGroup x State (pure premium): held-out error vs k")
        ax.legend(); fig.tight_layout()
        figout = f"{FIG_DIR}/validation_k_curve{TAG}.png"
        fig.savefig(figout, dpi=130); plt.close(fig)
        print(f"saved {figout}")
    except Exception as e:  # plotting is optional
        print(f"(plot skipped: {e})")
    return curve


# =============================================================================
# B. masked-cell recovery (mask observed high-exposure cells, score recovery)
# =============================================================================
def check_masked_recovery(groups, areas, pp_mat, exp_mat, W_full, obs,
                          mask_frac=0.20):
    print("\n########## B. masked-cell recovery ##########")
    # tune (k, lambda) once on the full observed matrix's training analogue:
    # reuse the canonical split's tuning to fix the MF hyperparameters.
    split = train_test_split(pp_mat, ratio=0.75, seed=123)
    best = optimize_params(split["X_train"], n_folds=4, k_values=K_GRID,
                           lambda_values=LAMBDA_GRID, W=W_full)
    k, lam = best["k"], best["lambda"]
    print(f"MF hyperparameters fixed at (k, lambda) = ({k}, {lam})")

    obs_r, obs_c = np.where(obs)
    obs_exp = exp_mat[obs_r, obs_c]
    hi = obs_exp >= np.median(obs_exp)          # high-exposure observed pool
    hi_idx = np.where(hi)[0]
    print(f"observed cells: {obs.sum()}, high-exposure pool: {hi_idx.size}, "
          f"masking {mask_frac:.0%} of the pool per seed")

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n_mask = int(round(mask_frac * hi_idx.size))
        pick = rng.choice(hi_idx, size=n_mask, replace=False)
        mr, mc = obs_r[pick], obs_c[pick]

        X_mask = pp_mat.copy()
        X_mask[mr, mc] = np.nan
        W_mask = W_full.copy()
        W_mask[mr, mc] = 0.0
        train_mask = ~np.isnan(X_mask)
        # keep only masked cells whose row & col are still populated in the rest
        rows_ok, cols_ok = train_mask.any(axis=1), train_mask.any(axis=0)
        keep = rows_ok[mr] & cols_ok[mc]
        mr_k, mc_k = mr[keep], mc[keep]
        act = pp_mat[mr_k, mc_k]
        expw = exp_mat[mr_k, mc_k]

        mf = _fit_weighted_mf(X_mask, W_mask, {"k": k, "lambda": lam})
        mf_pred = np.asarray(mf.predict(user=mr_k, item=mc_k), dtype=float)

        tr, tc = np.where(train_mask)
        glm_pred = _glm_predict(groups, areas, tr, tc, pp_mat, exp_mat, mr_k, mc_k)

        rows.append({
            "seed": seed, "n_masked": int(mr_k.size),
            "MF_wRMSE": float(weighted_rmse(mf_pred, act, expw)),
            "GLM_wRMSE": float(weighted_rmse(glm_pred, act, expw)),
            "MF_PoissonDev": float(poisson_deviance(act * expw, mf_pred * expw)),
            "GLM_PoissonDev": float(poisson_deviance(act * expw, glm_pred * expw)),
        })
    rec = pd.DataFrame(rows)
    rec["MF_wins_wRMSE"] = rec.MF_wRMSE < rec.GLM_wRMSE
    out = f"{DOCS_DIR}/validation_masked_recovery{TAG}.csv"
    rec.to_csv(out, index=False)
    print(rec.to_string(index=False))
    print(f"\nMF wRMSE  mean={rec.MF_wRMSE.mean():.2f} (sd {rec.MF_wRMSE.std():.2f})")
    print(f"GLM wRMSE mean={rec.GLM_wRMSE.mean():.2f} (sd {rec.GLM_wRMSE.std():.2f})")
    print(f"MF wins recovery on {rec.MF_wins_wRMSE.sum()}/{len(rec)} seeds")
    print(f"saved {out}")
    return rec


# =============================================================================
# C. multi-seed stability of the held-out comparison
# =============================================================================
def check_multiseed(groups, areas, pp_mat, exp_mat, W_full):
    print("\n########## C. multi-seed held-out stability ##########")
    rows = []
    for seed in SEEDS:
        split = train_test_split(pp_mat, ratio=0.75, seed=seed)
        X_train, X_test = split["X_train"], split["X_test"]
        train_mask = ~np.isnan(X_train)
        rows_ok, cols_ok = train_mask.any(axis=1), train_mask.any(axis=0)
        eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
        er, ec = np.where(eval_mask)
        act, expw = pp_mat[er, ec], exp_mat[er, ec]

        best = optimize_params(X_train, n_folds=4, k_values=K_GRID,
                               lambda_values=LAMBDA_GRID, W=W_full)
        mf = _fit_weighted_mf(X_train, W_full, best)
        mf_pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)
        tr, tc = np.where(train_mask)
        glm_pred = _glm_predict(groups, areas, tr, tc, pp_mat, exp_mat, er, ec)

        mf_w = float(weighted_rmse(mf_pred, act, expw))
        glm_w = float(weighted_rmse(glm_pred, act, expw))
        rows.append({"seed": seed, "k": best["k"], "lambda": best["lambda"],
                     "n_eval": int(er.size), "MF_wRMSE": mf_w, "GLM_wRMSE": glm_w,
                     "MF_minus_GLM": mf_w - glm_w, "MF_wins": mf_w < glm_w})
    ms = pd.DataFrame(rows)
    out = f"{DOCS_DIR}/validation_multiseed{TAG}.csv"
    ms.to_csv(out, index=False)
    print(ms.to_string(index=False))
    print(f"\nMF-GLM gap: mean={ms.MF_minus_GLM.mean():.2f} "
          f"(sd {ms.MF_minus_GLM.std():.2f}); MF wins {ms.MF_wins.sum()}/{len(ms)} seeds")
    print(f"saved {out}")
    return ms


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    groups, areas, pp_mat, exp_mat, W_full, obs = load_matrix()
    check_k_curve(groups, areas, pp_mat, exp_mat, W_full)
    check_masked_recovery(groups, areas, pp_mat, exp_mat, W_full, obs)
    check_multiseed(groups, areas, pp_mat, exp_mat, W_full)
    print("\nDone. See docs/validation_*_vehgroup_state.csv and "
          "figs/python_port/validation_k_curve_vehgroup_state.png")
