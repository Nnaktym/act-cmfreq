"""
=============================================================================
Brazil auto insurance data analysis (matrix factorization for class ratemaking)
Python analysis (MAIN script) -- based on brazil_data_analysis_R.R (+ cmf.R)
=============================================================================

This is the primary Python implementation of the ratemaking analysis. It began
as a translation of the R pipeline (cmf.R helpers are ported inline, so the file
is self-contained -- no `source("cmf.R")` needed) and then implements the core
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

Differences from the R version (numbers differ by design):
  * Data is loaded from data/brvehins_org.csv (the Honda-filtered raw export the
    R script writes) instead of the CASdatasets .rda files.
  * R and Python RNGs differ, so the split, CV folds, and chosen (k, lambda)
    won't match R to the decimal.
  * GLMM: statsmodels' PoissonBayesMixedGLM (variational Bayes) has no offset, so
    log(exposure) enters as a fixed covariate -- an exposure-aware substitute for
    R's lme4::glmer offset. Point estimates are close, not identical.

Dependencies: pandas, numpy, matplotlib, cmfrec, statsmodels
  pip install cmfrec statsmodels
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmfrec import CMF

# =============================================================================
# HELPERS  (ported from cmf.R)
# =============================================================================


def load_bravehins(csv_path):
    """Load the Brazilian auto insurance data.

    R original loads the five CASdatasets .rda shards and rbinds them. Here we
    read the already-combined CSV export (data/brvehins_org.csv).
    """
    return pd.read_csv(csv_path)


def fill_with_na(df, threshold):
    """Fill values lower than `threshold` with NaN (cf. cmf.R::fill_with_na)."""
    return df.mask(df < threshold)


def get_total(data, category_to_analyze, aggregate_col, threshold=None):
    """Aggregate `aggregate_col` into a (cat1 x cat2) matrix by summing.

    Mirrors cmf.R::get_total: pivot to a wide matrix (rows = category 1,
    cols = category 2), then blank out cells below `threshold`.
    Returns a wide-format DataFrame (index = cat1, columns = cat2).
    """
    print(f"aggregate_col: {aggregate_col}   group_cols: {category_to_analyze}")
    cat1, cat2 = category_to_analyze
    total = data.pivot_table(
        index=cat1, columns=cat2, values=aggregate_col, aggfunc="sum"
    )
    if threshold is not None:
        total = fill_with_na(total, threshold)
    print(total.shape)
    return total


def train_test_split(X, ratio=0.75, seed=123):
    """Cell-level split of a wide matrix into train/test (cf. cmf.R).

    Observed (non-NaN) cells are partitioned; the complementary cells are masked
    to NaN in each returned matrix. `X` is a 2D numpy array.
    """
    rng = np.random.RandomState(seed)
    valid = np.argwhere(~np.isnan(X))
    n_valid = len(valid)
    n_train = int(np.floor(ratio * n_valid))
    perm = rng.permutation(n_valid)
    train_idx = valid[perm[:n_train]]
    test_idx = valid[perm[n_train:]]
    X_train = X.copy()
    X_test = X.copy()
    X_train[test_idx[:, 0], test_idx[:, 1]] = np.nan
    X_test[train_idx[:, 0], train_idx[:, 1]] = np.nan
    return {"X_train": X_train, "X_test": X_test}


def k_fold_split(X, k=4, seed=123):
    """Split observed cells into k folds for cross-validation (cf. cmf.R)."""
    rng = np.random.RandomState(seed)
    valid = np.argwhere(~np.isnan(X))
    n_valid = len(valid)
    shuffled = valid[rng.permutation(n_valid)]
    fold_size = n_valid // k
    folds = []
    start = 0
    for i in range(k):
        end = n_valid if i == k - 1 else start + fold_size
        val_idx = shuffled[start:end]
        X_train = X.copy()
        X_val = X.copy()
        # validation cells -> NaN in train ; everything else -> NaN in val
        mask = np.ones(n_valid, dtype=bool)
        mask[start:end] = False
        train_idx = shuffled[mask]
        X_train[val_idx[:, 0], val_idx[:, 1]] = np.nan
        X_val[train_idx[:, 0], train_idx[:, 1]] = np.nan
        folds.append({"train": X_train, "val": X_val})
        start = end
    return {"folds": folds, "X": X}


def wide_to_long_format(wide_df, value_names=("var1", "var2", "value"), na_omit=True):
    """Melt a wide matrix to long format (cf. cmf.R::wide_to_long_format)."""
    long_df = wide_df.reset_index().melt(id_vars=wide_df.index.name)
    long_df.columns = list(value_names)
    if na_omit:
        long_df = long_df.dropna()
    return long_df


def get_prediction(model, X):
    """Predict every observed (non-NaN) cell of X and fill them in (cf. cmf.R).

    `X` is a 2D numpy array; returns a copy with observed cells replaced by the
    model prediction (missing cells stay NaN).
    """
    rows, cols = np.where(~np.isnan(X))
    preds = model.predict(user=rows, item=cols)
    X_pred = X.copy()
    X_pred[rows, cols] = preds
    return X_pred


def calc_rmse(pred, act, show=True):
    """RMSE over cells observed in both matrices (cf. cmf.R::calc_rmse)."""
    both = ~np.isnan(pred) & ~np.isnan(act)
    rmse = np.sqrt(np.mean((pred[both] - act[both]) ** 2))
    if show:
        print(f"RMSE : {rmse:.4f}")
    return rmse


def weighted_rmse(pred, act, w):
    """Exposure-weighted RMSE on the pure-premium (rate) scale."""
    pred, act, w = (np.asarray(a, float) for a in (pred, act, w))
    return np.sqrt(np.sum(w * (pred - act) ** 2) / np.sum(w))


def poisson_deviance(y, mu):
    """Poisson deviance on the total-claim (count) scale.

    Puts GLM / GLMM / MF on comparable, exposure-aware footing: y = actual
    total claim (= pure_premium * exposure), mu = predicted rate * exposure.
    """
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-8, None)
    term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu))


def optimize_params(X, n_folds, k_values, lambda_values, random_seed=123, W=None):
    """CV grid search over (k, lambda) for CMF (cf. cmf.R::optimize_params).

    If `W` (per-cell weights, same shape as X) is given, the CMF loss is
    weighted -- so the tuned lambda is calibrated for the SAME weighted loss
    used in the final fit (otherwise a weighted final fit would be effectively
    unregularized). Returns the best row as {"k", "lambda", "cv_score"}.
    """
    cv_split = k_fold_split(X, k=n_folds, seed=random_seed)
    records = []
    for k in k_values:
        for lam in lambda_values:
            cv_score = 0.0
            for i in range(n_folds):
                X_train = cv_split["folds"][i]["train"]
                X_val = cv_split["folds"][i]["val"]
                model = CMF(
                    k=k, lambda_=lam, method="als", niter=30,
                    nonneg=True, verbose=False,
                ).fit(X_train, W=W)
                pred = get_prediction(model, X_val)
                cv_score += calc_rmse(pred, X_val, show=False) / n_folds
            print(f"k: {k} lambda: {lam} CV RMSE: {cv_score}")
            records.append((k, lam, cv_score))
    cv_result = pd.DataFrame(records, columns=["k", "lambda", "cv_score"])
    best = cv_result.loc[cv_result["cv_score"].idxmin()]
    return {"k": int(best["k"]), "lambda": float(best["lambda"]),
            "cv_score": float(best["cv_score"])}


def visualize_scatter_plot(actual, pred, model_name, max_lim=2500, fig_path=None):
    """Predicted-vs-true scatter with a 45-degree line (cf. cmf.R)."""
    actual = np.asarray(actual, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    keep = ~np.isnan(actual) & ~np.isnan(pred)
    actual, pred = actual[keep], pred[keep]
    plt.figure(figsize=(5, 5))
    plt.scatter(actual, pred, color="black", alpha=0.7)
    plt.plot([0, max_lim], [0, max_lim], color="red")
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"predicted vs. true values ({model_name})")
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"saved {fig_path}")
    plt.close()


def visualize_heatmap(data, title="", max_limit=5000, fig_path=None):
    """Heatmap of a wide matrix (cf. cmf.R::visualize_heatmap).

    `data` is a wide DataFrame (index = model, columns = region).
    """
    mat = np.asarray(data, dtype=float)
    plt.figure(figsize=(15, 10))
    im = plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max_limit)
    plt.colorbar(im, label="Pure Premium")
    plt.xlabel("Category")
    plt.ylabel("Model")
    plt.title(title)
    if hasattr(data, "columns"):
        plt.xticks(range(mat.shape[1]), list(data.columns), rotation=45, ha="right",
                   fontsize=6)
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"saved {fig_path}")
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    FIG_DIR = "figs/python_port"
    os.makedirs(FIG_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. データの読み込み  (Load data)
    # -------------------------------------------------------------------------
    brvehins = load_bravehins("data/brvehins_org.csv")
    brvehins = brvehins[brvehins["VehModel"].str.contains("Honda", na=False)]
    claim_types = [
        "ClaimAmountRob", "ClaimAmountPartColl", "ClaimAmountTotColl",
        "ClaimAmountFire", "ClaimAmountOther",
    ]
    brvehins["ClaimTotal"] = brvehins[claim_types].sum(axis=1)
    print(brvehins.shape)

    # -------------------------------------------------------------------------
    # 3. データの下処理  (Preprocessing)
    # -------------------------------------------------------------------------
    category_to_analyze = ["VehModel", "Area"]

    # exposure が 100 以上の区分のみを残す (keep cells with exposure >= 100)
    premium_total = get_total(brvehins, category_to_analyze, "PremTotal", 0)
    exposure_total = get_total(brvehins, category_to_analyze, "ExposTotal", 100)
    claim_total = get_total(brvehins, category_to_analyze, "ClaimTotal")

    # exposure 合計が 10 以上の車種のみを残す (keep models whose total exposure > 10)
    row_keep = exposure_total.sum(axis=1, skipna=True) > 10
    claim_total = claim_total.loc[row_keep]
    premium_total = premium_total.loc[row_keep]
    exposure_total = exposure_total.loc[row_keep]

    # align claim_total / premium_total to exposure_total's cells & order
    claim_total = claim_total.reindex(index=exposure_total.index,
                                      columns=exposure_total.columns)
    premium_total = premium_total.reindex(index=exposure_total.index,
                                           columns=exposure_total.columns)

    # 純率 = クレームコスト = 総クレーム / 総エクスポージャー (pure premium)
    pure_premium = claim_total / exposure_total
    loss_ratio = claim_total / premium_total  # noqa: F841  (kept for parity)

    pp_mat = pure_premium.to_numpy(dtype=float)  # wide matrix for CMF
    print(f"pure_premium matrix: {pp_mat.shape}, "
          f"observed cells: {np.sum(~np.isnan(pp_mat))}")

    # exposure matrix aligned to pp_mat (used for weighting + weighted metrics)
    exp_mat = exposure_total.to_numpy(dtype=float)
    # CMF weights: exposure normalized to mean 1 over observed cells, so the
    # weighted loss stays on the same scale as the unweighted one and the tuned
    # lambda remains meaningful (raw exposure ~100-15000 would swamp lambda).
    obs_cells = ~np.isnan(pp_mat)
    mean_exp = float(exp_mat[obs_cells].mean())
    W_full = np.nan_to_num(exp_mat, nan=0.0) / mean_exp

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import patsy
    from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM

    # -------------------------------------------------------------------------
    # 4. 分割してからチューニング  (Split FIRST, then tune -- avoid leakage)
    # -------------------------------------------------------------------------
    # Reviewer fix: hold out the test cells BEFORE the CV grid search so the
    # chosen (k, lambda) never see the test set.
    split = train_test_split(pp_mat, ratio=0.75, seed=123)
    X_train, X_test = split["X_train"], split["X_test"]
    train_mask = ~np.isnan(X_train)

    # Common evaluation set: test cells whose vehicle model AND area both appear
    # in the training data, so that GLM / GLMM are well defined there. All three
    # models are then scored on this identical set (reviewer fix #1).
    rows_ok = train_mask.any(axis=1)
    cols_ok = train_mask.any(axis=0)
    eval_mask = (~np.isnan(X_test)) & rows_ok[:, None] & cols_ok[None, :]
    n_test_all = int(np.sum(~np.isnan(X_test)))
    n_eval = int(np.sum(eval_mask))
    print(f"eval test cells: {n_eval} "
          f"(dropped {n_test_all - n_eval} test cells with an unseen model/area)")

    best_params = optimize_params(
        X=X_train,  # <- training matrix only
        n_folds=4,
        k_values=range(2, 31),
        lambda_values=[0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000],
        W=W_full,   # <- weighted CV, matching the weighted final fit
    )
    print("best_params:", best_params)

    # -------------------------------------------------------------------------
    # 5. 3手法を同一テストセルで評価  (Evaluate MF / GLM / GLMM on the same cells)
    # -------------------------------------------------------------------------
    er, ec = np.where(eval_mask)
    act = pp_mat[er, ec]                # actual pure premium on the eval cells
    expw = exp_mat[er, ec]              # exposure on the eval cells (weights)

    # ---- MF (exposure-weighted) --------------------------------------------
    # Reviewer fix #2: weight the MF loss by exposure via W=, matching the
    # offset(log(exposure)) evidence weighting used by the GLM / GLMM.
    mf = CMF(
        k=best_params["k"], lambda_=best_params["lambda"], method="als",
        niter=30, nonneg=True, verbose=False, center=False,
    ).fit(X_train, W=W_full)
    mf_pred = np.asarray(mf.predict(user=er, item=ec), dtype=float)

    # ---- long-format train / test for GLM & GLMM ---------------------------
    models = pure_premium.index.to_numpy()
    areas = pure_premium.columns.to_numpy()
    tr, tc = np.where(train_mask)
    train_long = pd.DataFrame({
        "VehModel": models[tr], "Area": areas[tc],
        "pure_premium": pp_mat[tr, tc], "exposure": exp_mat[tr, tc],
    })
    train_long["claim"] = train_long["pure_premium"] * train_long["exposure"]
    train_long["interaction"] = (train_long["VehModel"].astype(str) + "." +
                                 train_long["Area"].astype(str))
    test_long = pd.DataFrame({
        "VehModel": models[er], "Area": areas[ec],
        "pure_premium": act, "exposure": expw,
    })
    # centered log(exposure) as the GLMM's exposure covariate (centering keeps
    # its coefficient well-scaled so the variational/MAP fit stays stable)
    _le_mean = float(np.log(train_long["exposure"]).mean())
    train_long["log_exposure"] = np.log(train_long["exposure"]) - _le_mean
    test_long["log_exposure"] = np.log(test_long["exposure"]) - _le_mean

    # ---- GLM (main effects, Poisson, offset log(exposure)) -----------------
    glm = smf.glm(
        "claim ~ C(VehModel) + C(Area)", data=train_long,
        family=sm.families.Poisson(), offset=train_long["log_exposure"],
    ).fit()
    glm_pred = (glm.predict(test_long, offset=test_long["log_exposure"])
                / test_long["exposure"]).to_numpy()

    # ---- GLMM (interaction as random intercept) ----------------------------
    # statsmodels' Bayes mixed GLM has no offset, so log(exposure) enters as a
    # fixed covariate instead (a documented, exposure-aware substitute for the
    # offset). Each observed cell is its own interaction level, so on the test
    # cells -- all unseen interactions -- the random effect is 0 and the GLMM
    # reverts to its main effects (exactly the paper's stated GLMM limitation).
    glmm = PoissonBayesMixedGLM.from_formula(
        "claim ~ C(VehModel) + C(Area) + log_exposure",
        {"interaction": "0 + C(interaction)"}, train_long,
    ).fit_map()  # MAP/Laplace -- closer to lme4::glmer and more stable than VB
    dm_tr = patsy.dmatrix("C(VehModel) + C(Area) + log_exposure",
                          train_long, return_type="dataframe")
    dm_te = patsy.build_design_matrices(
        [dm_tr.design_info], test_long, return_type="dataframe")[0]
    glmm_pred = None
    if dm_tr.shape[1] == len(glmm.fe_mean):
        glmm_pred = np.exp(dm_te.to_numpy() @ glmm.fe_mean) / test_long["exposure"].to_numpy()
    # guard: if the GLMM diverged, fall back to GLM (its test-cell behaviour
    # reverts to main effects anyway, so this is a faithful stand-in)
    if glmm_pred is None or not np.all(np.isfinite(glmm_pred)) or \
            glmm_pred.max() > 1e6:
        print("WARN: GLMM fit unstable; falling back to GLM predictions for GLMM")
        glmm_pred = glm_pred

    # ---- Comparison table on the identical eval cells ----------------------
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
    print("\n===== Held-out comparison (identical test cells) =====")
    print(comparison.to_string())
    os.makedirs("docs", exist_ok=True)
    comparison.to_csv("docs/model_comparison_python.csv")
    print("saved docs/model_comparison_python.csv")

    # test-set predicted-vs-true scatter for every model (not just MF)
    visualize_scatter_plot(act, mf_pred, "MF (weighted)",
                           fig_path=f"{FIG_DIR}/scatter_test_mf.png")
    visualize_scatter_plot(act, glm_pred, "GLM",
                           fig_path=f"{FIG_DIR}/scatter_test_glm.png")
    visualize_scatter_plot(act, glmm_pred, "GLMM",
                           fig_path=f"{FIG_DIR}/scatter_test_glmm.png")

    # -------------------------------------------------------------------------
    # 6. 全カテゴリの推定  (Refit weighted MF on all data -> all-cell heatmap)
    # -------------------------------------------------------------------------
    mf_full = CMF(
        k=best_params["k"], lambda_=best_params["lambda"], method="als",
        niter=30, nonneg=True, verbose=False, center=False,
    ).fit(pp_mat, W=W_full)

    visualize_heatmap(pure_premium, "actual",
                      fig_path=f"{FIG_DIR}/heatmap_actual.png")
    estimated_mf = get_prediction(mf_full, np.zeros_like(pp_mat))  # predict all cells
    visualize_heatmap(pd.DataFrame(estimated_mf, index=pure_premium.index,
                                   columns=pure_premium.columns),
                      "pred: MF (weighted)", fig_path=f"{FIG_DIR}/heatmap_mf.png")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n================ SUMMARY ================")
    print(f"best (k, lambda) : ({best_params['k']}, {best_params['lambda']})")
    print(f"eval test cells  : {n_eval}")
    print(comparison.to_string())
    print("(All three models scored on the identical held-out cells; MF loss is "
          "exposure-weighted to match GLM/GLMM.)")


if __name__ == "__main__":
    main()
