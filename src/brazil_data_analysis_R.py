"""
=============================================================================
Brazil auto insurance data analysis (matrix factorization for class ratemaking)
Python port of brazil_data_analysis_R.R  (+ helpers ported from cmf.R)
=============================================================================

This is a faithful Python translation of the R analysis pipeline. It keeps the
same 8 sections and the same modelling choices so the two implementations can be
compared. The cmf.R helper functions are ported inline (see the "HELPERS"
section) so this file is self-contained -- no `source("cmf.R")` equivalent needed.

Differences from the R version (numbers may differ slightly, by design):
  * Data is loaded from data/brvehins_org.csv (the Honda-filtered raw export that
    the R script writes) instead of the CASdatasets .rda files, which are awkward
    to read from Python.
  * Random number generators differ between R and Python, so the exact train/test
    split and CV folds -- and therefore the exact RMSE / chosen (k, lambda) -- will
    not match R to the decimal. The pipeline and conclusions are the same.
  * GLM uses statsmodels; GLMM (Poisson with a random interaction intercept) uses
    statsmodels' PoissonBayesMixedGLM, which is a variational-Bayes approximation
    of R's lme4::glmer (Laplace). Point estimates are close, not identical.

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


def optimize_params(X, n_folds, k_values, lambda_values, random_seed=123):
    """CV grid search over (k, lambda) for CMF (cf. cmf.R::optimize_params).

    Returns the best row as a dict {"k", "lambda", "cv_score"}.
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
                ).fit(X_train)
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

    # -------------------------------------------------------------------------
    # 4. ハイパーパラメータの最適化  (Hyperparameter optimization via CV)
    # -------------------------------------------------------------------------
    best_params = optimize_params(
        X=pp_mat,
        n_folds=4,
        k_values=range(2, 31),
        lambda_values=[0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000],
    )
    print("best_params:", best_params)

    # -------------------------------------------------------------------------
    # 5. 予測精度の検証  (Hold-out prediction accuracy)
    # -------------------------------------------------------------------------
    split = train_test_split(pp_mat)

    model = CMF(
        k=best_params["k"], lambda_=best_params["lambda"], method="als",
        niter=30, nonneg=True, verbose=False, center=False,
    ).fit(split["X_train"])

    pred = get_prediction(model, split["X_test"])
    test_rmse = calc_rmse(pred, split["X_test"])  # MF hold-out RMSE

    # -------------------------------------------------------------------------
    # 6. 予測結果の可視化  (Visualize predictions)
    # -------------------------------------------------------------------------
    # (1) test data reproducibility
    visualize_scatter_plot(split["X_test"], pred, "Matrix Factorization",
                           fig_path=f"{FIG_DIR}/mf_scatter_test.png")

    # (2) estimate missing cells: refit on the full matrix
    all_mat = pp_mat.copy()
    model_full = CMF(
        k=best_params["k"], lambda_=best_params["lambda"], method="als",
        niter=30, nonneg=True, verbose=False, center=False,
    ).fit(all_mat)

    pred_full = get_prediction(model_full, all_mat)
    calc_rmse(pred_full, all_mat)  # in-sample (all-data training) RMSE

    visualize_scatter_plot(all_mat, pred_full, "Matrix Factorization",
                           fig_path=f"{FIG_DIR}/mf_scatter_all.png")

    # heatmaps: actual, and MF estimate over ALL cells (incl. missing)
    visualize_heatmap(pure_premium, "actual",
                      fig_path=f"{FIG_DIR}/heatmap_actual.png")

    # predict every cell (fill missing with any value -> predict all positions)
    all_positions = np.zeros_like(all_mat)  # non-NaN everywhere -> predict all
    estimated_mf = get_prediction(model_full, all_positions)
    estimated_mf_df = pd.DataFrame(estimated_mf, index=pure_premium.index,
                                   columns=pure_premium.columns)
    visualize_heatmap(estimated_mf_df, "pred: Matrix Factorization",
                      fig_path=f"{FIG_DIR}/heatmap_mf.png")

    # -------------------------------------------------------------------------
    # 7. 交互作用なしの GLM との比較  (GLM, no interaction)
    # -------------------------------------------------------------------------
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    exposure_long = wide_to_long_format(
        exposure_total, ("VehModel", "Area", "exposure"), na_omit=False)
    pp_long = wide_to_long_format(
        pure_premium, ("VehModel", "Area", "pure_premium"), na_omit=False)
    all_data = pp_long.merge(exposure_long, on=["VehModel", "Area"])

    fit_df = all_data.dropna().copy()
    fit_df["claim"] = fit_df["pure_premium"] * fit_df["exposure"]  # total claim = response

    glm_model = smf.glm(
        "claim ~ C(VehModel) + C(Area)",
        data=fit_df,
        family=sm.families.Poisson(),
        offset=np.log(fit_df["exposure"]),
    ).fit()

    glm_pred = glm_model.predict(
        fit_df, offset=np.log(fit_df["exposure"])) / fit_df["exposure"]
    visualize_scatter_plot(fit_df["pure_premium"], glm_pred, "GLM (no interaction)",
                           fig_path=f"{FIG_DIR}/glm_scatter.png")

    glm_test_rmse = np.sqrt(np.mean((glm_pred.values - fit_df["pure_premium"].values) ** 2))

    # GLM heatmap over observed cells (no missing-cell extrapolation)
    glm_wide = fit_df.assign(pred=glm_pred.values).pivot_table(
        index="VehModel", columns="Area", values="pred", aggfunc="first")
    glm_wide = glm_wide.reindex(index=pure_premium.index, columns=pure_premium.columns)
    visualize_heatmap(glm_wide, "pred: GLM (no interaction)",
                      fig_path=f"{FIG_DIR}/heatmap_glm.png")

    # -------------------------------------------------------------------------
    # 8. 交互作用を変量効果に入れた GLMM との比較  (GLMM, interaction as random effect)
    # -------------------------------------------------------------------------
    from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM

    glmm_df = all_data.dropna().copy()
    glmm_df["claim"] = glmm_df["pure_premium"] * glmm_df["exposure"]
    glmm_df["interaction"] = (glmm_df["VehModel"].astype(str) + "." +
                              glmm_df["Area"].astype(str))
    # offset folded into the response scale: model log(mu) = X b + u + log(exposure)
    glmm_df["log_exposure"] = np.log(glmm_df["exposure"])

    # random intercept per interaction (one level per observed cell)
    vc = {"interaction": "0 + C(interaction)"}
    glmm_model = PoissonBayesMixedGLM.from_formula(
        "claim ~ C(VehModel) + C(Area)", vc, glmm_df,
    ).fit_vb()
    print(glmm_model.summary())

    # NOTE: statsmodels' Bayes mixed GLM has no offset in from_formula; the
    # exposure scaling is therefore only approximate here compared with R's
    # offset(log(exposure)). Numbers may differ from the R GLMM accordingly.
    # Its .predict() wants a design matrix and ignores random effects, so we
    # reconstruct the fitted counts from the model's own design matrices, which
    # includes the interaction random intercept:
    #   log(mu) = X @ fe_mean + Z @ vc_mean
    lin_pred = (glmm_model.model.exog @ glmm_model.fe_mean
                + glmm_model.model.exog_vc @ glmm_model.vc_mean)
    glmm_pred_count = np.exp(np.asarray(lin_pred).ravel())
    glmm_pred = glmm_pred_count / glmm_df["exposure"].values
    visualize_scatter_plot(glmm_df["pure_premium"], glmm_pred, "GLMM",
                           fig_path=f"{FIG_DIR}/glmm_scatter.png")

    glmm_test_rmse = np.sqrt(np.mean((glmm_pred - glmm_df["pure_premium"].values) ** 2))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n================ SUMMARY ================")
    print(f"best (k, lambda)      : ({best_params['k']}, {best_params['lambda']})")
    print(f"MF   hold-out RMSE    : {test_rmse:.4f}")
    print(f"GLM  in-sample RMSE   : {glm_test_rmse:.4f}")
    print(f"GLMM in-sample RMSE   : {glmm_test_rmse:.4f}")
    print("(GLM/GLMM are in-sample fits, matching the R script's design.)")


if __name__ == "__main__":
    main()
