"""
Shared helpers for the Brazilian auto-insurance ratemaking analysis.

Ported from the R helpers in cmf.R and reused by every Python entry point
(brazil_data_analysis_R.py, glmm_pymc.py). Grouped into:
  * data loading / aggregation  -- load_bravehins, get_total, load_pure_premium,
    wide_to_long_format
  * splitting & CV              -- train_test_split, k_fold_split, optimize_params,
    get_prediction
  * metrics                     -- calc_rmse, weighted_rmse, poisson_deviance
  * visualisation               -- visualize_scatter_plot, visualize_heatmap

Dependencies: pandas, numpy, matplotlib, cmfrec
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmfrec import CMF

CLAIM_TYPES = [
    "ClaimAmountRob", "ClaimAmountPartColl", "ClaimAmountTotColl",
    "ClaimAmountFire", "ClaimAmountOther",
]


# =============================================================================
# Data loading / aggregation
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


def load_pure_premium(csv_path="data/brvehins_org.csv", brand="Honda",
                      cell_exposure_min=100, model_exposure_min=10):
    """Build the vehicle-model x region pure-premium and exposure matrices.

    Reproduces the R preprocessing (Sections 2-3): filter to one brand, sum the
    claim components, aggregate claims and exposure to a model x region matrix,
    keep only cells with exposure >= `cell_exposure_min` (others become NaN =
    missing) and models whose total exposure exceeds `model_exposure_min`.

    Returns
    -------
    (pure_premium, exposure_total) : both wide DataFrames sharing index/columns.
        pure_premium = total claim / total exposure (NaN where missing).
    """
    brv = load_bravehins(csv_path)
    brv = brv[brv["VehModel"].str.contains(brand, na=False)]
    brv["ClaimTotal"] = brv[CLAIM_TYPES].sum(axis=1)

    cats = ["VehModel", "Area"]
    exposure_total = get_total(brv, cats, "ExposTotal", cell_exposure_min)
    claim_total = get_total(brv, cats, "ClaimTotal")

    keep = exposure_total.sum(axis=1, skipna=True) > model_exposure_min
    exposure_total = exposure_total.loc[keep]
    claim_total = claim_total.reindex(index=exposure_total.index,
                                      columns=exposure_total.columns)

    pure_premium = claim_total / exposure_total
    return pure_premium, exposure_total


def wide_to_long_format(wide_df, value_names=("var1", "var2", "value"), na_omit=True):
    """Melt a wide matrix to long format (cf. cmf.R::wide_to_long_format)."""
    long_df = wide_df.reset_index().melt(id_vars=wide_df.index.name)
    long_df.columns = list(value_names)
    if na_omit:
        long_df = long_df.dropna()
    return long_df


# =============================================================================
# Splitting & cross-validation
# =============================================================================

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
                    nonneg=True, verbose=False, center=False,
                ).fit(X_train, W=W)
                pred = get_prediction(model, X_val)
                cv_score += calc_rmse(pred, X_val, show=False) / n_folds
            print(f"k: {k} lambda: {lam} CV RMSE: {cv_score}")
            records.append((k, lam, cv_score))
    cv_result = pd.DataFrame(records, columns=["k", "lambda", "cv_score"])
    best = cv_result.loc[cv_result["cv_score"].idxmin()]
    return {"k": int(best["k"]), "lambda": float(best["lambda"]),
            "cv_score": float(best["cv_score"])}


# =============================================================================
# Metrics
# =============================================================================

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


# =============================================================================
# Visualisation
# =============================================================================

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
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
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
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"saved {fig_path}")
    plt.close()
