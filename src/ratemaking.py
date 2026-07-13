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

# Collision-only components (部分衝突 + 全損衝突). The active analysis targets
# collision claims: heavy-tail perils (theft/total-loss/fire) dominate the L2
# loss and "Other" swamps the frequency signal, so restricting the numerator to
# collision gives a meaningful, well-conditioned target for both pure premium
# and frequency. CLAIM_TYPES (all 5) is retained only for load_standardized_relativity.
COLLISION_AMOUNT = ["ClaimAmountPartColl", "ClaimAmountTotColl"]
COLLISION_NB = ["ClaimNbPartColl", "ClaimNbTotColl"]


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


def load_cell_matrix(csv_path="data/brvehins1_full.csv", brand=None,
                     target="pure_premium", cell_exposure_min=100,
                     model_exposure_min=10, row_col="VehModel", col_col="Area"):
    """Build the vehicle x region rate and exposure matrices.

    Reproduces the R preprocessing (Sections 2-3): optionally filter to one
    brand (`brand=None` keeps every manufacturer), aggregate the COLLISION
    claim numerator and exposure to a `row_col` x region matrix, keep only cells
    with exposure >= `cell_exposure_min` (others become NaN = missing) and
    rows whose total exposure exceeds `model_exposure_min`.

    `row_col` chooses the row granularity: "VehModel" (~4200 trim-level models,
    the default / canonical analysis) or "VehGroup" (~436 model families, a
    coarser, much denser matrix).

    The numerator is restricted to collision claims (部分衝突 + 全損衝突):
      * target="pure_premium": numerator = sum(COLLISION_AMOUNT)  (claim amount)
      * target="frequency":    numerator = sum(COLLISION_NB)      (claim count)
    Either way the rate is numerator / ExposTotal, and total exposure serves as
    the credibility weight / GLM offset. Cells whose collision claims are zero
    (rate = 0) are valid observations, not missing.

    Returns
    -------
    (rate, exposure_total) : both wide DataFrames sharing index/columns.
    """
    if target == "pure_premium":
        num_cols = COLLISION_AMOUNT
    elif target == "frequency":
        num_cols = COLLISION_NB
    else:
        raise ValueError(f"target must be 'pure_premium' or 'frequency', got {target!r}")

    brv = load_bravehins(csv_path)
    if brand is not None:
        brv = brv[brv["VehModel"].str.contains(brand, na=False)]
    brv = brv.copy()
    brv["Numerator"] = brv[num_cols].sum(axis=1)

    cats = [row_col, col_col]
    exposure_total = get_total(brv, cats, "ExposTotal", cell_exposure_min)
    numerator_total = get_total(brv, cats, "Numerator")

    keep = exposure_total.sum(axis=1, skipna=True) > model_exposure_min
    exposure_total = exposure_total.loc[keep]
    numerator_total = numerator_total.reindex(index=exposure_total.index,
                                              columns=exposure_total.columns)

    rate = numerator_total / exposure_total
    return rate, exposure_total


def load_pure_premium(csv_path="data/brvehins1_full.csv", brand=None,
                      cell_exposure_min=100, model_exposure_min=10):
    """Collision pure-premium rate matrix -- thin wrapper over load_cell_matrix.

    Kept so existing imports (`from ratemaking import load_pure_premium`) stay
    valid. Returns (pure_premium, exposure_total).
    """
    return load_cell_matrix(csv_path=csv_path, brand=brand, target="pure_premium",
                            cell_exposure_min=cell_exposure_min,
                            model_exposure_min=model_exposure_min)


def load_standardized_relativity(csv_path="data/brvehins1_full.csv", brand=None,
                                 cell_exposure_min=100, model_exposure_min=10):
    """Build a demographically-standardized vehicle-model x region risk matrix.

    The raw cell pure premium (load_pure_premium) confounds model x region risk
    with each cell's gender / driver-age / vehicle-year MIX, which varies
    strongly across cells (per-cell male-exposure share ranges 0..1). To isolate
    the model x region signal, we first fit a record-level Poisson GLM on those
    demographic factors -- controlling for model/area so the demographic
    relativities are unbiased -- then form a demographic *expected-claims* base

        E*_record = exposure x exp(intercept + demographic linear predictor)

    by predicting with VehModel and Area held at their reference level. The cell
    target becomes the standardized relativity  r_ij = sum(claim) / sum(E*), and
    E* replaces exposure as the credibility weight / GLM offset. Demographic mix
    is thereby removed identically for every downstream model (GLM/GLMM/MF/CMF),
    so the comparison reflects the model x region interaction, not who happens to
    drive those cars in those regions.

    Assumption: demographics act multiplicatively and do not interact with the
    model x region cell (no three-way interaction) -- the standard working
    assumption for a-priori relativity offsets.

    Returns (relativity, expected_base): a drop-in replacement for the
    (pure_premium, exposure_total) pair returned by load_pure_premium().
    """
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import OneHotEncoder

    claim_nb = ["ClaimNbRob", "ClaimNbPartColl", "ClaimNbTotColl",
                "ClaimNbFire", "ClaimNbOther"]
    brv = load_bravehins(csv_path)
    if brand is not None:
        brv = brv[brv["VehModel"].str.contains(brand, na=False)].copy()
    brv["ClaimTotal"] = brv[CLAIM_TYPES].sum(axis=1)
    brv["ClaimNbTotal"] = brv[claim_nb].sum(axis=1)
    brv = brv[brv["ExposTotal"] > 0].copy()
    # missing demographics -> explicit "Unknown" level so every record keeps an
    # E* (dropping would lose exposure and leave those cells un-aggregatable)
    brv["Gender"] = brv["Gender"].fillna("Unknown")
    brv["DrivAge"] = brv["DrivAge"].fillna("Unknown")

    # demographic FREQUENCY GLM (claim counts, Poisson) -> stable, standard for
    # a-priori relativities. We control for coarse vehicle risk via VehGroup and
    # for Area so the demographic relativities are de-biased, then strip those
    # out to leave a demographic-adjusted "equivalent exposure". (A Poisson fit
    # on claim AMOUNTS diverges here; frequency is the natural, stable choice.)
    #
    # Poisson is closed under aggregation of identical-covariate records, so we
    # fit on counts COLLAPSED to the unique (Gender, DrivAge, VehYear, VehGroup,
    # Area) combos. We fit with a SPARSE one-hot design (scikit-learn) rather than
    # statsmodels' dense patsy matrix: with a 436-level VehGroup on the full ~2M
    # -row multi-brand data the dense design is ~550 wide and blows up memory,
    # whereas the sparse one has only 5 non-zeros per row. Fitting the rate
    # y = count / exposure with sample_weight = exposure reproduces the offset
    # -Poisson MLE exactly; a tiny L2 (alpha) just resolves the one-hot collinearity.
    gcols = ["Gender", "DrivAge", "VehYear", "VehGroup", "Area"]
    agg = (brv.groupby(gcols, observed=True)
              .agg(ClaimNbTotal=("ClaimNbTotal", "sum"),
                   ExposTotal=("ExposTotal", "sum")).reset_index())
    agg = agg[agg["ExposTotal"] > 0]
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
    X = enc.fit_transform(agg[gcols].astype(str))
    demo = PoissonRegressor(alpha=1e-8, fit_intercept=True, max_iter=1000)
    demo.fit(X, agg["ClaimNbTotal"] / agg["ExposTotal"],
             sample_weight=agg["ExposTotal"].to_numpy())

    # E* = exposure x rate, with VehGroup & Area forced to their reference level
    # so only exposure + demographics survive. The predicted rate then depends
    # ONLY on (Gender, DrivAge, VehYear) -> a small lookup we predict once and
    # broadcast onto every record (memory-light for millions of rows).
    ref_keys = ["Gender", "DrivAge", "VehYear"]
    rate_tbl = brv[ref_keys].drop_duplicates().copy()
    rate_tbl["VehGroup"] = sorted(brv["VehGroup"].dropna().unique())[0]
    rate_tbl["Area"] = sorted(brv["Area"].dropna().unique())[0]
    rate_tbl["rate"] = demo.predict(enc.transform(rate_tbl[gcols].astype(str)))
    brv = brv.merge(rate_tbl[ref_keys + ["rate"]], on=ref_keys, how="left")
    brv["expected_base"] = brv["ExposTotal"] * brv["rate"]

    cats = ["VehModel", "Area"]
    exposure_total = get_total(brv, cats, "ExposTotal", cell_exposure_min)
    claim_total = get_total(brv, cats, "ClaimTotal")
    ebase_total = get_total(brv, cats, "expected_base")

    keep = exposure_total.sum(axis=1, skipna=True) > model_exposure_min
    exposure_total = exposure_total.loc[keep]
    idx, cols = exposure_total.index, exposure_total.columns
    claim_total = claim_total.reindex(index=idx, columns=cols)
    ebase_total = ebase_total.reindex(index=idx, columns=cols)

    # relativity, missing exactly where the cell has too little exposure (<min)
    relativity = (claim_total / ebase_total).mask(exposure_total.isna())
    ebase_total = ebase_total.mask(exposure_total.isna())
    return relativity, ebase_total


def wide_to_long_format(wide_df, value_names=("var1", "var2", "value"), na_omit=True):
    """Melt a wide matrix to long format (cf. cmf.R::wide_to_long_format)."""
    long_df = wide_df.reset_index().melt(id_vars=wide_df.index.name)
    long_df.columns = list(value_names)
    if na_omit:
        long_df = long_df.dropna()
    return long_df


def build_side_info(pure_premium, csv_path="data/brvehins1_full.csv",
                    density_path="data/brazil_population_density.csv"):
    """Build row/column side-information matrices for Collective MF (CMF).

    For the VehGroup x State configuration:

    * Row side info (U): the manufacturer / COMPANY of each vehicle-group row,
      taken as the first token of the VehGroup label (e.g. "Gm Chevrolet Kadett"
      -> "Gm", "Honda Motos Ate 450cc" -> "Honda"), one-hot encoded. It groups
      the ~200 vehicle groups into their maker so sparse/cold rows can borrow
      strength from same-company groups.
    * Column side info (I): the population-density CLASS of each State (IBGE
      Censo 2022), tertiled across the 27 states into low / medium / high and
      one-hot encoded. Density spans ~2.5-493 hab/km²; a three-level class is a
      robust urban/rural proxy that avoids committing to a single cut point.

    U is aligned to `pure_premium`'s index (vehicle groups), I to its columns
    (States). Returns (U, I, u_labels, i_labels) with U, I as float numpy arrays
    of shape (n_groups, p_u) and (n_states, p_i).

    `csv_path` is accepted for signature stability but no longer read (the
    company is derived from the row label itself).
    """
    groups = pure_premium.index
    states = pure_premium.columns

    # ---- row side info: manufacturer / company one-hot ----------------------
    company = groups.to_series().str.split().str[0]
    U = pd.get_dummies(company).astype(float)

    # ---- column side info: State population-density class (tertiles) --------
    dens = pd.read_csv(density_path)
    # each State's IBGE density: single-state rows carry the state-level value
    state_density = (dens[dens["note"] == "single-state"]
                     .drop_duplicates("parent_state")
                     .set_index("parent_state")["density_km2"])
    d = state_density.reindex(states).astype(float)
    dclass = pd.qcut(d, q=3, labels=["dens_low", "dens_med", "dens_high"])
    # unmatched state (if any) -> all-zero class row (get_dummies drops NaN)
    I = pd.get_dummies(dclass).astype(float).reindex(states, fill_value=0.0)

    return (U.to_numpy(dtype=float), I.to_numpy(dtype=float),
            list(U.columns), list(I.columns))


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


def optimize_params(X, n_folds, k_values, lambda_values, random_seed=123, W=None,
                    U=None, I=None, w_main=1.0, w_user=1.0, w_item=1.0):
    """CV grid search over (k, lambda) for CMF (cf. cmf.R::optimize_params).

    If `W` (per-cell weights, same shape as X) is given, the CMF loss is
    weighted -- so the tuned lambda is calibrated for the SAME weighted loss
    used in the final fit (otherwise a weighted final fit would be effectively
    unregularized). If `U` / `I` (row / column side-information matrices) are
    given, the search tunes the Collective MF variant with those attributes,
    using the same w_main/w_user/w_item weighting as the final fit so the tuned
    (k, lambda) transfer. Returns the best row as {"k", "lambda", "cv_score"}.
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
                    w_main=w_main, w_user=w_user, w_item=w_item,
                ).fit(X_train, W=W, U=U, I=I)
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
