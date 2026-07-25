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
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch

# Shared visual constants so every heatmap in the paper renders consistently.
MISSING_COLOR = "#d9d9d9"   # explicit off-ramp grey for low-exposure (missing) cells
SEQ_CMAP = "viridis"        # sequential (magnitude) ramp
DIV_CMAP = "RdBu_r"         # diverging ramp for interaction (ratio-to-additive) maps

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

# Theft (robbery) components. NOTE: brvehins1 ships ExposFireRob/PremFireRob
# (the intended fire+theft exposure) as all-zero in every shard, so theft has no
# dedicated exposure base; we measure robbery claims against ExposTotal (the
# comprehensive-cover exposure every peril's claims arise from). See
# docs/theft_exposure_note.md.
THEFT_AMOUNT = ["ClaimAmountRob"]
THEFT_NB = ["ClaimNbRob"]

# peril -> (amount columns, count columns)
_PERIL_COLS = {
    "collision": (COLLISION_AMOUNT, COLLISION_NB),
    "theft": (THEFT_AMOUNT, THEFT_NB),
}


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
                     model_exposure_min=10, row_col="VehModel", col_col="Area",
                     peril="collision"):
    """Build the vehicle x region rate and exposure matrices.

    Reproduces the R preprocessing (Sections 2-3): optionally filter to one
    brand (`brand=None` keeps every manufacturer), aggregate the claim numerator
    and exposure to a `row_col` x region matrix, keep only cells with exposure
    >= `cell_exposure_min` (others become NaN = missing) and rows whose total
    exposure exceeds `model_exposure_min`.

    `row_col` chooses the row granularity: "VehModel" (~4200 trim-level models,
    the default / canonical analysis) or "VehGroup" (~436 model families, a
    coarser, much denser matrix).

    `peril` chooses the numerator claim components:
      * "collision" (default): partial + total collision (部分衝突 + 全損衝突)
      * "theft":               robbery / vehicle theft (ClaimAmountRob / NbRob)
    and `target` chooses amount vs count within that peril:
      * target="pure_premium": numerator = claim AMOUNT
      * target="frequency":    numerator = claim COUNT
    Either way the rate is numerator / ExposTotal, and total exposure serves as
    the credibility weight / GLM offset. (brvehins1's dedicated fire+theft
    exposure ExposFireRob is empty, so theft also uses ExposTotal -- see the
    peril-column note above.) Cells whose claims are zero (rate = 0) are valid
    observations, not missing.

    Returns
    -------
    (rate, exposure_total) : both wide DataFrames sharing index/columns.
    """
    brv = load_bravehins(csv_path)
    if brand is not None:
        brv = brv[brv["VehModel"].str.contains(brand, na=False)]
    brv = brv.copy()
    # numerator = claim AMOUNT (pure premium) or COUNT (frequency) for the peril
    amount_cols, nb_cols = _PERIL_COLS[peril]
    num_cols = amount_cols if target == "pure_premium" else nb_cols
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

    Kept so existing imports (`from helper import load_pure_premium`) stay
    valid. Returns (pure_premium, exposure_total).
    """
    return load_cell_matrix(csv_path=csv_path, brand=brand, target="pure_premium",
                            cell_exposure_min=cell_exposure_min,
                            model_exposure_min=model_exposure_min)


def load_standardized_relativity(csv_path="data/brvehins1_full.csv", brand=None,
                                 cell_exposure_min=100, model_exposure_min=10,
                                 row_col="VehModel", col_col="Area",
                                 collision_only=False):
    """Build a demographically-standardized row x column risk matrix.

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

    `row_col` / `col_col` choose the matrix axes (default "VehModel" x "Area";
    e.g. "VehGroup" x "State" for the coarser variant). Both are included in the
    demographic GLM design and forced to their reference level when forming E*,
    so E* carries only the demographic + intercept effect regardless of which
    axes are chosen. When `collision_only=True` the relativity numerator is the
    collision claim AMOUNT (COLLISION_AMOUNT) and the demographic frequency GLM
    response is the collision claim COUNT (COLLISION_NB); otherwise the existing
    all-claims (5-peril) numerator / count are used.

    Returns (relativity, expected_base): a drop-in replacement for the
    (pure_premium, exposure_total) pair returned by load_pure_premium().
    """
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import OneHotEncoder

    if collision_only:
        amount_cols = COLLISION_AMOUNT
        nb_cols = COLLISION_NB
    else:
        amount_cols = CLAIM_TYPES
        nb_cols = ["ClaimNbRob", "ClaimNbPartColl", "ClaimNbTotColl",
                   "ClaimNbFire", "ClaimNbOther"]
    brv = load_bravehins(csv_path)
    if brand is not None:
        brv = brv[brv["VehModel"].str.contains(brand, na=False)].copy()
    brv["ClaimTotal"] = brv[amount_cols].sum(axis=1)
    brv["ClaimNbTotal"] = brv[nb_cols].sum(axis=1)
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
    gcols = ["Gender", "DrivAge", "VehYear", row_col, col_col]
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
    rate_tbl[row_col] = sorted(brv[row_col].dropna().unique())[0]
    rate_tbl[col_col] = sorted(brv[col_col].dropna().unique())[0]
    rate_tbl["rate"] = demo.predict(enc.transform(rate_tbl[gcols].astype(str)))
    brv = brv.merge(rate_tbl[ref_keys + ["rate"]], on=ref_keys, how="left")
    brv["expected_base"] = brv["ExposTotal"] * brv["rate"]

    cats = [row_col, col_col]
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


def seriation_order(pp_mat, exp_mat):
    """Row/column order that sorts the matrix by exposure-weighted marginal cost.

    Ordering both axes low->high turns the dominant vehicle-group gradient into a
    smooth ramp, so any vehicle-group x state interaction reads as a *departure*
    from that ramp. Both scripts that draw paper heatmaps (the GLM/MF driver and
    glmm_pymc) call this on the SAME actual matrix, so every figure shares one
    order and the panels compare cell-for-cell. Unobserved rows/cols sort first.
    """
    pp = np.asarray(pp_mat, dtype=float)
    ex = np.asarray(exp_mat, dtype=float)
    obs = ~np.isnan(pp)
    w = np.where(obs, np.nan_to_num(ex, nan=0.0), 0.0)
    v = np.where(obs, pp, 0.0)

    def _marg(axis):
        num = np.sum(v * w, axis=axis)
        den = np.sum(w, axis=axis)
        out = np.full_like(num, np.nan)
        return np.divide(num, den, out=out, where=den > 0)

    row_eff, col_eff = _marg(1), _marg(0)
    row_order = np.argsort(np.where(np.isnan(row_eff), -1.0, row_eff))
    col_order = np.argsort(np.where(np.isnan(col_eff), -1.0, col_eff))
    return row_order, col_order


def two_way_residual(mat, obs, w, n_iter=25):
    """Ratio of a surface to its own exposure-weighted log-additive (row+col) fit.

    Returns exp(log v - a_i - b_j): 1.0 means the cell is fully explained by the
    vehicle-group and state main effects, so departures from 1 isolate the
    interaction. A log-link main-effects GLM surface is additive in log space by
    construction and therefore maps to ~1 everywhere (flat white); only genuine
    interaction (the MF, the noisy actual data) departs from it.
    """
    mat = np.asarray(mat, dtype=float)
    use = obs & (mat > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.where(use, np.log(mat), np.nan)
    ww = np.where(use, w, 0.0)
    a = np.zeros(mat.shape[0])
    b = np.zeros(mat.shape[1])
    for _ in range(n_iter):
        for eff, other, axis in ((a, b[None, :], 1), (b, a[:, None], 0)):
            r = logv - other
            num = np.nansum(np.where(np.isnan(r), 0.0, r * ww), axis=axis)
            den = np.nansum(np.where(np.isnan(r), 0.0, ww), axis=axis)
            eff[:] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.exp(logv - a[:, None] - b[None, :])


def visualize_heatmap(data, title="", max_limit=5000, fig_path=None,
                      row_order=None, col_order=None, log=False,
                      mark_missing=False, vmin=None):
    """Heatmap of a wide matrix (cf. cmf.R::visualize_heatmap).

    `data` is a wide DataFrame (index = model, columns = region).

    Paper figures pass `row_order`/`col_order` (shared seriation, see
    `seriation_order`), `log=True` (a log colour scale spreads the right-skewed
    pure-premium bulk instead of crushing it into the dark end) and
    `mark_missing=True` (low-exposure cells shown as explicit grey rather than
    white). Callers that omit these -- e.g. a small single-brand subset -- get the
    original linear, unsorted rendering.
    """
    df = data if hasattr(data, "columns") else pd.DataFrame(data)
    if row_order is not None:
        df = df.iloc[row_order, :]
    if col_order is not None:
        df = df.iloc[:, col_order]
    mat = np.asarray(df, dtype=float)
    n_rows = mat.shape[0]
    # Height grows with the row count so the full vehicle-group x state matrix
    # (231 x 27) stays legible; capped so it never becomes an unwieldy canvas.
    height = min(22, max(8, n_rows * 0.06))
    plt.figure(figsize=(11, height))

    cmap = plt.get_cmap(SEQ_CMAP).copy()
    if mark_missing:
        cmap.set_bad(MISSING_COLOR)
    # NaN cells are always masked to grey when mark_missing; otherwise shown raw.
    m = np.ma.masked_invalid(mat) if mark_missing else np.ma.masked_array(mat, mask=False)
    if log:
        lo = vmin if vmin is not None else 30.0
        disp = np.ma.masked_array(np.clip(m.filled(lo), lo, max_limit),
                                  mask=np.ma.getmaskarray(m))
        im = plt.imshow(disp, aspect="auto", cmap=cmap, norm=LogNorm(vmin=lo, vmax=max_limit))
        cbar_label = "Pure Premium (log scale)"
    else:
        im = plt.imshow(m, aspect="auto", cmap=cmap, vmin=vmin or 0, vmax=max_limit)
        cbar_label = "Pure Premium"
    plt.colorbar(im, label=cbar_label)
    ordered = row_order is not None or col_order is not None
    plt.xlabel("State  (low → high cost)" if ordered else "State")
    plt.ylabel(f"Vehicle Group (n={n_rows})"
               + ("  (low → high cost)" if ordered else ""))
    plt.title(title)
    if mark_missing:
        plt.legend(handles=[Patch(facecolor=MISSING_COLOR, edgecolor="none",
                                  label="no data (low exposure)")],
                   loc="lower right", fontsize=7, framealpha=0.9)
    plt.xticks(range(mat.shape[1]), list(df.columns), rotation=90, fontsize=7)
    # Row labels are only legible when there are few rows (e.g. a single-brand
    # subset); for the full matrix the per-row ticks are suppressed and the
    # ordered colour field itself carries the model x state pattern.
    if n_rows <= 30:
        plt.yticks(range(n_rows), list(df.index), fontsize=7)
    else:
        plt.yticks([])
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"saved {fig_path}")
    plt.close()


def visualize_marginals(pp_mat, exp_mat, areas, fig_path=None):
    """Sorted vehicle-group and state marginal effects (paper Fig 4.2.2).

    Communicates the paper's central empirical point -- the dominant axis of
    variation is the vehicle group, not the state -- more directly than a heatmap:
    a step curve of the ~exposure-weighted vehicle-group effect (a large spread)
    beside a bar chart of the much smaller state effect.
    """
    pp = np.asarray(pp_mat, dtype=float)
    ex = np.asarray(exp_mat, dtype=float)
    obs = ~np.isnan(pp)
    w = np.where(obs, np.nan_to_num(ex, nan=0.0), 0.0)
    v = np.where(obs, pp, 0.0)
    def _marg(axis):
        num, den = (v * w).sum(axis), w.sum(axis)
        out = np.full_like(num, np.nan)
        return np.divide(num, den, out=out, where=den > 0)

    row_eff, col_eff = _marg(1), _marg(0)

    re = np.sort(row_eff[~np.isnan(row_eff)])
    ce = pd.Series(col_eff, index=list(areas)).dropna().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].plot(re, np.arange(len(re)), lw=2, color="#3b6fb0")
    axes[0].set_title(f"Vehicle-group effect  "
                      f"({re.min():.0f}–{re.max():.0f}, "
                      f"~{re.max() / max(re.min(), 1):.0f}× spread)")
    axes[0].set_xlabel("Exposure-weighted pure premium")
    axes[0].set_ylabel(f"Vehicle group (sorted, n={len(re)})")
    axes[1].barh(range(len(ce)), ce.values, color="#3b6fb0")
    axes[1].set_yticks(range(len(ce)))
    axes[1].set_yticklabels(ce.index, fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_title(f"State effect  "
                      f"({ce.min():.0f}–{ce.max():.0f}, ~{ce.max() / ce.min():.1f}× spread)")
    axes[1].set_xlabel("Exposure-weighted pure premium")
    if fig_path:
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"saved {fig_path}")
    plt.close(fig)


def visualize_interaction_panels(panels, exp_mat, row_order, col_order,
                                 areas, n_row_total, fig_path=None):
    """Diverging interaction maps: each surface / its own additive fit (Fig 4.5.3).

    `panels` is a list of (name, matrix, obs_mask). Each surface is divided by its
    OWN weighted log-additive main-effects fit, so 1.0 (white) means "no
    interaction". The main-effects GLM is white by construction; the MF and the
    (sparse, noisy) actual data carry structure -- making "MF captures the
    interaction the additive model cannot" the figure itself.
    """
    norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=2.5)
    cmap = plt.get_cmap(DIV_CMAP).copy()
    cmap.set_bad(MISSING_COLOR)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 8.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, mat, obs) in zip(axes, panels):
        ratio = two_way_residual(mat, obs, np.asarray(exp_mat, dtype=float))
        ratio = ratio[np.ix_(row_order, col_order)]
        disp = np.ma.masked_where(~obs[np.ix_(row_order, col_order)], ratio)
        im = ax.imshow(disp, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(np.asarray(areas)[col_order], rotation=90, fontsize=5)
        ax.set_yticks([])
        ax.set_xlabel("State")
    axes[0].set_ylabel(f"Vehicle Group (n={n_row_total})  (low → high cost)")
    fig.colorbar(im, ax=axes, label="value ÷ own additive fit", shrink=0.55, pad=0.02)
    fig.suptitle("Vehicle-group × state interaction (red = above additive, "
                 "blue = below, white = none). The GLM is white by construction; "
                 "the MF carries structure.", y=0.98, fontsize=11)
    if fig_path:
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"saved {fig_path}")
    plt.close(fig)
