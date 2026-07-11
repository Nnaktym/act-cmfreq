# cmfrec API reference (for ratemaking analysis)

`cmfrec` (author: David Cortes) fits (Collective) Matrix Factorization with biases,
non-negativity, weights, and side information. Same model in R and Python; APIs differ.
The estimated model is biased MF: `X ≈ A Bᵀ + μ + b_A 1ᵀ + 1 b_Bᵀ` (Koren et al., 2009),
optionally with side-info matrices `U ≈ A Cᵀ`, `I ≈ B Dᵀ`.

## Model matrix convention (this project)

- Rows = one high-cardinality factor (vehicle model → "user"), columns = another (region → "item").
- Cell value = **pure premium** (claim cost per exposure) for that model×region.
- Unobserved / low-exposure cells = `NA` (R) / dropped rows (Python long form). MF *completes* them.

## R — `cmfrec::CMF`

```r
model <- CMF(
  X,                 # matrix (or sparse/long triplet). NA = missing = imputed by the model.
  k        = 22,     # number of latent factors (rank of the interaction term ABᵀ)
  lambda   = 30,     # L2 regularization weight
  method   = "als",  # ALS (default) or "lbfgs"
  nonneg   = TRUE,   # constrain A, B ≥ 0 (biases stay unconstrained → predictions CAN go negative)
  center   = FALSE,  # FALSE keeps the raw non-negative scale (this project's choice)
  weight   = wt,     # <-- per-observation weights, same shape as X. USE exposure here (see guardrail)
  niter    = 30,
  verbose  = FALSE
)
pred <- predict(model, user = row_idx, item = col_idx)   # predict specific cells
```

Reusable helpers already in `src/cmf.R`: `load_bravehins`, `get_total` (aggregate to a
model×region matrix with an exposure threshold), `train_test_split` / `k_fold_split`
(mask cells to NA — cell-level split), `get_prediction` (fill NA cells), `calc_rmse`,
`optimize_params` (CV grid over k×lambda), `visualize_scatter_plot`, `visualize_heatmap`.

## Python — `cmfrec.CMF`

```python
from cmfrec import CMF
model = CMF(k=22, lambda_=30, method="als", center=False, nonneg=True)  # note: lambda_ (trailing _)
model.fit(X_long)                        # long df with columns UserId, ItemId, Rating
pred = model.predict(user=test["UserId"], item=test["ItemId"])
```

Collective MF with side information (cold-start help via attributes):

```python
model = CMF(k=22, lambda_=30, method="als", center=False, nonneg=True,
            w_main=0.5, w_user=0.25, w_item=0.25)   # weights across the three factorizations
model.fit(X=ratings, U=user_side_info_onehot, I=item_side_info_onehot)
```

In this repo: user side info = vehicle group (`VehGroup`); item side info = population-density
class (urban/rural from `data/brazil_population_density.csv`). One-hot encode side info.

## Gotchas confirmed in this repo's code

- **`lambda` (R) vs `lambda_` (Python)** — different names.
- **Predictions can be negative** even with `nonneg=TRUE`, because biases are unconstrained
  (`cmf.r:54` flags this). Clip or model on a log/positive scale if strict non-negativity is required.
- **`.iloc` with label indices** (`brazil_data_analysis_python.py:199-204`) only works while the
  frame keeps a default RangeIndex — brittle; prefer `.loc` or reset the index.
- **R and Python must be kept in parity** — currently they diverge (R: k=22, λ=30, 4-fold CV, 75/25;
  Python: k=20/50, λ=10, no CV, 80/20). Pick one as the source of truth for reported numbers.
- **Sparse/long input**: cmfrec accepts triplets; NA cells are simply absent, not zeros. Never
  fill missing pure-premium cells with 0 before fitting — 0 is a real (very cheap) rate, not "unknown".
