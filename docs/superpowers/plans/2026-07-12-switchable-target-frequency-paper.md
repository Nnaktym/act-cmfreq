# 目的変数切替（純保険料率⇄頻度）＋衝突限定＋頻度版JP論文 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** モデルの目的変数を「衝突事故（部分衝突+全損衝突）の純保険料率」と「衝突事故の頻度」で引数切替可能にし、日本語論文を2本（純保険料率版=更新、頻度版=新規）並存させる。

**Architecture:** スイッチはローダ (`load_cell_matrix(target=...)`) に集約する。パイプライン (`prepare_data`→`run_comparison`→`generate_paper_figures`) は `(rate行列, exposure行列)` にのみ依存し目的変数非依存なので、`target` 文字列を末端まで通し、出力パスと図に `_freq` 接尾辞で分岐させる。露出加重・GLMオフセット・ポアソン逸脱度の計算はそのまま流用。

**Tech Stack:** Python (pandas, numpy, matplotlib, cmfrec, statsmodels, pymc)。テストフレームワークは無いため、検証はパイプラインをend-to-end実行し、docs CSV・図・観測セル構造を確認することで行う（このリポジトリの既存慣習に一致）。

## Global Constraints

- 衝突の定義: `COLLISION_AMOUNT = ["ClaimAmountPartColl", "ClaimAmountTotColl"]`、`COLLISION_NB = ["ClaimNbPartColl", "ClaimNbTotColl"]`（部分衝突+全損衝突のみ）。
- 目的変数: `target="pure_premium"`（分子=衝突金額合計）| `target="frequency"`（分子=衝突件数合計）。露出は両者とも `ExposTotal`。
- 出力接尾辞: `sfx = "" if target=="pure_premium" else "_freq"`。純保険料率版は既存ファイル名を維持し、頻度版は `_freq` を付す。
- データ: `data/brvehins1_full.csv`、全製造業者 (`brand=None`)、セル露出 ≥ 100、車種総露出 > 10。
- 既存 `CLAIM_TYPES`（全5金額列）は**変更しない**（`load_standardized_relativity` が参照。温存のみ、本分析では未使用）。
- 日本語版のみ。英語版 (`..._Class_Ratemaking.md`) は本計画の対象外。
- コミットはユーザー指示があるまで行わない（ハーネス規則）。各タスクの "Commit" ステップは指示があった場合のみ実行。

---

### Task 1: `src/ratemaking.py` — 衝突限定ローダ

**Files:**
- Modify: `src/ratemaking.py:21-24`（定数追加）, `:63-98`（`load_pure_premium` を `load_cell_matrix` + ラッパに再構成）

**Interfaces:**
- Produces:
  - `COLLISION_AMOUNT: list[str]`, `COLLISION_NB: list[str]`
  - `load_cell_matrix(csv_path="data/brvehins1_full.csv", brand=None, target="pure_premium", cell_exposure_min=100, model_exposure_min=10) -> (rate: DataFrame, exposure_total: DataFrame)`
  - `load_pure_premium(...) -> (rate, exposure_total)`（`load_cell_matrix(target="pure_premium", ...)` の薄いラッパ、既存importを非破壊に）

- [ ] **Step 1: 衝突定数を追加**

`src/ratemaking.py` の `CLAIM_TYPES = [...]` ブロック直後（24行目の `]` の後）に追加:

```python
# Collision-only components (部分衝突 + 全損衝突). The active analysis targets
# collision claims: heavy-tail perils (theft/total-loss/fire) dominate the L2
# loss and "Other" swamps the frequency signal, so restricting the numerator to
# collision gives a meaningful, well-conditioned target for both pure premium
# and frequency. CLAIM_TYPES (all 5) is retained only for load_standardized_relativity.
COLLISION_AMOUNT = ["ClaimAmountPartColl", "ClaimAmountTotColl"]
COLLISION_NB = ["ClaimNbPartColl", "ClaimNbTotColl"]
```

- [ ] **Step 2: `load_pure_premium` を `load_cell_matrix` + ラッパに置換**

`src/ratemaking.py:63-98`（`def load_pure_premium(...)` 全体、docstring・本体・`return pure_premium, exposure_total` まで）を次で置換:

```python
def load_cell_matrix(csv_path="data/brvehins1_full.csv", brand=None,
                     target="pure_premium", cell_exposure_min=100,
                     model_exposure_min=10):
    """Build the vehicle-model x region rate and exposure matrices.

    Reproduces the R preprocessing (Sections 2-3): optionally filter to one
    brand (`brand=None` keeps every manufacturer), aggregate the COLLISION
    claim numerator and exposure to a model x region matrix, keep only cells
    with exposure >= `cell_exposure_min` (others become NaN = missing) and
    models whose total exposure exceeds `model_exposure_min`.

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

    cats = ["VehModel", "Area"]
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
    """Collision pure-premium rate matrix — thin wrapper over load_cell_matrix.

    Kept so existing imports (`from ratemaking import load_pure_premium`) stay
    valid. Returns (pure_premium, exposure_total).
    """
    return load_cell_matrix(csv_path=csv_path, brand=brand, target="pure_premium",
                            cell_exposure_min=cell_exposure_min,
                            model_exposure_min=model_exposure_min)
```

- [ ] **Step 3: import順の健全性を確認**

`load_cell_matrix` は `load_bravehins` と `get_total` を呼ぶ。両者は同ファイル内で `load_cell_matrix` より前（31行目・45行目）に定義済み。`COLLISION_*` 定数はファイル冒頭（`CLAIM_TYPES` 直後）に定義済み。前方参照なし。

- [ ] **Step 4: 両targetでローダが動くことを検証**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && python -c "
from src.ratemaking import load_cell_matrix
for t in ['pure_premium', 'frequency']:
    r, e = load_cell_matrix(target=t)
    import numpy as np
    obs = (~r.isna()).to_numpy().sum()
    v = r.to_numpy()[~np.isnan(r.to_numpy())]
    print(t, 'shape', r.shape, 'obs', obs, 'rate[min,median,max]',
          round(float(v.min()),4), round(float(np.median(v)),4), round(float(v.max()),2))
"
```
Expected: 両target共に `shape (1049, 40)` 前後・`obs ≈ 8637`（露出フィルタは目的変数非依存なので一致）。`pure_premium` の rate は通貨スケール（median 数百〜千、max 数万）、`frequency` は 0〜数程度（median < 1）。両者 min ≥ 0。

- [ ] **Step 5: Commit（ユーザー指示時のみ）**

```bash
git add src/ratemaking.py
git commit -m "feat(ratemaking): collision-only load_cell_matrix with switchable target"
```

---

### Task 2: `src/brazil_data_analysis_R.py` — target を末端まで通す

**Files:**
- Modify: `src/brazil_data_analysis_R.py` — import (`:52-62`), `prepare_data` (`:102-128`), `run_comparison` (`:135-292`), `generate_paper_figures` (`:299-382`), `main` (`:385-399`)

**Interfaces:**
- Consumes: `load_cell_matrix` (Task 1)
- Produces:
  - `prepare_data(target="pure_premium")`
  - `run_comparison(pure_premium, pp_mat, exp_mat, W_full, U_mat, I_mat, target="pure_premium")`
  - `generate_paper_figures(pure_premium, pp_mat, exp_mat, obs_cells, W_full, best, ctx, target="pure_premium")`
  - `main(target="pure_premium")`（`sys.argv[1]` があれば上書き）

- [ ] **Step 1: import を差し替え**

`src/brazil_data_analysis_R.py:52-62` の import ブロックで `load_pure_premium,` を `load_cell_matrix,` に変更（アルファベット順の位置: `get_prediction,` の次）。`import sys` が無ければ `import os` の下（44行目付近）に追加:

```python
import os
import sys
```
import ブロック:
```python
from ratemaking import (
    build_side_info,
    get_prediction,
    load_cell_matrix,
    optimize_params,
    poisson_deviance,
    train_test_split,
    visualize_heatmap,
    visualize_scatter_plot,
    weighted_rmse,
)
```

- [ ] **Step 2: `prepare_data` に target を追加**

`src/brazil_data_analysis_R.py:102-128` の `def prepare_data():` シグネチャと本文冒頭を変更:

```python
def prepare_data(target="pure_premium"):
    """Return (rate df, rate_mat, exp_mat, obs_cells, W_full, U_mat, I_mat).

    The numerator is restricted to collision claims (部分衝突 + 全損衝突);
    `target="pure_premium"` uses collision claim amount / exposure, and
    `target="frequency"` uses collision claim count / exposure. `exp_mat`
    holds the total exposure, serving as the credibility weight / GLM offset.
    """
    pure_premium, exposure_total = load_cell_matrix(
        csv_path="data/brvehins1_full.csv", brand=None, target=target)
```

（以降 `pp_mat = ...` 以下は不変。返り値の変数名 `pure_premium`/`pp_mat` はrate行列を指す汎用名として維持し、下流の変更を最小化する。）

- [ ] **Step 3: `run_comparison` に target と出力接尾辞を追加**

`src/brazil_data_analysis_R.py:135` のシグネチャを変更:

```python
def run_comparison(pure_premium, pp_mat, exp_mat, W_full, U_mat, I_mat, target="pure_premium"):
```

docstring直後（`os.makedirs(FIG_DIR, exist_ok=True)` の前, 140行目付近）に追加:

```python
    sfx = "" if target == "pure_premium" else "_freq"
```

同関数内の**4つの出力パス**と**scatterタグ**に `sfx` を挿入:

`:251` →
```python
    comparison.to_csv(f"{DOCS_DIR}/model_comparison_python{sfx}.csv")
    print(f"saved {DOCS_DIR}/model_comparison_python{sfx}.csv")
```
`:272-273` →
```python
    strat.to_csv(f"{DOCS_DIR}/model_comparison_by_exposure_python{sfx}.csv", index=False)
    print(f"saved {DOCS_DIR}/model_comparison_by_exposure_python{sfx}.csv")
```
`:280-281` →
```python
    }).to_csv(f"{DOCS_DIR}/model_predictions_percell_python{sfx}.csv", index=False)
    print(f"saved {DOCS_DIR}/model_predictions_percell_python{sfx}.csv")
```
`:284-287`（scatterループ）→
```python
    max_lim = float(np.nanpercentile(act, 99)) or 1.0
    for name, pred in preds.items():
        tag = name.split()[0].lower()
        visualize_scatter_plot(act, pred, name, max_lim=max_lim,
                               fig_path=f"{FIG_DIR}/scatter_test_{tag}{sfx}.png")
```

（`max_lim` を act の99パーセンタイルで自動決定。頻度では通貨用の2500では空図になるため必須。`or 1.0` は全ゼロ異常時のフォールバック。）

- [ ] **Step 4: `generate_paper_figures` に target と接尾辞・自動軸を追加**

`src/brazil_data_analysis_R.py:299` のシグネチャを変更:

```python
def generate_paper_figures(pure_premium, pp_mat, exp_mat, obs_cells, W_full, best, ctx, target="pure_premium"):
```

docstring直後（`os.makedirs(FIG_DIR, exist_ok=True)` の前, 305行目付近）に追加:

```python
    sfx = "" if target == "pure_premium" else "_freq"
    # target-aware plot ceiling: currency for pure premium, small for frequency.
    hmax = float(np.nanpercentile(pp_mat[~np.isnan(pp_mat)], 99)) or 1.0
    smax = float(np.nanpercentile(ctx["act"], 99)) or 1.0
```

同関数内の全 `visualize_heatmap(...)` 呼び出しに `max_limit=hmax` を追加し、`f"{PAPER_DIR}/fig_*.png"` を `f"{PAPER_DIR}/fig_*{sfx}.png"` に変更。全 `visualize_scatter_plot(...)` に `max_lim=smax` を追加し、paper図パスに `{sfx}` を挿入。対象行と変更後:

`:324-327`（FIG_DIR診断ヒートマップ、接尾辞不要・軸のみ）→
```python
    visualize_heatmap(_honda(pure_premium), "actual (Honda)", max_limit=hmax,
                      fig_path=f"{FIG_DIR}/heatmap_actual{sfx}.png")
    visualize_heatmap(_honda(_to_df(estimated_mf)), "pred: MF (weighted, Honda)",
                      max_limit=hmax, fig_path=f"{FIG_DIR}/heatmap_mf{sfx}.png")
```
`:329-338`（paper図）→
```python
    visualize_heatmap(_honda(pure_premium),
                      "Actual Claim Costs by Vehicle Model and Region (Honda)",
                      max_limit=hmax, fig_path=f"{PAPER_DIR}/fig_4_2_1{sfx}.png")
    visualize_scatter_plot(act, mf_pred, "Matrix Factorization", max_lim=smax,
                           fig_path=f"{PAPER_DIR}/fig_4_5_1{sfx}.png")
    visualize_scatter_plot(act, ctx["cmf_pred"], "Collective Matrix Factorization",
                           max_lim=smax, fig_path=f"{PAPER_DIR}/fig_4_6_1{sfx}.png")
    visualize_heatmap(_honda(_to_df(estimated_mf)),
                      "Estimated Pure Premium Rates (Matrix Factorization, Honda)",
                      max_limit=hmax, fig_path=f"{PAPER_DIR}/fig_4_5_2{sfx}.png")
```
`:353-355`（GLM観測セル）→
```python
    visualize_heatmap(_honda(_to_df(g_obs)),
                      "Estimated Pure Premium Rates -- GLM (Honda; white = missing)",
                      max_limit=hmax, fig_path=f"{PAPER_DIR}/fig_4_3_2{sfx}.png")
```
`:375-377`（GLM全セル）→
```python
    visualize_heatmap(_honda(_to_df(glm_all_flat.reshape(len(models), len(areas)))),
                      "Predicted Pure Premium Rates -- Main-Effects GLM (Honda, all cells)",
                      max_limit=hmax, fig_path=f"{PAPER_DIR}/fig_4_3_1{sfx}.png")
```

- [ ] **Step 5: `main` に target を追加**

`src/brazil_data_analysis_R.py:385-399` を置換:

```python
def main(target="pure_premium"):
    pure_premium, pp_mat, exp_mat, obs_cells, W_full, U_mat, I_mat = prepare_data(target)
    best, ctx = run_comparison(pure_premium, pp_mat, exp_mat, W_full, U_mat, I_mat, target)
    generate_paper_figures(pure_premium, pp_mat, exp_mat, obs_cells, W_full, best, ctx, target)

    print("\n================ SUMMARY ================")
    print(f"target           : {target}")
    print(f"best (k, lambda) : ({best['k']}, {best['lambda']})")
    print(f"eval test cells  : {ctx['n_eval']}")
    print(ctx["comparison"].to_string())
    print("(All three models scored on the identical held-out cells; MF loss is "
          "exposure-weighted to match GLM/GLMM.)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pure_premium"
    main(target)
```

- [ ] **Step 6: 構文チェック**

Run: `cd /Users/spectee/projects/act-cmfreq && python -c "import ast; ast.parse(open('src/brazil_data_analysis_R.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit（ユーザー指示時のみ）**

```bash
git add src/brazil_data_analysis_R.py
git commit -m "feat(pipeline): thread target through pipeline, _freq outputs, auto plot limits"
```

---

### Task 3: `src/glmm_pymc.py` — target 対応

**Files:**
- Modify: `src/glmm_pymc.py:39`（import）, `:42-108`（`main` に target・接尾辞・自動軸）

**Interfaces:**
- Consumes: `load_cell_matrix` (Task 1)
- Produces: `main(target="pure_premium")`（`sys.argv[1]` で上書き）

- [ ] **Step 1: import を差し替え**

`src/glmm_pymc.py:39` を:
```python
from ratemaking import load_cell_matrix, visualize_heatmap
```

- [ ] **Step 2: `main` に target・接尾辞・自動軸・ローダ切替**

`src/glmm_pymc.py:42-48` を置換:
```python
def main(target="pure_premium"):
    os.makedirs("docs", exist_ok=True)
    sfx = "" if target == "pure_premium" else "_freq"
    pure_premium, exposure_total = load_cell_matrix(target=target)
    pp = pure_premium.to_numpy(dtype=float)
    exp_mat = exposure_total.to_numpy(dtype=float)
    models = pure_premium.index.to_numpy()
    areas = pure_premium.columns.to_numpy()
```

- [ ] **Step 3: docs CSV とヒートマップに接尾辞・自動軸**

`src/glmm_pymc.py:97`（summary CSV）を:
```python
    }).to_csv(f"docs/glmm_pymc_summary{sfx}.csv", index=False)
    print(f"saved docs/glmm_pymc_summary{sfx}.csv")
```
`:100-108`（ヒートマップ2枚）を:
```python
    grid = np.full(pp.shape, np.nan)
    grid[r, c] = rate_mean
    grid_df = pd.DataFrame(grid, index=pure_premium.index, columns=pure_premium.columns)
    hmax = float(np.nanpercentile(rate_mean, 99)) or 1.0
    visualize_heatmap(grid_df, "Estimated Pure Premium Rates -- GLMM (pymc, observed cells)",
                      max_limit=hmax, fig_path=f"paper/fig_4_4_1{sfx}.png")
    visualize_heatmap(grid_df, "Estimated Pure Premium Rates -- GLMM (pymc, white = missing)",
                      max_limit=hmax, fig_path=f"paper/fig_4_4_2{sfx}.png")
    print(f"saved paper/fig_4_4_1{sfx}.png, paper/fig_4_4_2{sfx}.png")
```

- [ ] **Step 4: `__main__` で target を受け取る**

`src/glmm_pymc.py:111-112` を:
```python
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pure_premium"
    main(target)
```
（`import sys` は既に39行目付近で存在。無ければ追加。）

- [ ] **Step 5: 構文チェック**

Run: `cd /Users/spectee/projects/act-cmfreq && python -c "import ast; ast.parse(open('src/glmm_pymc.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit（ユーザー指示時のみ）**

```bash
git add src/glmm_pymc.py
git commit -m "feat(glmm): target support with _freq outputs"
```

---

### Task 4: 両target（衝突限定）をend-to-end実行し数値・図を生成

**Files:**
- 生成/上書き: `docs/model_comparison_python.csv` / `..._freq.csv`, `docs/model_comparison_by_exposure_python.csv` / `..._freq.csv`, `docs/model_predictions_percell_python*.csv`, `paper/fig_*.png` / `fig_*_freq.png`, `docs/glmm_pymc_summary*.csv`

**Interfaces:**
- Consumes: Tasks 1-3

- [ ] **Step 1: 純保険料率版を実行（重い; バックグラウンド）**

Run（macOSに `timeout` は無いので付けない。CV グリッドで数分〜十数分かかるためバックグラウンド実行し完了通知を待つ）:
```bash
cd /Users/spectee/projects/act-cmfreq && python src/brazil_data_analysis_R.py pure_premium
```
Expected: `saved docs/model_comparison_python.csv` 等 + SUMMARY に `target : pure_premium`、`eval test cells : ≈2068`。`paper/fig_4_2_1.png` 等（接尾辞なし）が更新される。数値を書き留める（表4.5.1 / 4.5.2 用）。

- [ ] **Step 2: 頻度版を実行（重い; バックグラウンド）**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && python src/brazil_data_analysis_R.py frequency
```
Expected: `saved docs/model_comparison_python_freq.csv` 等 + `paper/fig_4_2_1_freq.png` 等。頻度スケールの数値（0〜数程度）。GLMが最小誤差の想定。

- [ ] **Step 3: GLMMヒートマップを両target生成（pymc; 重い）**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && python src/glmm_pymc.py pure_premium && python src/glmm_pymc.py frequency
```
Expected: `paper/fig_4_4_1.png` / `fig_4_4_1_freq.png` 等 + `docs/glmm_pymc_summary.csv` / `..._freq.csv`。

- [ ] **Step 4: 観測セル構造の一致を検証**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && python -c "
import pandas as pd
a=pd.read_csv('docs/model_predictions_percell_python.csv')
b=pd.read_csv('docs/model_predictions_percell_python_freq.csv')
print('pp eval cells', len(a), '| freq eval cells', len(b), '| equal:', len(a)==len(b))
"
```
Expected: 両者の eval セル数が一致（露出フィルタは目的変数非依存）。`equal: True`。

- [ ] **Step 5: 生成された数値を1ファイルに集約（論文更新の参照用）**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && for f in docs/model_comparison_python.csv docs/model_comparison_python_freq.csv docs/model_comparison_by_exposure_python.csv docs/model_comparison_by_exposure_python_freq.csv; do echo "=== $f ==="; cat "$f"; echo; done
```
Expected: 4ファイルの中身が表示され、Task 5/6 の論文数値の出典として使える。

---

### Task 5: 純保険料率版JP論文を衝突・新数値へ更新

**Files:**
- Modify: `paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP.md`

**Interfaces:**
- Consumes: Task 4 の `docs/model_comparison_python.csv`, `docs/model_comparison_by_exposure_python.csv`

- [ ] **Step 1: 目的変数の記述を衝突限定に更新**

`:151` の「予測の目的変数は、過去の純保険料率（事故コスト＝総事故金額÷総露出）である。」を次に置換:
```
　予測の目的変数は、衝突事故（部分衝突＋全損衝突）の過去の純保険料率（事故コスト＝衝突事故の総事故金額÷総露出）である。全ペリルのうち衝突事故に限定するのは、盗難・全損・火災といった低頻度・高severityの裾ペリルがL2損失を支配し交互作用構造の推定を歪める一方、件数の大半を占める「その他」ペリルは平均severityが極めて小さく信号として希薄なためである。衝突事故（部分衝突＋全損衝突）は全事故金額の約6割・件数の約3割を占め、料率算定の信号として十分な量を持つ。
```

- [ ] **Step 2: §3.2 の目的変数記述を衝突限定に更新**

`:137` の「観測行列 X は事故コスト、具体的には過去の純保険料率（historical pure premium rate）から構成される。」を次に置換:
```
観測行列 X は事故コスト、具体的には衝突事故（部分衝突＋全損衝突）の過去の純保険料率（historical pure premium rate）から構成される。
```

- [ ] **Step 3: §4.2 のデータ記述の観測セル数を実測値へ更新**

`:153` の「1,049車種 × 40地域、観測セル数8,637」を Task 4 Step 1 の実測 `matrix shape` / `observed cells` に置換（現状の設計では 1,049×40・8,637 のはず。実行結果と異なれば実測値に合わせる）。

- [ ] **Step 4: §4.5 MF の交差検証記述を新 (k, λ) へ更新**

`:243` の「選択値は $k=2$, $\lambda=100$」を Task 4 Step 1 の SUMMARY `best (k, lambda)` に置換。同段落の露出加重RMSE基準の記述は現行のままでよい（基準は不変）。

- [ ] **Step 5: §4.6 CMF の記述を新数値へ更新**

`:281` の「選択された重みは $(1.0,\ 0.25,\ 0.25)$、$(k, \lambda) = (29, 50)$」を実行ログの `best_params (CMF+side info)` と `weights(m,u,i)` に置換。
`:289` と `:335` の「ポアソン逸脱度は素の行列分解を大きく下回り（$5.10\times10^8 \to 2.88\times10^8$、約44%減…）」を、Task 4 Step 1 の `docs/model_comparison_python.csv` の MF・CMF の `PoissonDeviance` 実測値へ置換。**CMFがMFを下回らない場合は「約44%減」等の主張を削除し、実測の大小関係を正直に記述する**（spec §成果物(a)）。

- [ ] **Step 6: 表4.5.1 を実測値へ置換**

`:301-306` の表を `docs/model_comparison_python.csv`（列: RMSE, wRMSE(exposure), PoissonDeviance / 行: GLM, GLMM=GLMと同値, 行列分解=MF, CMF）の実測値へ置換。`:299` の表キャプション内のハイパーパラメータ `k=2, λ=100` / `k=29, λ=50, (1.0,0.25,0.25)` も新値へ。

- [ ] **Step 7: 表4.5.2 を実測値へ置換**

`:320-325` の表を `docs/model_comparison_by_exposure_python.csv`（sparse/dense × GLM/MF/CMF の wRMSE・PoissonDev）の実測値へ置換。

- [ ] **Step 8: §4.5/§4.7/§5 の本文の大小関係を実測に合わせて記述**

`:312`, `:314`, `:327`, `:331`, `:335` の定性的主張（「スパース層でMFがGLMをわずかに上回る」「密層でGLMが先行」「CMF逸脱度が密層で素MFを下回る」等）を、Task 4 の実測表と照合し、成り立つものは数値を差し替え、成り立たないものは記述を修正・削除する。数値（例 `2024.2 対 2094.0`, `993.7 対 1325.0`, `2.15\times10^8 対 4.51\times10^8`）はすべて実測へ。

- [ ] **Step 9: 図の整合を確認**

`paper/ICA2026..._JP.md` が参照する図（`fig_4_2_1.png`〜`fig_4_6_1.png`、接尾辞なし）は Task 4 Step 1・Step 3 で上書き済み。ファイル名の参照は変更不要。

- [ ] **Step 10: 数値の相互確認**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && grep -nE "10\^8|×10|\$k *=|\\\\lambda|中央値|観測セル数" paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP.md | head -40
```
Expected: 残存する数値がすべて Task 4 の docs CSV と一致していること（旧 standardized-relativity 由来の 1899.0 / 1198.1 / 1.25×10⁸ 等が残っていないこと）を目視確認。

- [ ] **Step 11: Commit（ユーザー指示時のみ）**

```bash
git add paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP.md paper/fig_*.png docs/model_comparison*.csv docs/model_predictions_percell_python.csv docs/glmm_pymc_summary.csv
git commit -m "docs(paper-JP): collision-only pure-premium numbers & figures"
```

---

### Task 6: 頻度版JP論文を新規作成

**Files:**
- Create: `paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP_frequency.md`

**Interfaces:**
- Consumes: Task 5 の更新済み純保険料率版JP（複製元）, Task 4 の `docs/model_comparison_python_freq.csv`, `docs/model_comparison_by_exposure_python_freq.csv`

- [ ] **Step 1: 更新済み純保険料率版を複製**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && cp paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP.md paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP_frequency.md
```

- [ ] **Step 2: 図参照を `_freq` に差し替え**

`paper/..._JP_frequency.md` 内の全図参照 `fig_4_X_Y.png` を `fig_4_X_Y_freq.png` に置換（`fig_4_2_1`, `4_3_1`, `4_3_2`, `4_4_1`, `4_4_2`, `4_5_1`, `4_5_2`, `4_6_1` の Markdown 埋め込み `![...](....png)` 両所）。Edit で1件ずつ、または:
```bash
cd /Users/spectee/projects/act-cmfreq && sed -i '' -E 's/\(fig_(4_[0-9]_[0-9])\.png\)/(fig_\1_freq.png)/g' paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP_frequency.md
```
確認: `grep -c "_freq.png" paper/..._JP_frequency.md` が図の枚数（8）と一致。

- [ ] **Step 3: 翻訳注記を頻度版向けに修正**

`:19` の翻訳注記の対象ファイル名を英語正本のままにしつつ、本ファイルが頻度分析版である旨を1文追記:
```
> **本版について：** 本ファイルは目的変数を「衝突事故（部分衝突＋全損衝突）のクレーム頻度（＝衝突事故件数÷総露出）」とした頻度分析版である。純保険料率版は `ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP.md` を参照。数式構造は共通で、目的変数と数値・図のみが異なる。
```

- [ ] **Step 4: 目的変数の記述を頻度へ差し替え**

`:151`（純保険料率版で Step 1 更新済みの箇所）を頻度版向けに置換:
```
　予測の目的変数は、衝突事故（部分衝突＋全損衝突）の過去のクレーム頻度（＝衝突事故の総事故件数÷総露出）である。全ペリルのうち衝突事故に限定するのは、件数の大半を占める「その他」ペリルは平均severityが極めて小さく頻度分析の信号として希薄で、これに埋もれると意味のある頻度モデルにならないためである。衝突事故（部分衝突＋全損衝突）は全事故件数の約3割を占め、頻度分析の信号として十分な量を持つ。
```
`:137`（§3.2）を:
```
観測行列 X はクレーム頻度、具体的には衝突事故（部分衝突＋全損衝突）の過去のクレーム頻度（historical claim frequency）から構成される。
```

- [ ] **Step 5: 目的変数の呼称を全文で頻度へ統一**

`paper/..._JP_frequency.md` 内の「純保険料率」「事故コスト」「事故金額」「pure premium」を、文脈に応じて「クレーム頻度」「事故件数」「claim frequency」へ置換する。GLM/GLMMの目的変数の数式は同形（`Y_ij ~ Poisson(λ_ij E_ij)`、`Y_ij`=衝突事故件数）なので数式自体は変更しないが、`Y_{ij}` の語義説明「総事故金額」→「総事故件数」、「期待総事故コスト」→「期待総事故件数」に修正。図キャプションの「純保険料率」→「クレーム頻度」。**通貨に固有の表現**（「目的変数の最大値は約111,000」等）は頻度スケールの実測値へ差し替えるか、頻度に合わせて書き換える。Edit で該当箇所を1件ずつ処理。

- [ ] **Step 6: 表4.5.1 を頻度実測値へ置換**

`docs/model_comparison_python_freq.csv` の実測値で表4.5.1を置換。キャプションのハイパーパラメータも頻度ランの値へ。

- [ ] **Step 7: 表4.5.2 を頻度実測値へ置換**

`docs/model_comparison_by_exposure_python_freq.csv` の実測値で表4.5.2を置換。

- [ ] **Step 8: §4.5/§4.6/§4.7/§5 の定性的主張を頻度ランの実測に合わせる**

CMF の逸脱度削減率、スパース/密層の大小関係を頻度ランの実測表と照合し、成り立つものは数値を差し替え、成り立たないものは修正・削除（純保険料率版 Step 5・Step 8 と同じ手順を頻度データで）。

- [ ] **Step 9: 旧数値の残存チェック**

Run:
```bash
cd /Users/spectee/projects/act-cmfreq && grep -nE "純保険料率|事故金額|事故コスト|111,000|1899|1198" paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP_frequency.md | head -40
```
Expected: 純保険料率・金額系の語や旧数値が本文（目的変数として）に残っていないこと。参考文献名や一般論の「純保険料」など置換すべきでない箇所は残ってよいが、目的変数を指す用法は全て頻度へ。

- [ ] **Step 10: Commit（ユーザー指示時のみ）**

```bash
git add paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP_frequency.md paper/fig_*_freq.png docs/model_comparison*_freq.csv docs/model_predictions_percell_python_freq.csv docs/glmm_pymc_summary_freq.csv
git commit -m "docs(paper-JP): add collision claim-frequency paper version"
```

---

## Self-Review

**Spec coverage:**
- ratemaking.py 衝突定数 + `load_cell_matrix(target=...)` + `load_pure_premium` ラッパ → Task 1 ✓
- brazil_data_analysis_R.py target 通し + `_freq` 出力 + target対応プロット軸 → Task 2 ✓
- glmm_pymc.py target 対応 + `_freq` 出力 → Task 3 ✓
- 両target（衝突限定）再実行 → Task 4 ✓
- 純保険料率版JP更新（数値・図・narrative） → Task 5 ✓
- 頻度版JP新規作成 → Task 6 ✓
- エッジケース（衝突0セル→rate 0 有効）: `load_cell_matrix` は 0 を欠測にしない（Task 1 docstring明記）、`poisson_deviance` の `y>0` ガード既存 ✓
- `CLAIM_TYPES` 参照モジュール確認: grep 済み。参照は `ratemaking.py` 内 `load_standardized_relativity` のみ（温存）、パイプラインは `load_cell_matrix`/`load_pure_premium` 経由 ✓
- スコープ外（英語版・Tweedie・標準化削除）: 各タスクとも触れない ✓

**Placeholder scan:** 実測値に置換する箇所（Task 5/6）は出典CSVを明示し、大小関係が変わった場合の分岐（正直に記述/削除）を指定済み。実行前に確定できない数値をプレースホルダにはせず「Task 4 の docs CSV から転記」と手順化。

**Type consistency:** `load_cell_matrix` の返り値 `(rate, exposure_total)` は `load_pure_premium` と同一形状。`prepare_data`/`run_comparison`/`generate_paper_figures`/`main` の `target` 引数名・既定値 `"pure_premium"` は全タスクで一貫。`sfx` の算出式 `"" if target=="pure_premium" else "_freq"` は Task 2/3 で同一。
