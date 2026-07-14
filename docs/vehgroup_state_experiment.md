# データ粒度を替えた探索：VehGroup 行 / State 列（純率）

> **位置づけ**：探索段階のメモ。実際に採用するかは未定のため、論文本線
> （`paper/`）や正規の比較結果（`docs/model_comparison_python.csv`）には
> 反映していない。ここに数値と再現手順だけ残す。

## 動機

正規分析（`src/brazil_data_analysis_R.py`、行 = 個別車種 VehModel、列 = 地域 Area）
では、保留セルの露出加重RMSE・Poisson deviance で **MF が主効果 GLM に負けていた**。
行列がトリムレベルで極端に疎なことが一因と見て、行列の粒度を粗くして密度を上げる
データ変種をいくつか試した。

- 行軸：VehModel（約4,200種、疎）→ **VehGroup**（約436種のモデルファミリー）
- 列軸：Area（40区分）→ **State**（27の連邦単位。Area は State を都市圏で細分化したもの）

前処理・フィルタ（衝突クレーム分子、セル露出≥100、行総露出>10）、分割→チューニング
の順序（リーク回避）、露出加重の CMF 損失、(k, λ) の CV グリッド、評価指標は
すべて正規パイプラインと同一。CMF（サイド情報あり）は、行が VehGroup になると
行サイド情報（VehGroup ワンホット）が単位行列に退化し情報を持たないため除外し、
中核比較の MF / GLM / GLMM のみを載せる。

## 結果（純率、保留セル、露出加重が本命指標）

各構成内の MF vs GLM 比較が妥当（構成間で評価セルが異なるため絶対値の横比較は不可）。

| 行 × 列 | 行列サイズ / 観測セル | 選択 (k, λ) | MF wRMSE | GLM wRMSE | MF vs GLM |
|---|---|---|---|---|---|
| VehModel × Area（正規） | 疎・大 | — | 190.5 | 172.5 | +10.4%（負け） |
| VehGroup × Area | 210×40 / 3,289 | (2, 50) | 181.5 | 178.2 | +1.9%（僅差負け） |
| **VehGroup × State** | 231×27 / 2,233 | (2, 10) | **167.3** | 181.0 | **−7.6%（勝ち）** |

### VehGroup × State の詳細（`docs/model_comparison_python_vehgroup_state.csv`）

| 指標 | MF (weighted) | GLM / GLMM | 勝者 |
|---|---|---|---|
| RMSE | 373.9 | 378.0 | MF |
| wRMSE(露出加重) | 167.3 | 181.0 | **MF** |
| Poisson deviance | 3.92e7 | 4.20e7 | **MF** |

露出層別（`docs/model_comparison_by_exposure_python_vehgroup_state.csv`）でも両層で MF が勝利：

| 層 | n | MF wRMSE | GLM wRMSE |
|---|---|---|---|
| 疎（露出 < 中央値） | 273 | 401.6 | 425.4 |
| 密（露出 ≥ 中央値） | 274 | 142.3 | 155.4 |

## 解釈と注意（過大評価しないために）

- **VehGroup × State が、MF が GLM を全指標で上回った初めての構成**。論文の中心主張
  （MF が車種×地域の交互作用を捉える）を支持する方向。
- ただし CV は依然 **k=2** の低ランクを選択。勝因は「高次元交互作用の捕捉」よりも、
  密な行列で少数の潜在因子が GLM 主効果より地域×車種の粗い信号を**縮約（credibility 的な
  収縮）**できたこと、に近い解釈が自然。露出の少ない層でも MF が勝っている点がそれを示唆。
- 3構成の評価セルは異なるので、絶対値の改善（190.5→167.3 等）を一直線の改善と読んではいけない。

## 補強検証の結果（`src/vehgroup_state_validation.py`）

上の「勝ち」が本物か、3つの頑健性チェックを実施した。**結論：頑健ではない。**
seed 123 では勝つが、シードを替えると優位が崩れる。

### A. k-vs-error 曲線（`docs/validation_k_curve_vehgroup_state.csv`、図 `figs/python_port/validation_k_curve_vehgroup_state.png`）
MF が保留 wRMSE で GLM を下回るのは **k=1 と k=2 のみ**。k=2 で最小（167.3 < GLM 181.0）、
k≥3 では急悪化（k=3 で 327）。最適ランクが 2 という極低ランクで、勝因は
「高次元交互作用の捕捉」ではなく少数因子による**縮約効果**であることを裏づける。

### B. マスクセル復元（観測済み高露出セルの 20% を隠して復元、5 シード）
`docs/validation_masked_recovery_vehgroup_state.csv`。

| seed | MF wRMSE | GLM wRMSE | 勝者 |
|---|---|---|---|
| 123 | 90.7 | 96.6 | MF |
| 1 | 208.0 | 144.0 | GLM（大敗） |
| 7 | 88.4 | 98.9 | MF |
| 42 | 93.2 | 93.3 | MF（僅差） |
| 2024 | 115.8 | 122.9 | MF |

**MF 勝 4/5**。復元は比較的好意的だが seed 1 で大きく崩れ、決定打にならない。

### C. 複数シードの保留比較（本命、`docs/validation_multiseed_vehgroup_state.csv`）

| seed | 選択 k | MF wRMSE | GLM wRMSE | MF−GLM |
|---|---|---|---|---|
| 123 | 2 | 167.3 | 181.0 | −13.7（勝） |
| 1 | 2 | 189.8 | 169.6 | +20.2（負） |
| 7 | 2 | 220.2 | 133.4 | +86.8（大敗） |
| 42 | 2 | 149.5 | 155.5 | −6.1（勝） |
| 2024 | 2 | 161.1 | 162.5 | −1.5（勝） |

**MF 勝 3/5、平均 MF−GLM = +17.2（MF がむしろ平均で悪い）、SD 40.9 と極端に大きい**。

## 改訂結論

- 「VehGroup × State で MF が GLM に勝つ」は **単一分割（seed 123）のアーティファクト**の
  色が濃い。シード横断では平均的に MF がやや劣り、ばらつきが極端。
- 復元テストは MF 寄り（4/5）だが 1 回の大敗があり、頑健性の担保にはならない。
- k 曲線は k≤2 でしか勝てず、この構成に捕捉すべき交互作用がほとんど無いことを示す。
- **現状のまま論文本線に採用するのは非推奨。** 交互作用シグナルを立たせる別レバー
  （標準化相対度を目的変数にする、ブランド絞り込み等）を先に試す方が筋が良い。

## 再現手順（検証）

```bash
python3 src/vehgroup_state_validation.py   # A/B/C をまとめて実行
```

## 再現手順

```bash
# VehGroup × State（純率）
python3 src/vehgroup_variant.py pure_premium State

# 参考：VehGroup × Area（純率 / 頻度）
python3 src/vehgroup_variant.py pure_premium
python3 src/vehgroup_variant.py frequency
```

行・列の粒度は `src/ratemaking.py::load_cell_matrix(row_col=..., col_col=...)` で切替。
出力は `docs/model_comparison_python_vehgroup*.csv`（正規結果は非上書き）。
