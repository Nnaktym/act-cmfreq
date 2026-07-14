# 設計: 目的変数の切り替え(純保険料率 ⇄ 頻度)＋衝突限定＋頻度版JP論文

**日付:** 2026-07-12
**ブランチ:** review-revisions-center-fix
**関連:** レビュー指摘 Major #2(記述と実装の目的変数の不一致)への対応

---

## 背景と目的

レビューで、論文が記述する目的変数(純保険料率 = 総事故金額 ÷ 総露出)と、コードが実際に用いる目的変数(人口統計標準化した相対度 claim/E\*)が食い違っていることが判明した。ユーザーの決定により、

1. 目的変数を論文記述どおりの**素の純保険料率**に戻す(標準化相対度を廃止)。
2. さらに、**クレーム頻度**(件数ベース、ポアソン)を分析する版を**別途**作る。目的変数は**引数で切り替え可能**にする。
3. 目的変数の分子は**衝突事故のみ**(部分衝突 PartColl + 全損衝突 TotColl)に限定する。金額版・頻度版の両方に同じ衝突定義を適用。
4. 日本語版論文を**2本並存**させる(純保険料率版=既存を更新、頻度版=新規)。英語版の同期は本スペックの対象外(別途)。

### 衝突限定の根拠(全データのペリル構成)

| ペリル | 件数share | 金額share | 平均severity |
|---|---:|---:|---:|
| 盗難(Rob) | 4.0% | 31.5% | 24,724 |
| 部分衝突(PartColl) | 29.6% | 35.4% | 3,738 |
| 全損衝突(TotColl) | 2.4% | 26.0% | 33,255 |
| 火災(Fire) | 0.1% | 0.8% | 23,384 |
| その他(Other) | 63.8% | 6.2% | 304 |

- 「その他」は件数の63.8%を占めるが平均severityは304。全ペリルで頻度分析すると「その他」の頻度に埋もれる → 衝突限定で意味のある頻度になる。
- 盗難・全損・火災は低頻度・高severityの裾ペリルで金額の約58%を占め、L2損失のMFを支配する(レビュー指摘の弱点)。衝突限定で裾支配が緩和。
- 衝突(部分+全損)= 件数32%・金額61%で十分な信号量。
- セルの観測判定は総露出基準のため、ペリル限定でも観測セル集合は不変(比較の一貫性を保つ)。

---

## アーキテクチャ(アプローチA: 単一 `target` 引数)

既存パイプライン(`prepare_data`→`run_comparison`→`generate_paper_figures`)は目的変数に非依存で、`(rate行列, exposure行列)` だけで動く。純保険料率と頻度の唯一の違いは**分子**(衝突金額の合計 vs 衝突件数の合計)。露出加重・GLMオフセット・ポアソン逸脱度の計算はそのまま流用できる。したがってスイッチは**ローダに集約**する。

```
CSV → load_cell_matrix(target) → (rate, exposure)
    → train_test_split(先) → CV tune(k, λ; CMFは重みも)
    → MF/CMF/GLM/GLMM fit → metrics(docs, target接尾辞)
    → figures(paper, target接尾辞) → 論文本文
```

---

## コンポーネント

### 1. `src/ratemaking.py` — ローダ

- 定数を衝突限定に変更・追加:
  - `COLLISION_AMOUNT = ["ClaimAmountPartColl", "ClaimAmountTotColl"]`
  - `COLLISION_NB = ["ClaimNbPartColl", "ClaimNbTotColl"]`
  - 既存 `CLAIM_TYPES`(全5金額列)は**変更しない**(温存する `load_standardized_relativity` が参照するため)。アクティブな経路は新設の `COLLISION_*` 定数のみを使う。
- 新設 `load_cell_matrix(csv_path="data/brvehins1_full.csv", brand=None, target="pure_premium", cell_exposure_min=100, model_exposure_min=10)`:
  - `target="pure_premium"`: 分子 = `COLLISION_AMOUNT` の行合計(保険金額)
  - `target="frequency"`: 分子 = `COLLISION_NB` の行合計(件数)
  - どちらも `get_total(..., "ExposTotal", cell_exposure_min)` で露出行列を作り、`分子 / ExposTotal` を rate とする。models フィルタ(総露出 > `model_exposure_min`)は共通。
  - 返り値 `(rate, exposure_total)`(両者とも wide DataFrame、index/columns 共有)。
- `load_pure_premium(...)` は `load_cell_matrix(target="pure_premium", ...)` を呼ぶ薄いラッパとして残す(既存importの非破壊)。
- `load_standardized_relativity` は本分析では未使用となるが、削除はしない(温存)。

**インターフェース契約:** 呼び出し側は目的変数の種類を `target` 文字列でのみ指定し、返り値の形状・意味(rate と weight)は target に依らず同一。

### 2. `src/brazil_data_analysis_R.py` — 分析パイプライン

- `prepare_data(target="pure_premium")`: `load_cell_matrix(csv_path="data/brvehins1_full.csv", brand=None, target=target)` を呼ぶ。以降のロジック(W_full = 露出を平均1に正規化、side info 構築)は不変。
- `run_comparison(..., target=...)`: ロジック不変。出力パスに target 接尾辞:
  - `docs/model_comparison_python{sfx}.csv`
  - `docs/model_comparison_by_exposure_python{sfx}.csv`
  - `docs/model_predictions_percell_python{sfx}.csv`
  - `figs/python_port/scatter_test_*{sfx}.png`
  - ここで `sfx = "" if target=="pure_premium" else "_freq"`。
- `generate_paper_figures(..., target=...)`: paper 図の出力名に target 接尾辞:
  - pure_premium: `paper/fig_4_2_1.png` 等(既存名)
  - frequency: `paper/fig_4_2_1_freq.png` 等
- `main(target="pure_premium")`: `target = sys.argv[1]` があれば上書き(`"pure_premium"` | `"frequency"`)。
- **プロット軸上限を target 対応に**: 現状ハードコードの `max_lim=2500` / `max_limit=5000` は通貨スケール専用。target 別にデータ分位点(例: 観測値の99パーセンタイル)から自動決定する引数を渡す。頻度は値域が小さく、そのままだと空図になるため必須。
- GLM は `claim ~ C(VehModel) + C(Area)` + offset `log(exposure)`、ポアソン。頻度時は "claim" = 件数となり、カウントに対する自然な頻度GLMになる。`poisson_deviance(rate*exp, pred*exp)` は頻度時に `poisson_deviance(件数, 予測件数)` となり適切。

### 3. `src/glmm_pymc.py` — GLMMヒートマップ(§4.4)

- 同じ `target` 引数を追加し、`load_cell_matrix(target=...)` を使用。
- 出力を target 接尾辞に: `paper/fig_4_4_1{sfx}.png` / `paper/fig_4_4_2{sfx}.png`。

---

## 成果物(日本語版論文2本、いずれも衝突ベース)

### (a) 純保険料率版JP(既存を更新) — `paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP.md`

- 衝突限定・全製造業者で**再実行**して数値・図を差し替え。
- §4.2: 目的変数の記述を「衝突事故(部分衝突+全損衝突)の純保険料率 = 衝突保険金額 ÷ 総露出」に。ペリル限定の理由を1段落。
- 表4.5.1 / 表4.5.2、§4.6(k, λ, CMF重み, 逸脱度の記述)、§4.7 / §5 の本文を新数値へ。
- 結論の骨子は新ランに従って正直に記述。全ペリル版の予備結果では「観測セルでGLMが全指標・全層で最良、MFは両層で劣り、CMFは素MFより悪化」が見えており、衝突版で再確認して反映(スパース優位・CMF逸脱度44%減の旧主張は成り立たなければ削除)。

### (b) 頻度版JP(新規) — `paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking_JP_frequency.md`

- 更新後の(a)を複製し、目的変数の記述を差し替え:
  - 「純保険料率 / 事故コスト / 事故金額 / historical pure premium」→「クレーム頻度 / 事故件数 / claim frequency」
  - GLM/GLMM の目的変数説明を「総事故件数 Y_ij ~ Poisson(λ_ij E_ij)」(件数)に統一(数式自体はほぼ同形)。
- 頻度パイプライン(`target="frequency"`)実行後の数値・図(`_freq`)を反映。
- 章構成(GLM / GLMM / MF / CMF)は(a)と完全並行。結論は頻度ランを見て正直に。

---

## エッジケースとエラー処理

- 衝突限定により、露出≥100でも衝突クレームが0のセルが増える → rate=0。純保険料率0・頻度0 は有効値。`poisson_deviance` は `y>0` ガード済み、MF の非負制約とも整合。
- プロット軸上限の自動決定でゼロ割・空図を防ぐ(全観測値0の異常時は既定値にフォールバック)。
- `load_pure_premium` ラッパで既存呼び出しを非破壊に保つ。
- `CLAIM_TYPES` を参照する他モジュール(`brazil_data_analysis_python.py`, `glmm_pymc.py` 等)の有無を実装前に grep で確認し、衝突限定の影響範囲を確定。

---

## テスト / 検証

1. 両 target をend-to-end実行し、観測セル構造が**両版で同一**であることを確認(exposure フィルタは目的変数非依存)。期待: 観測 ≈ 8,637 / テスト 2,068 / 層別 1,034 ずつ(全ペリル版と同一; 値のみ変化)。
2. docs CSV が target 別に生成され、数値が有限・妥当(純保険料率は通貨スケール、頻度は0〜数程度、GLMが最小誤差になる想定)。
3. 各論文の掲載数値が、対応する docs CSV と一致することを確認(手動対応チェック)。
4. 図が target 別ファイルに出力され、2論文が互いの図を上書きしないこと。

---

## スコープ外(本スペックでは扱わない)

- 英語版(提出正本)の同期(レビュー Major #1)。別途対応。
- Tweedie / 頻度-severity 分解などの損失族拡張(将来課題)。
- 標準化相対度ロジックの削除(温存のみ)。
