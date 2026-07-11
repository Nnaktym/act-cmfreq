> **⚠️ HISTORICAL / SUPERSEDED (2026-07-11).** This report checked whether the
> original **R notebook** matched the paper *before* the review revisions. Its
> headline gap — "the 3 methods are not evaluated on the same test set" — has
> since been **fixed**: the analysis is now in Python
> (`src/brazil_data_analysis_R.py` + `src/glmm_pymc.py`), MF/GLM/GLMM are scored
> on one identical hold-out set with exposure-weighted metrics, and the paper was
> revised accordingly. See the repository `README.md` for the current state and
> findings. This document is retained only as a record of the pre-revision review.

# Notebook ↔ 論文 対応確認レポート

**対象論文:** `paper/ICA2026_Matrix_Factorization_for_Class_Ratemaking.md`
**対象分析:** `src/brazil_data_analysis_R.ipynb`（= `src/brazil_data_analysis_R.R` に変換済み、依存: `src/cmf.R`）
**作成日:** 2026-07-11
**目的:** notebook の分析内容が、現在論文に掲載されている分析内容と対応しているかの検証。

---

## 結論

notebook は論文の分析内容と **おおむね対応** している。特に MF（行列分解）パイプラインの数値アンカー（**k=22, λ=30**）は CV 結果の真の最小値と完全一致し、再現性は堅い。
ただし **比較評価の設計に、論文の比較優位の主張を裏づけられない重要な非対応が1点** ある（GLM/GLMM がテスト分割で評価されていない）。

---

## 1. 一致が確認できた部分（数値レベル）

| 論文の記述 | notebook の実装・出力 | 判定 |
|---|---|---|
| 全データ 4,259 車種 × 40 地域 | `VehModel` factor 4259 levels / `Area` 40 levels（cell 4 出力） | ✅ |
| Honda に絞り 48 車種 | 行フィルタ後 `[1] 48 40`（cell 20/21/25/26 出力） | ✅ |
| exposure ≥ 100 のセルのみ使用、他は欠測扱い | `get_total(..., "ExposTotal", 100)` で <100 を NA 化 | ✅ |
| 純率 = クレームコスト = 総クレーム / 総エクスポージャー | `pure_premium <- claim_total / exposure_total` | ✅ |
| GLM: Poisson + offset log(exposure)、交互作用なし | `pure_premium*exposure ~ VehModel + Area + offset(log(exposure))` | ✅ |
| GLMM: 交互作用をランダム効果 N(0, δ²) | `~ VehModel + Area + (1|interaction)`, poisson, offset | ✅ |
| MF: cmfrec, ALS, L2, nonneg=TRUE, center=FALSE | CMF 引数が一致 | ✅ |
| CV 4-fold グリッドサーチ → **k=22, λ=30** | 261 点探索の最小が **k=22, λ=30（CV RMSE=464.34）** | ✅ **完全一致** |
| ホールドアウト 25% テスト / 75% 学習 | `train_test_split`（ratio=0.75） | ✅ |
| Fig 4.5.1 予測 vs 実測 散布図 / Fig 4.5.2 全セルヒートマップ | cell 13 / cell 16 | ✅ |

**CV 上位点（notebook cell 8 出力より、261 点中）:**

| 順位 | k | λ | CV RMSE |
|---|---|---|---|
| 1 | 22 | 30 | 464.34 |
| 2 | 26 | 50 | 465.61 |
| 3 | 27 | 50 | 467.35 |
| 4 | 23 | 100 | 467.49 |
| 5 | 22 | 100 | 467.92 |

---

## 2. 非対応・要注意（重要度順）

### 【最重要】1) 3 手法が同一テストセットで評価されていない
- notebook では **MF だけ** が 75/25 ホールドアウトで **テスト RMSE=609.25**（cell 10）を算出。
- GLM（cell 19）と GLMM（cell 24）の散布図は `na.omit(all_data)` = **全観測セル（=学習データそのもの）への in-sample フィット**で、テスト分割に一度も触れていない（cell 24 のコメントも "for sanity check"）。
- → notebook 内に **「MF が GLM/GLMM より優れる」を裏づける同一分割・横並びの誤差比較が存在しない**。論文の比較優位の主張を、現状の notebook は支えていない。

### 2) テスト RMSE=609.25 が論文本文に未記載
数値は notebook にあるが、4.5 節本文は「RMSE で評価した」とのみで値が無い。

### 3) MF は exposure 非加重（GLM/GLMM は offset で加重）
notebook の CMF に `weight=` は無く非加重。論文とは整合するが、GLM/GLMM が offset で加重される一方 MF は非加重という **評価の非対称性が notebook にもそのまま存在**（ルート `cmf.r` 旧プロトタイプでは `weight=pt` を使用していたのと対照的）。

### 4) 「total exposure 10 or greater」の記述が実装とややズレ
論文は「総エクスポージャー 10 以上」と記述するが、コードは閾値 100 適用後の行に対し `rowSums(...) > 10`（≥ではなく厳密に >、かつ 100 未満セル除外後の合計）。結果 48 車種は合致するが、本文の言い回しは不正確。

### 5) notebook には GLMM 全セル外挿ヒートマップ（cell 26）があるが、論文は「外挿は省略」と記述
notebook cell 26 は `allow.new.levels=TRUE` で全組合せを予測（主効果に回帰）しヒートマップ化しているが、論文 Fig 4.4.1 のキャプションは「外挿は省略」。notebook にある成果物が論文に反映されていない。

### 6) side-info CMF（人口密度・車種グループ）は R notebook に無い
Python 版（`src/brazil_data_analysis_python.py`）のみに存在。論文でも結論の「Preliminary experiments (CMF)」として将来課題扱いのため整合はするが、R notebook 単体では再現不可。

---

## 3. まとめ

- **MF パイプライン（CV・ハイパーパラメータ・ホールドアウト・図）は論文と完全対応**。再現性は良好。
- **決定的なギャップは「比較の土台」**: GLM/GLMM がテスト分割で評価されておらず、論文の比較優位の主張を notebook が支えていない。
- 改善方針（別途の変更計画パート A）で、3 手法を **同一分割・exposure 加重あり** で評価する比較表を新規作成すれば、この穴を埋められる。

---

## 付録: 検証に用いた notebook 実行出力（抜粋）

- cell 4: `81115 obs. of 24 variables`（Honda 抽出後）、`VehModel` 4259 levels、`Area` 40 levels
- cell 6: 前処理後 `[1] 125 40`（行フィルタ前）
- cell 8: CV 261 点、最小 `k=22 λ=30 cv_score=464.3377`
- cell 10: MF テスト RMSE `609.2454`
- cell 15: MF 全データ学習 RMSE `11.5563`
- cell 20/21/25/26: 予測ヒートマップ次元 `[1] 48 40`
