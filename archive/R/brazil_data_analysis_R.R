# =============================================================================
# Brazil auto insurance data analysis (matrix factorization for class ratemaking)
# Converted from brazil_data_analysis_R.ipynb
# =============================================================================

# -----------------------------------------------------------------------------
# 1. ライブラリ・関数の読み込み
# -----------------------------------------------------------------------------
# install.packages(c("Matrix", "MatrixExtra", "recommenderlab", "cmfrec", "ggplot2", "reshape2", "viridis", "gridExtra"))
library(cmfrec)
library(dplyr)
library(lme4)
library(Matrix)
library(MatrixExtra)
library(recommenderlab)
library(cmfrec)
library(ggplot2)
library(reshape2)
library(viridis)
library(gridExtra)
source("cmf.R")

# -----------------------------------------------------------------------------
# 2. データの読み込み
# -----------------------------------------------------------------------------
brvehins <- load_bravehins(cas_dataset_path = "/Users/nanakato/projects/CASdatasets/data")
# brvehins <- load_bravehins(cas_dataset_path = "/Users/spectee/research/CASdatasets-master/data")
brvehins <- brvehins[grepl("Honda", brvehins$VehModel), ]
claim_types <- c("ClaimAmountRob", "ClaimAmountPartColl", "ClaimAmountTotColl", "ClaimAmountFire", "ClaimAmountOther")
brvehins[, "ClaimTotal"] <- rowSums(brvehins[, claim_types])
str(brvehins)

# write.csv(brvehins, file = "data/brvehins_org.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# 3. データの下処理
# -----------------------------------------------------------------------------
# category_to_analyze <- c("Gender", "DrivAge")
# category_to_analyze <- c("VehGroup", "VehYear")
category_to_analyze <- c("VehModel", "Area")

# exposure が 100 以上の区分のみを残す
premium_total <- get_total(brvehins, category_to_analyze, "PremTotal", 0)
exposure_total <- get_total(brvehins, category_to_analyze, "ExposTotal", 100)
claim_total <- get_total(brvehins, category_to_analyze, "ClaimTotal")

# exposure 合計が 10 以上の車種のみを残す
claim_total <- claim_total[rowSums(exposure_total, na.rm = TRUE) > 10, ]
premium_total <- premium_total[rowSums(exposure_total, na.rm = TRUE) > 10, ]
exposure_total <- exposure_total[rowSums(exposure_total, na.rm = TRUE) > 10, ]

# write.csv(exposure_total, "exposure_total.csv")
# write.csv(claim_total, "claim_total.csv")

# 純率にするか損害率にするかは要相談
# pure_premium <- premium_total / exposure_total
pure_premium <- claim_total / exposure_total
loss_ratio <- claim_total / premium_total

# write.csv(pure_premium, "pure_premium.csv")
# write.csv(loss_ratio, "loss_ratio.csv")

# -----------------------------------------------------------------------------
# 4. ハイパーパラメータの最適化
# -----------------------------------------------------------------------------
options(warn = -1)
best_params <- optimize_params(
    X = pure_premium,
    n_folds = 4,
    k_values = 2:30,
    lambda_values = c(0.01, 0.1, 1, 10, 20, 30, 50, 100, 1000),
)
print(best_params)

# -----------------------------------------------------------------------------
# 5. 予測精度の検証
# -----------------------------------------------------------------------------
split <- train_test_split(pure_premium)
# # export split$X_train to csv
# write.csv(split$X_train, "data/split$X_train.csv")
# # export split$X_test to csv
# write.csv(split$X_test, "data/split$X_test.csv")

# cmf_args <- list(X = split$X_train, k = best_params$k, lambda = best_params$lambda, niter = 30, nonneg = TRUE, verbose = FALSE)
# cmf_args <- list(X = split$X_train, k = best_params$k, lambda = best_params$lambda, niter = 30, nonneg = FALSE, verbose = FALSE)
cmf_args <- list(X = split$X_train, k = best_params$k, lambda = best_params$lambda, niter = 30, nonneg = TRUE, verbose = FALSE, center = FALSE)

model <- do.call(CMF, cmf_args)

pred <- get_prediction(model, split$X_test)
calc_rmse(pred, split$X_test)

# -----------------------------------------------------------------------------
# 6. 予測結果の可視化
# -----------------------------------------------------------------------------

# (1) テストデータにおける再現性の確認
# テストデータにおける予測値と実測値の散布図
options(repr.plot.width = 5, repr.plot.height = 5)
visualize_scatter_plot(as.vector(split$X_test), as.vector(pred), "Matrix Factorization")

# (2) 欠測値の推定
options(repr.plot.width = 5, repr.plot.height = 5)

# 全データによる予測値作成
all_data <- pure_premium
cmf_args <- list(X = all_data, k = best_params$k, lambda = best_params$lambda, niter = 30, nonneg = TRUE, verbose = FALSE, center = FALSE)
model <- do.call(CMF, cmf_args)

pred <- get_prediction(model, all_data)
calc_rmse(pred, all_data) # training RMSE with all data

visualize_scatter_plot(actual = as.vector(all_data), pred = as.vector(pred), "Matrix Factorization")

options(repr.plot.width = 15, repr.plot.height = 10)

# 全区分に対するヒートマップ（実績）
p1 <- visualize_heatmap(all_data, "actual")

# 全区分に対するヒートマップ（欠測を含む推定値）
actual <- all_data
actual[is.na(actual)] <- 0
estimated <- get_prediction(model, actual)
p2 <- visualize_heatmap(estimated, "pred: Matrix Factorization")

# grid.arrange(p1, p2, ncol = 2)
p1
p2

# -----------------------------------------------------------------------------
# 7. 交互作用なしの GLM との比較
# -----------------------------------------------------------------------------
exposure <- wide_to_long_format(exposure_total, c("VehModel", "Area", "exposure"), na_omit = FALSE)
all_data <- wide_to_long_format(pure_premium, c("VehModel", "Area", "pure_premium"), na_omit = FALSE)
all_data <- merge(all_data, exposure, by = c("VehModel", "Area"))

# glm_args <- list(formula = pure_premium * exposure ~ VehModel + Area + offset(log(exposure)), family = poisson(link = "log"), data = all_data)
glm_args <- list(formula = pure_premium * exposure ~ VehModel + Area + offset(log(exposure)), family = poisson(link = "log"), data = na.omit(all_data))
glm_model <- do.call(glm, glm_args)
# summary(glm_model)

options(repr.plot.width = 5, repr.plot.height = 5)

pred <- predict(glm_model, newdata = na.omit(all_data), type = "response") / na.omit(all_data)$exposure
visualize_scatter_plot(
    actual = as.vector(na.omit(all_data)$pure_premium),
    pred = as.vector(pred),
    model_name = "GLM (no interaction)"
)

options(repr.plot.width = 15, repr.plot.height = 10) # 欠測推定なし

row_names <- row.names(na.omit(all_data))
all_data[, "pred"] <- NA
all_data[row_names, "pred"] <- pred
all_data_pred_wide_format <- get_total(all_data, category_to_analyze, "pred", 0)

# 全区分に対するヒートマップ（実績）
p1 <- visualize_heatmap(pure_premium, "actual")

# 全区分に対するヒートマップ（推定値）：もともと観察がないカテゴリーの推定は行えない
actual <- pure_premium
actual[is.na(actual)] <- 0
estimated <- all_data_pred_wide_format
p2 <- visualize_heatmap(estimated, "pred: GLM (no interaction)")

# grid.arrange(p1, p2, ncol = 2)
p1
p2

# options(repr.plot.width = 25, repr.plot.height = 40) # 欠測推定あり
options(repr.plot.width = 15, repr.plot.height = 10)

exposure <- wide_to_long_format(exposure_total, c("VehModel", "Area", "exposure"), na_omit = FALSE)
all_data <- wide_to_long_format(pure_premium, c("VehModel", "Area", "pure_premium"), na_omit = FALSE)
all_data <- merge(all_data, exposure, by = c("VehModel", "Area"))

valid_veh_types <- unique(glm_model$data$VehModel)
valid_areas <- unique(glm_model$data$Area)

data_with_all_combinations <- expand.grid(VehModel = valid_veh_types, Area = valid_areas)
data_with_all_combinations <- merge(data_with_all_combinations, all_data, by = c("VehModel", "Area"), all.x = TRUE)
data_with_all_combinations[is.na(data_with_all_combinations$exposure), "exposure"] <- 1 # exposure が NA の区分は 1 として予測

pred <- predict(glm_model, newdata = data_with_all_combinations, type = "response") / data_with_all_combinations$exposure
data_with_all_combinations$pred <- pred

all_data <- merge(all_data, data_with_all_combinations, by = c("VehModel", "Area"), all.x = TRUE)
all_data_pred_wide_format <- get_total(all_data, category_to_analyze, "pred", 0)

# 全区分に対するヒートマップ（実績）
p1 <- visualize_heatmap(pure_premium, "actual")

# 全区分に対するヒートマップ（推定値）：もともと観察がないカテゴリーの推定は行えない
actual <- pure_premium
actual[is.na(actual)] <- 0
estimated <- all_data_pred_wide_format
p2 <- visualize_heatmap(estimated, "pred: GLM (no interaction)")

# grid.arrange(p1, p2, ncol = 2)
p1
p2

# -----------------------------------------------------------------------------
# 8. 交互作用を変量効果に入れた GLMM との比較
# -----------------------------------------------------------------------------
exposure <- wide_to_long_format(exposure_total, c("VehModel", "Area", "exposure"), na_omit = FALSE)
all_data <- wide_to_long_format(pure_premium, c("VehModel", "Area", "pure_premium"), na_omit = FALSE)
all_data[, "interaction"] <- interaction(as.character(all_data$VehModel), as.character(all_data$Area))
all_data <- merge(all_data, exposure, by = c("VehModel", "Area"))
# write.csv(all_data, "all_data.csv")

glmm_args <- list(formula = pure_premium * exposure ~ VehModel + Area + (1 | interaction), family = poisson(link = "log"), data = all_data, offset = log(all_data$exposure))
glmm_model <- do.call(glmer, glmm_args)

options(repr.plot.width = 5, repr.plot.height = 5)

# show scatter plot for training data for sanity check
pred <- predict(glmm_model, newdata = na.omit(all_data), type = "response")
visualize_scatter_plot(
  actual = as.vector(na.omit(all_data)$pure_premium),
  pred = as.vector(pred),
  model_name = "GLMM"
)

# options(repr.plot.width = 25, repr.plot.height = 40) # 欠測推定なし
options(repr.plot.width = 15, repr.plot.height = 10)

row_names <- row.names(na.omit(all_data))
all_data[, "pred"] <- NA
all_data[row_names, "pred"] <- pred
all_data_pred_wide_format <- get_total(all_data, category_to_analyze, "pred", 0)

# 全区分に対するヒートマップ（実績）
p1 <- visualize_heatmap(pure_premium, "actual")

# 全区分に対するヒートマップ（推定値）：もともと観察がないカテゴリーの推定は行えない
actual <- pure_premium
actual[is.na(actual)] <- 0
estimated <- all_data_pred_wide_format
p2 <- visualize_heatmap(estimated, "pred: GLMM (with interaction)")

# grid.arrange(p1, p2, ncol = 2)
p1
p2

# options(repr.plot.width = 25, repr.plot.height = 40) # 欠測推定あり
options(repr.plot.width = 15, repr.plot.height = 10)

exposure <- wide_to_long_format(exposure_total, c("VehModel", "Area", "exposure"), na_omit = FALSE)
all_data <- wide_to_long_format(pure_premium, c("VehModel", "Area", "pure_premium"), na_omit = FALSE)
all_data[, "interaction"] <- interaction(as.character(all_data$VehModel), as.character(all_data$Area))
all_data <- merge(all_data, exposure, by = c("VehModel", "Area"))

data_with_all_combinations <- expand.grid(VehModel = valid_veh_types, Area = valid_areas)
data_with_all_combinations <- merge(data_with_all_combinations, all_data, by = c("VehModel", "Area"), all.x = TRUE)
data_with_all_combinations[is.na(data_with_all_combinations$exposure), "exposure"] <- 1 # exposure が NA の区分は 1 として予測
data_with_all_combinations[is.na(data_with_all_combinations$exposure), "interaction"] <- interaction(as.character(data_with_all_combinations[is.na(data_with_all_combinations$exposure), "VehModel"], as.character(data_with_all_combinations[is.na(data_with_all_combinations$exposure), "Area"])))

pred <- predict(glmm_model, newdata = data_with_all_combinations, type = "response", allow.new.levels = TRUE) / data_with_all_combinations$exposure
data_with_all_combinations$pred <- pred

all_data <- merge(all_data, data_with_all_combinations, by = c("VehModel", "Area"), all.x = TRUE)
all_data_pred_wide_format <- get_total(all_data, category_to_analyze, "pred", 0)

# 全区分に対するヒートマップ（実績）
p1 <- visualize_heatmap(pure_premium, "actual")

# 全区分に対するヒートマップ（推定値）
actual <- pure_premium
actual[is.na(actual)] <- 0
estimated <- all_data_pred_wide_format
p2 <- visualize_heatmap(estimated, "pred: GLMM (with interaction)")

# grid.arrange(p1, p2, ncol = 2)
p1
p2
