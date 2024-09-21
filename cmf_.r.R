
library(cmfrec)
library(ggplot2)

# library(CASdatasets) ---> 環境の問題？かインストールに失敗したため、ローカルからデータを読み込んだ
# data(brvehins1a)
# data(brvehins1b)
# data(brvehins1c)
# data(brvehins1d)
# data(brvehins1e)
## ! cd /Users/nanakato/projects
## ! git clone https://github.com/dutangc/CASdatasets.git
cas_dataset_path = "/Users/nanakato/projects/CASdatasets/data"
load(paste(cas_dataset_path, "brvehins1a.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1b.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1c.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1d.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1e.rda", sep="/"))
brvehins = rbind(brvehins1a, brvehins1b, brvehins1c, brvehins1d, brvehins1e)

cas_dataset_path = "/Users/nanakato/projects/CASdatasets/data"
load(paste(cas_dataset_path, "brvehins1a.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1b.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1c.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1d.rda", sep="/"))
load(paste(cas_dataset_path, "brvehins1e.rda", sep="/"))
brvehins = rbind(brvehins1a, brvehins1b, brvehins1c, brvehins1d, brvehins1e)



premium_total = tapply(brvehins$PremTotal, list(brvehins$VehModel,brvehins$Area), sum) # 型・地域毎の保険料の集計
claim_total = tapply(rowSums(brvehins[,19:23]), list(brvehins$VehModel,brvehins$Area), sum) # 型・地域毎のクレーム総額の集計
# > dim(premium_total)
# [1] 4259   40
# > dim(claim_total)
# [1] 4259   40

# 保険料計10000以上、クレーム総額計5000以上の型だけ残す
premium_total_10000 = premium_total[rowSums(premium_total,na.rm=T)>=10000 & rowSums(claim_total,na.rm=T)>=5000,]
claim_total_10000 = claim_total[rowSums(premium_total,na.rm=T)>=10000 & rowSums(claim_total,na.rm=T)>=5000,]

premium_total_10000[is.na(premium_total_10000)] = 0 # added
claim_total_10000[is.na(claim_total_10000)] = 0 # added

loss_ratio = claim_total_10000 / premium_total_10000 # 型・地域毎の損害率
loss_ratio[premium_total_10000==0] = NA # added

cmf = CMF(
  X=loss_ratio,
  k=1,
  weight=premium_total_10000,
  lambda=1000000,
)

A = cmf$matrices$A
B = cmf$matrices$B
type_bias = cmf$matrices$user_bias
region_bias = cmf$matrices$item_bias
mu = cmf$matrices$glob_mean

# 型・地域毎の損害率の予測値
pred_loss_ratio = mu + type_bias%o%rep(1, ncol(loss_ratio)) + rep(1,nrow(loss_ratio))%o%region_bias + t(A)%*%B

# 型・地域毎の損害率のヒートマップ比較
par(mfrow=c(1,2))
image(pred_loss_ratio, xlab="type", ylab="region", breaks=0:12/10) # 予測値
image(loss_ratio, xlab="type",ylab="region",breaks=0:12/10) # 実績値

# 型・地域毎の損害率の散布図比較
compare = data.frame(
  pred = as.vector(pred_loss_ratio),
  actual = as.vector(loss_ratio)
)
ggplot(compare,aes(x=pred,y=actual)) + geom_point() # 予測値と実績値の散布図

areas = data.frame(
  area = colnames(loss_ratio),
  region_bias = region_bias,
  B = c(B),
  lr = colSums(claim_total,na.rm=T)/colSums(premium_total,na.rm=T),
)

ggplot(areas,aes(x=area,y=region_bias,col=area)) + geom_point() # 地域毎の損害率（主効果）を図示
ggplot(areas,aes(x=area,y=B,col=area)) + geom_point() # 地域毎の損害率（交互作用部分）を図示
ggplot(areas,aes(x=area,y=lr,col=area)) + geom_point() #地域毎の損害率実績を図示


# To Do
# ・交差検証でk(特異ベクトルの使用数)やlambda(L2正則化パラメータ)を調整し、予測精度を検証
# ・一部の予測値がマイナス（損害率なのに）となってしまうので、場合によっては対数変換も検討



