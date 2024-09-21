# original

library(cmfrec)
library(ggplot2)

# library(CASdatasets)
# cas_dataset_path = "/Users/nanakato/projects/CASdatasets/data"
#
# load(paste(cas_dataset_path, "brvehins1a.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1b.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1c.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1d.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1e.rda", sep="/"))
#
# brvehins = rbind(brvehins1a, brvehins1b, brvehins1c, brvehins1d, brvehins1e)
# write.csv(brvehins, file="brvehins.csv")
brvehins = read.csv("data/brvehins.csv")

# > str(brvehins)
# 'data.frame':	1965355 obs. of  24 variables:
#  $ X                  : int  565192 1549299 803785 1735441 1848349 89535 1037912 1753914 1083762 897406 ...
#  $ Gender             : chr  "Female" "Female" "Female" "Male" ...
#  $ DrivAge            : chr  ">55" "36-45" "18-25" ">55" ...
#  $ VehYear            : int  1997 2010 2008 2004 2009 1998 1999 2006 2010 2002 ...
#  $ VehModel           : chr  "Gm - Chevrolet - Kadett Gl 2.0 Mpfi / Efi" "Gm - Chevrolet - Montana 1.4 8v Conquest Econoflex  2p" "Vw - Volkswagen - Fox City 1.0mi/ 1.0mi Total Flex 8v 3p" "Harley-davidson - Fat Boy" ...
#  $ VehGroup           : chr  "Gm Chevrolet Kadett" "Gm Chevrolet Montana" "Vw Volkswagen Fox 1.0" "Harley-davidson Motos - Todas" ...
#  $ Area               : chr  "Interior" "Maranhao" "Mato Grosso do Sul" "Met. Porto Alegre e Caxias do Sul" ...
#  $ State              : chr  "Rio de Janeiro" "Maranhao" "Mato Grosso do Sul" "Rio Grande do Sul" ...
#  $ StateAb            : chr  "RJ" "MA" "MS" "RS" ...
#  $ ExposTotal         : num  1.01 3 1.01 1.45 4.55 1 1.01 1.51 0.25 0.93 ...
#  $ ExposFireRob       : int  0 0 0 0 0 0 0 0 0 0 ...
#  $ PremTotal          : num  743 5026 916 1602 53031 ...
#  $ PremFireRob        : int  0 0 0 0 0 0 0 0 0 0 ...
#  $ SumInsAvg          : num  10853 31010 24977 35933 301890 ...
#  $ ClaimNbRob         : int  0 0 0 0 0 0 0 0 0 0 ...
#  $ ClaimNbPartColl    : int  0 0 0 0 1 1 0 0 0 0 ...
#  $ ClaimNbTotColl     : int  0 0 0 0 0 0 0 0 0 0 ...
#  $ ClaimNbFire        : int  0 0 0 0 0 0 0 0 0 0 ...
#  $ ClaimNbOther       : int  0 0 0 0 0 2 0 0 0 0 ...
#  $ ClaimAmountRob     : num  0 0 0 0 0 0 0 0 0 0 ...
#  $ ClaimAmountPartColl: int  0 0 0 0 648 2773 0 0 0 0 ...
#  $ ClaimAmountTotColl : num  0 0 0 0 0 0 0 0 0 0 ...
#  $ ClaimAmountFire    : int  0 0 0 0 0 0 0 0 0 0 ...
#  $ ClaimAmountOther   : int  0 0 0 0 0 120 0 0 0 0 ...

premium_total = tapply(brvehins$PremTotal, list(brvehins$VehModel,brvehins$Area), sum) # 型・地域毎の保険料の集計
claim_total = tapply(rowSums(brvehins[,19:23]), list(brvehins$VehModel,brvehins$Area), sum) # 型・地域毎のクレーム総額の集計



# > dim(premium_total)
# [1] 4259   40
# > dim(claim_total)
# [1] 4259   40

# memo:
# there are categories that has missing values

# 料率区分が細かいとき、欠測が出たりふつうにあるし実績が安定しない。交互作用なんてみてられない。

# 保険料計10000以上、クレーム総額計5000以上の型だけ残す
premium_total_10000 = premium_total[rowSums(premium_total,na.rm=T)>=10000 & rowSums(claim_total,na.rm=T)>=5000,]
claim_total_10000 = claim_total[rowSums(premium_total,na.rm=T)>=10000 & rowSums(claim_total,na.rm=T)>=5000,]

premium_total_10000[is.na(premium_total_10000)] = 0 # added
claim_total_10000[is.na(claim_total_10000)] = 0 # added

loss_ratio = claim_total_10000 / premium_total_10000 # 型・地域毎の損害率
loss_ratio[premium_total_10000==0] = NA # added

# write.csv(loss_ratio, file="data/loss_ratio.csv")
# > dim(lr)
# [1] 2517   40


# premium = ifelse(is.na(premium_total_10000),0, premium_total_10000) # weight
# > wt = ifelse(is.na(pt10000),0,pt10000) # weight
# > dim(wt)
# [1] 2517   40

# cmf = CMF_implicit(
#   X=loss_ratio,
#   # k=1,
#   # weight=premium,
#   # lambda=1000000,
# )

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

# 型・地域毎の損害率の予測値のヒートマップ
image(pred_loss_ratio, xlab="type", ylab="region", breaks=0:12/10)
image(loss_ratio, xlab="type",ylab="region",breaks=0:12/10) # 型・地域毎の損害率のヒートマップ

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



