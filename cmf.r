
library(CASdatasets)
library(cmfrec)
library(ggplot2)

data(brvehins1a)
data(brvehins1b)
data(brvehins1c)
data(brvehins1d)
data(brvehins1e)

brvehins=rbind(brvehins1a,brvehins1b,brvehins1c,brvehins1d,brvehins1e)

pt = tapply(brvehins$PremTotal,list(brvehins$VehModel,brvehins$Area),sum) # 型・地域毎の保険料の集計
ct = tapply(rowSums(brvehins[,19:23]),list(brvehins$VehModel,brvehins$Area),sum) # 型・地域毎のクレーム総額の集計

pt10000 = pt[rowSums(pt,na.rm=T)>=10000 & rowSums(ct,na.rm=T)>=5000,] # 保険料計10000以上、クレーム総額計5000以上の型だけ残す
ct10000 = ct[rowSums(pt,na.rm=T)>=10000 & rowSums(ct,na.rm=T)>=5000,] # 保険料計10000以上、クレーム総額計5000以上の型だけ残す

lr = ct10000/pt10000 # 型・地域毎の損害率

image(lr,xlab="型",ylab="地域",breaks=0:12/10) # 型・地域毎の損害率のヒートマップ


wt = ifelse(is.na(pt10000),0,pt10000) # 分析に用いる重み
cmf=CMF(lr,k=1,weight=wt,lambda=1000000)

A=cmf$matrices$A
B=cmf$matrices$B
ub=cmf$matrices$user_bias
ib=cmf$matrices$item_bias
mu=cmf$matrices$glob_mean

lrpred = mu + ub%o%rep(1,ncol(lr)) + rep(1,nrow(lr))%o%ib + t(A)%*%B # 型・地域毎の損害率の予測値

image(lrpred,xlab="型",ylab="地域",breaks=0:12/10) # 型・地域毎の損害率の予測値のヒートマップ

areas = data.frame(area = colnames(lr), ib = ib, B = c(B), lr = colSums(ct,na.rm=T)/colSums(pt,na.rm=T))

ggplot(areas,aes(x=area,y=ib,col=area)) + geom_point() # 地域毎の損害率（主効果）を図示

ggplot(areas,aes(x=area,y=B,col=area)) + geom_point() # 地域毎の損害率（交互作用部分）を図示

ggplot(areas,aes(x=area,y=lr,col=area)) + geom_point() # 地域毎の損害率実績を図示


# To Do
# ・交差検証でk(特異ベクトルの使用数)やlambda(L2正則化パラメータ)を調整し、予測精度を検証
# ・一部の予測値がマイナス（損害率なのに）となってしまうので、場合によっては対数変換も検討

