# original

library(cmfrec)
library(ggplot2)

# library(CASdatasets)
# cas_dataset_path <- "/Users/nanakato/projects/CASdatasets/data"
#
# load(paste(cas_dataset_path, "brvehins1a.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1b.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1c.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1d.rda", sep="/"))
# load(paste(cas_dataset_path, "brvehins1e.rda", sep="/"))
#
# brvehins <- rbind(brvehins1a, brvehins1b, brvehins1c, brvehins1d, brvehins1e)
# write.csv(brvehins, file="brvehins.csv")
brvehins <- read.csv("data/brvehins.csv")

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

pt <- tapply(brvehins$PremTotal, list(brvehins$VehModel,brvehins$Area), sum) # 型・地域毎の保険料の集計
ct <- tapply(rowSums(brvehins[,19:23]), list(brvehins$VehModel,brvehins$Area), sum) # 型・地域毎のクレーム総額の集計

# > dim(pt)
# [1] 4259   40
# > dim(ct)
# [1] 4259   40

# memo:
# there are categories that has missing values

# 料率区分が細かいとき、欠測が出たりふつうにあるし実績が安定しない。交互作用なんてみてられない。

pt10000 <- pt[rowSums(pt,na.rm=T)>=10000 & rowSums(ct,na.rm=T)>=5000,] # 保険料計10000以上、クレーム総額計5000以上の型だけ残す
ct10000 <- ct[rowSums(pt,na.rm=T)>=10000 & rowSums(ct,na.rm=T)>=5000,] # 保険料計10000以上、クレーム総額計5000以上の型だけ残す

lr <- ct10000/pt10000 # 型・地域毎の損害率

image(lr,xlab="�^",ylab="�n��",breaks=0:12/10) # 型・地域毎の損害率のヒートマップ

wt = ifelse(is.na(pt10000),0,pt10000) # ���͂ɗp����d��
cmf=CMF(lr,k=1,weight=wt,lambda=1000000)

A=cmf$matrices$A
B=cmf$matrices$B
ub=cmf$matrices$user_bias
ib=cmf$matrices$item_bias
mu=cmf$matrices$glob_mean

lrpred = mu + ub%o%rep(1,ncol(lr)) + rep(1,nrow(lr))%o%ib + t(A)%*%B # �^�E�n�斈�̑��Q���̗\���l

image(lrpred,xlab="�^",ylab="�n��",breaks=0:12/10) # �^�E�n�斈�̑��Q���̗\���l�̃q�[�g�}�b�v

areas = data.frame(area = colnames(lr), ib = ib, B = c(B), lr = colSums(ct,na.rm=T)/colSums(pt,na.rm=T))

ggplot(areas,aes(x=area,y=ib,col=area)) + geom_point() # �n�斈�̑��Q���i����ʁj��}��

ggplot(areas,aes(x=area,y=B,col=area)) + geom_point() # �n�斈�̑��Q���i���ݍ�p�����j��}��

ggplot(areas,aes(x=area,y=lr,col=area)) + geom_point() # �n�斈�̑��Q�����т�}��


# To Do
# �E�������؂�k(���كx�N�g���̎g�p��)��lambda(L2�������p�����[�^)�𒲐����A�\�����x������
# �E�ꕔ�̗\���l���}�C�i�X�i���Q���Ȃ̂Ɂj�ƂȂ��Ă��܂��̂ŁA�ꍇ�ɂ���Ă͑ΐ��ϊ�������



