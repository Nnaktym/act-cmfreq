
library(CASdatasets)
library(cmfrec)
library(ggplot2)

data(brvehins1a)
data(brvehins1b)
data(brvehins1c)
data(brvehins1d)
data(brvehins1e)

brvehins=rbind(brvehins1a,brvehins1b,brvehins1c,brvehins1d,brvehins1e)

set.seed(1)
brvehins$cvgroup = sample(1:5,nrow(brvehins),T)

premium_total = tapply(brvehins$PremTotal, list(brvehins$VehModel,brvehins$Area,brvehins$cvgroup), sum) # �^�E�n��ECV�O���[�v���̕ی����̏W�v
claim_total = tapply(rowSums(brvehins[,19:23]), list(brvehins$VehModel,brvehins$Area,brvehins$cvgroup), sum) # �^�E�n��ECV�O���[�v���̃N���[�����z�̏W�v
print(dim(premium_total))
print(dim(claim_total))


# �ی����v10000�ȏ�A�N���[�����z�v5000�ȏ�̌^�����c��
premium_total_10000 = premium_total[apply(premium_total,1,sum,na.rm=T)>=10000 & apply(claim_total,1,sum,na.rm=T)>=5000,,]
claim_total_10000 = claim_total[apply(premium_total,1,sum,na.rm=T)>=10000 & apply(claim_total,1,sum,na.rm=T)>=5000,,]

# �N���[���Ȃ��E�ی����Ȃ���0�ɕϊ�����B
premium_total_10000[is.na(premium_total_10000)] = 0 # added
claim_total_10000[is.na(claim_total_10000)] = 0 # added


cverror_nonneg = matrix(0,30,5)
cverror = matrix(0,30,5)

for(j in 1:nrow(cverror))for(i in 1:5){
# �P���f�[�^�A���؃f�[�^�̍쐬
premium_total_cvtrain = apply(premium_total_10000[,,-i],1:2,sum)
premium_total_cvtest = premium_total_10000[,,i]
claim_total_cvtrain = apply(claim_total_10000[,,-i],1:2,sum)
claim_total_cvtest = claim_total_10000[,,i]
loss_ratio_cvtrain = claim_total_cvtrain / premium_total_cvtrain # �^�E�n�斈�̑��Q��
loss_ratio_cvtrain[premium_total_cvtrain==0] = NA 
loss_ratio_cvtest = claim_total_cvtest / premium_total_cvtest # �^�E�n�斈�̑��Q��
loss_ratio_cvtest[premium_total_cvtest==0] = NA 

# �񕉐���Ȃ�
cmf = CMF( # <----- �����̏�����ς��ĐF�X������
  X=loss_ratio_cvtrain,
  k=1,
  verbose=F,
  nonneg=FALSE, # added
  weight=premium_total_cvtrain,
  lambda=2^j, # lambda=2^j�ɃZ�b�g
)

A = cmf$matrices$A
B = cmf$matrices$B
type_bias = cmf$matrices$user_bias
region_bias = cmf$matrices$item_bias
mu = cmf$matrices$glob_mean
pred_loss_ratio = mu + type_bias%o%rep(1, ncol(loss_ratio_cvtest)) + rep(1,nrow(loss_ratio_cvtest))%o%region_bias + t(A)%*%B

cverror[j,i] = sum((loss_ratio_cvtest-pred_loss_ratio)^2*premium_total_cvtest,na.rm=T)

# �񕉐��񂠂�
cmf = CMF( # <----- �����̏�����ς��ĐF�X������
  X=loss_ratio_cvtrain,
  k=1,
  verbose=F,
  nonneg=TRUE, # added
  weight=premium_total_cvtrain,
  lambda=2^j, # lambda=2^j�ɃZ�b�g
)

A = cmf$matrices$A
B = cmf$matrices$B
type_bias = cmf$matrices$user_bias
region_bias = cmf$matrices$item_bias
mu = cmf$matrices$glob_mean
pred_loss_ratio = mu + type_bias%o%rep(1, ncol(loss_ratio_cvtest)) + rep(1,nrow(loss_ratio_cvtest))%o%region_bias + t(A)%*%B

cverror_nonneg[j,i] = sum((loss_ratio_cvtest-pred_loss_ratio)^2*premium_total_cvtest,na.rm=T)
}

rowMeans(cverror_nonneg)
rowMeans(cverror)

# �񕉐��񂠂�̂Ƃ�j=21�ilambda=2^21�j�A�񕉐���Ȃ��̂Ƃ�j=22�ilambda=2^22�j�Ō덷���ŏ��ƂȂ���
#> rowMeans(cverror_nonneg)
# [1] 26611302875 25690432496 24676949187 23698338383 22794696706 21975048736 21234278348 20965259439 19344951096 11156285961
#[11] 10807467697  9547257745 57058900645 53005061264 36104437769  4173563555  4150854673  4145463741  4138636087  4130358726
#[21]  4124528173  4124810563  4129533301  4135493983  4148434935  4173910093  4217304219  4280592310  4360093653  4447061453
#> rowMeans(cverror)
# [1] 38265691961 37346602222 36114601298 34681008863 33227662451 31920888262 30866570788 29670945783 28063670530 25670527347
#[11] 22149255942 57229415837 66625141471 62858872367 41224687221  4232338816  4167867950  4148554278  4137780231  4130153855
#[21]  4124213098  4123354592  4128334989  4134756766  4148136272  4173832590  4217293632  4280592587  4360094196  4447061665


# A��B�ɑ΂���lambda�����ɑ�ɂ���0�Ő��肳�����ꍇ�ii.e.A=B=0�ɂ��čs�񕪉��Ƃ������minimum-bias�@�ɂ����ꍇ�j
cverror_nointeract = numeric(5)
for(i in 1:5){
# �P���f�[�^�A���؃f�[�^�̍쐬
premium_total_cvtrain = apply(premium_total_10000[,,-i],1:2,sum)
premium_total_cvtest = premium_total_10000[,,i]
claim_total_cvtrain = apply(claim_total_10000[,,-i],1:2,sum)
claim_total_cvtest = claim_total_10000[,,i]
loss_ratio_cvtrain = claim_total_cvtrain / premium_total_cvtrain # �^�E�n�斈�̑��Q��
loss_ratio_cvtrain[premium_total_cvtrain==0] = NA 
loss_ratio_cvtest = claim_total_cvtest / premium_total_cvtest # �^�E�n�斈�̑��Q��
loss_ratio_cvtest[premium_total_cvtest==0] = NA 

# A��B�ɑ΂���lambda�����ɑ�ɂ���A=B=0�i���ݍ�p�Ȃ��j�Ő��肳�����ꍇ
cmf = CMF( # <----- �����̏�����ς��ĐF�X������
  X=loss_ratio_cvtrain,
  k=1,
  verbose=F,
  nonneg=TRUE, # added
  weight=premium_total_cvtrain,
  lambda=c(0,0,1e20,1e20,0,0), # ����'user_bias', 'item_bias', 'A', 'B', 'C', 'D'�ɑ΂���lambda�i'C', 'D'�͖{���͂Ŏg�p����Ȃ��j
)
A = cmf$matrices$A
B = cmf$matrices$B
type_bias = cmf$matrices$user_bias
region_bias = cmf$matrices$item_bias
mu = cmf$matrices$glob_mean
pred_loss_ratio = mu + type_bias%o%rep(1, ncol(loss_ratio_cvtest)) + rep(1,nrow(loss_ratio_cvtest))%o%region_bias + t(A)%*%B

cverror_nointeract[i] = sum((loss_ratio_cvtest-pred_loss_ratio)^2*premium_total_cvtest,na.rm=T)
}

mean(cverror_nointeract)
#> mean(cverror_nointeract)
#[1] 4332424417 # ���ݍ�p����̕����������،덷���������Ȃ邱�Ƃ��m�F�ł���


# �������،덷���ŏ��ƂȂ����񕉐��񂠂�j=21�̂Ƃ��̑S�f�[�^�ɑ΂��镪��
j=21

# ���Q���̒�`
loss_ratio = apply(claim_total_10000,1:2,sum) / apply(premium_total_10000,1:2,sum) # �^�E�n�斈�̑��Q��
loss_ratio[apply(premium_total_10000,1:2,sum)==0] = NA # added�iloss_ratio��Inf�������cmf$matrices$glob_mean��NA�ɂȂ�œK�������܂������Ă��Ȃ��������߁j

cmf = CMF( # <----- �����̏�����ς��ĐF�X������
  X=loss_ratio,
  k=1,
  verbose=F,
  nonneg=T, # added
  weight=apply(premium_total_10000,1:2,sum),
  lambda=2^j,
)

A = cmf$matrices$A
B = cmf$matrices$B
type_bias = cmf$matrices$user_bias
region_bias = cmf$matrices$item_bias
mu = cmf$matrices$glob_mean
pred_loss_ratio = mu + type_bias%o%rep(1, ncol(loss_ratio)) + rep(1,nrow(loss_ratio))%o%region_bias + t(A)%*%B

range(pred_loss_ratio)

#> range(pred_loss_ratio)
#[1] 0.2000173 2.2347726

mean(t(A)%*%B)
sd(t(A)%*%B)
fivenum(t(A)%*%B)

# ���ݍ�p���̉e���̑傫�����`�F�b�N
#> mean(t(A)%*%B)
#[1] 0.003549444
#> sd(t(A)%*%B)
#[1] 0.02802605
#> fivenum(t(A)%*%B)
#[1] 0.000000 0.000000 0.000000 0.000000 1.563961



# �q�[�g�}�b�v
par(mfrow=c(1, 2))
image(pred_loss_ratio, xlab="type", ylab="region", breaks=0:12/10, main="predicted")
image(loss_ratio, xlab="type",ylab="region",breaks=0:12/10, main="actual")


# Area���̌���
result_area = cbind(table(brvehins$Area),region_bias,c(B))
colnames(result_area)[1] = "#policy"
colnames(result_area)[3] = "B"
result_area

# �_�񐔂̑���Met. de Sao Paulo�ARibeirao Preto e Demais Mun. de Campinas�Ō��ݍ�p���傫��
> result_area
                                           #policy region_bias           B
Acre                                          6822   0.4573011 0.023844793
Alagoas                                      24859   0.4454659 0.018881381
Amapa                                         5318   0.3916090 0.020989126
Amazonas                                     20015   0.3872655 0.023269196
Bahia                                        58159   0.4995535 0.000000000
Blumenau e demais regioes                    76611   0.6173985 0.026629514
Brasilia                                     56017   0.5248110 0.000000000
Ceara                                        39874   0.4459910 0.000000000
Demais regioes                              147802   0.5926821 0.000000000
Espirito Santo                               45105   0.5738535 0.039921397
F.Iguatu-Medianeira-Cascavel-Toledo          43076   0.8641696 0.199087367
Goias                                        54890   0.5648363 0.000000000
Grande Campinas                              77376   0.5358403 0.543208711
Interior                                     51735   0.5742729 0.000000000
Litoral Norte e Baixada Santista             32208   0.5625041 0.005407455
Maranhao                                     23209   0.4055569 0.000000000
Mato Grosso                                  35536   0.5246150 0.000000000
Mato Grosso do Sul                           35005   0.4506986 0.000000000
Met. Curitiba                                88414   0.5798730 0.000000000
Met. de Sao Paulo                           155942   0.4411520 2.736841422
Met. do Rio de Janeiro                       94120   0.4811597 0.242635949
Met. Florianopolis e Sul                     58033   0.5386389 0.000000000
Met. Porto Alegre e Caxias do Sul            89100   0.5608247 0.000000000
Met.BH-Centro Oeste-Zona Mata-C. Vertentes   82911   0.4875648 0.036226536
Oeste                                        42944   0.6160853 0.000000000
Para                                         28497   0.4565976 0.000000000
Paraiba                                      24725   0.4407620 0.000000000
Pernambuco                                   46187   0.4655513 0.000000000
Piaui                                        17848   0.4467668 0.000000000
Ribeirao Preto e Demais Mun. de Campinas    126142   0.5367363 0.942632225
Rio Grande do Norte                          27789   0.4400326 0.000000000
Rondonia                                      4349   0.3206692 0.032091907
Roraima                                       4408   0.2000084 0.030072697
Sergipe                                      21825   0.4694641 0.000000000
Sudeste de Goias                             15441   0.5439597 0.012219174
Sul                                          43634   0.5104504 0.026976945
Tocantins                                    16698   0.4961504 0.000000000
Triangulo mineiro                            51291   0.5394972 0.000000000
Vale do Aco-Norte-Vale Jequitinhonha         25322   0.4845693 0.000000000
Vale do Paraiba e Ribeira                    66105   0.5301170 0.330187210
