# original



library(CASdatasets)
library(cmfrec)
library(ggplot2)



data(brvehins1a)
data(brvehins1b)
data(brvehins1c)
data(brvehins1d)
data(brvehins1e)

brvehins=rbind(brvehins1a,brvehins1b,brvehins1c,brvehins1d,brvehins1e)

pt = tapply(brvehins$PremTotal,list(brvehins$VehModel,brvehins$Area),sum) # �^�E�n�斈�̕ی����̏W�v
ct = tapply(rowSums(brvehins[,19:23]),list(brvehins$VehModel,brvehins$Area),sum) # �^�E�n�斈�̃N���[�����z�̏W�v

pt10000 = pt[rowSums(pt,na.rm=T)>=10000 & rowSums(ct,na.rm=T)>=5000,] # �ی����v10000�ȏ�A�N���[�����z�v5000�ȏ�̌^�����c��
ct10000 = ct[rowSums(pt,na.rm=T)>=10000 & rowSums(ct,na.rm=T)>=5000,] # �ی����v10000�ȏ�A�N���[�����z�v5000�ȏ�̌^�����c��

lr = ct10000/pt10000 # �^�E�n�斈�̑��Q��

image(lr,xlab="�^",ylab="�n��",breaks=0:12/10) # �^�E�n�斈�̑��Q���̃q�[�g�}�b�v


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

