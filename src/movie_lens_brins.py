import pickle
import pandas as pd
import numpy as np
from cmfrec import CMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# preprocess movie lens data
# https://nbviewer.org/github/david-cortes/cmfrec/blob/master/example/load_data.ipynb

# tutorial notebook (movie lens dataset):
# https://nbviewer.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb

def plot_actual_vs_predicted(test, pred, fig_name):
    plt.scatter(test["Rating"], pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(f"{fig_name}.png")
    plt.close()
    return None


# brazilian dataset

population_density_path = "/Users/spectee/research/act-cmfreq/data/brazil_population.csv"
density = pd.read_csv(population_density_path)
density["area_id"] = density.index

brazil_data_path = "/Users/spectee/research/act-cmfreq/data/brvehins.csv"
df = pd.read_csv(brazil_data_path)
df = df[df['Gender'] != 'Corporate']
df.columns
# ['Gender', 'DrivAge', 'VehYear', 'VehModel', 'VehGroup', 'Area', 'State',
#        'StateAb', 'ExposTotal', 'ExposFireRob', 'PremTotal', 'PremFireRob',
#        'SumInsAvg', 'ClaimNbRob', 'ClaimNbPartColl', 'ClaimNbTotColl',
#        'ClaimNbFire', 'ClaimNbOther', 'ClaimAmountRob', 'ClaimAmountPartColl',
#        'ClaimAmountTotColl', 'ClaimAmountFire', 'ClaimAmountOther',
#        'ClaimTotal']

df = df.merge(density, on='Area', how='left')
df["pure_premium"] = np.where(df["ExposTotal"] > 0, df["ClaimTotal"] / df["ExposTotal"], np.nan)

df = df.reset_index(drop=True)
df["pol_id"] = df.index

cols_to_use = ["pol_id", 'area_id', 'pure_premium']
policy_side_info_cols = ["pol_id", 'Gender', 'DrivAge'] # 'VehModel'
area_side_info_cols = ['area_id', 'density_km2']
ratings_cols = ["pure_premium"]

ratings = df[cols_to_use]
len(ratings["pol_id"].unique())
# 69217
len(ratings["area_id"].unique())
# 40
len(ratings["pure_premium"].unique())
# 13974
policy_side_info = df[policy_side_info_cols]
area_side_info = df[area_side_info_cols]
area_id_to_name = {i: j for i, j in zip(density["area_id"], density["Area"])}

ratings = ratings.rename(columns={"pol_id": "UserId", "area_id": "ItemId", "pure_premium": "Rating"})
policy_side_info = policy_side_info.rename(columns={"pol_id": "UserId"})
area_side_info = area_side_info.rename(columns={"area_id": "ItemId"})

# In [30]: ratings
#        pol_id  area_id  pure_premium
# 0           0       12      0.000000
# 1           1       18      0.000000
# 2           2       29      0.000000

# In [32]: policy_side_info
#        pol_id  Gender DrivAge
# 0           0  Female   36-45
# 1           1    Male   36-45
# 2           2  Female   26-35
# 3           3    Male     >55
# 4           4    Male   46-55

# In [33]: area_side_info
#        area_id density_km2
# 0           12       737.7
# 1           18       308.3
# 2           29       104.4
# 3           29       104.4
# 4            3         2.7



# train test split ratings


train, test = train_test_split(ratings.dropna(), test_size=0.2, random_state=0)
# train, test = train_test_split(ratings, test_size=0.2, random_state=0)



# train 53788
# test 13447
# In [47]: len(ratings.dropna())
# Out[47]: 67235
# In [48]: len(ratings)
# Out[48]: 69217

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# 1. classical model ------------------------------------------------------
# 従来の行列分解
# 従来の行列分解では、ユーザーとアイテムの相互作用を表す行列を、ユーザー因子行列とアイテム因子行列に分解することで、ユーザーの嗜好とアイテムの特徴を表現します。

# X ≈ AB^T + μ + b_A + b_B
# X: ユーザーとアイテムの相互作用行列 (評価値など)
# A: ユーザー因子行列
# B: アイテム因子行列
# μ: 全体の平均値
# b_A: ユーザーバイアス
# b_B: アイテムバイアス

# model_no_sideinfo = CMF(method="als", k=40, lambda_=1e+1)
model_no_sideinfo = CMF(
    k=40,
    lambda_=10.0,
    method='als',
    use_cg=True,
    user_bias=True,
    item_bias=True,
    center=False, #True,
    add_implicit_features=False,
    scale_lam=False,
    scale_lam_sideinfo=False,
    scale_bias_const=False,
    k_user=0,
    k_item=0,
    k_main=0,
    w_main=1.0,
    w_user=1.0,
    w_item=1.0,
    w_implicit=0.5,
    l1_lambda=0.0,
    center_U=True,
    center_I=True,
    maxiter=800,
    niter=10,
    parallelize='separate',
    corr_pairs=4,
    max_cg_steps=3,
    precondition_cg=False,
    finalize_chol=True,
    NA_as_zero=False,
    NA_as_zero_user=False,
    NA_as_zero_item=False,
    nonneg=True, # false
    nonneg_C=False,
    nonneg_D=False,
    max_cd_steps=100,
    precompute_for_predictions=True,
    include_all_X=True,
    use_float=True,
    random_state=1,
    verbose=False,
    print_every=10,
    handle_interrupt=True,
    produce_dicts=False,
    nthreads=-1,
    n_jobs=None,
)
model_no_sideinfo.fit(train)

pred = model_no_sideinfo.predict(user=test["UserId"], item=test["ItemId"])
rmse = np.sqrt(mean_squared_error(pred, test["Rating"]))

plot_actual_vs_predicted(test, pred, "classical_model")

# --> パーソナライズドされたレーティングには用いることができそうだが、保険料の場合、属性の等しいユーザーには同じ保険料を適用したいため、この使い方は実務には適していない
# なお、結果が悪いのでハズレ値除去など試す必要がある。

# 2. Collective Matrix Factorization (CMF)  -------------------------------
# https://www.researchgate.net/publication/354733006_Improving_cold-start_recommendations_using_item-based_stereotypes
# これは、Collective Matrix Factorization (CMF) と呼ばれるレコメンデーションシステムの手法について説明したものです。
# CMFは、ユーザーとアイテムの相互作用に加えて、ユーザー属性とアイテム属性も考慮することで、より精度の高い推薦を可能にするモデルです。

# CMFにおける拡張

# CMFでは、上記の式に加えて、ユーザー属性とアイテム属性も低ランク近似で因子分解します。

# U ≈ AC^T + μ_U
# I ≈ BD^T + μ_I
# U: ユーザー属性行列
# I: アイテム属性行列
# C: ユーザー属性因子行列
# D: アイテム属性因子行列
# μ_U: ユーザー属性の列平均
# μ_I: アイテム属性の列平均
# これらの式は、ユーザー属性とアイテム属性も潜在因子空間で表現することで、より詳細な情報を利用することを意味します。

# CMFの利点

# コールドスタート問題の緩和: 新規ユーザーや新規アイテムに対しても、属性情報を利用して推薦を生成できます。
# データスパース性の問題の緩和: ユーザーとアイテムの相互作用が少ない場合でも、属性情報を利用することで、より効果的な推薦が可能です。
# 推薦精度の向上: ユーザーとアイテムの属性情報を考慮することで、よりユーザーの嗜好に合ったアイテムを推薦できます。

# 具体例

# 映画レコメンデーション:
# ユーザー属性: 年齢、性別、職業、居住地など
# アイテム属性: ジャンル、監督、俳優、公開年など
# これらの属性情報を利用することで、ユーザーの年齢や性別に合わせた映画や、好みのジャンルに合った映画を推薦できます。
# ECサイトのレコメンデーション:
# ユーザー属性: 購入履歴、閲覧履歴、年齢、性別など
# アイテム属性: 商品カテゴリ、価格、ブランド、レビューなど
# これらの属性情報を利用することで、過去の購入履歴や閲覧履歴に基づいた商品だけでなく、ユーザーの属性に合った商品も推薦できます。
# CMFの応用
# CMFは、推薦システム以外にも、様々な分野で応用されています。

# 欠損値補完: 欠損値を持つデータに対して、CMFを用いて欠損値を推定することができます。
# 多層ネットワーク分析: 複数の関係性を持つネットワークデータを分析するために、CMFが利用されています。

# In addition, this package can also apply sigmoid transformations on the attribute columns which are binary.
# Note that this requires a different optimization approach which is slower than the ALS (alternating least-squares) method used here.

# one-hot encode policy_side_info
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
encoded_policy_side_info = enc.fit_transform(policy_side_info.drop("UserId", axis=1)).toarray()
encoded_policy_side_info = pd.DataFrame(encoded_policy_side_info, columns=enc.get_feature_names_out())
encoded_policy_side_info["UserId"] = policy_side_info["UserId"]

model_with_sideinfo = CMF(method="als", k=40, lambda_=1e+1, w_main=0.5, w_user=0.25, w_item=0.25)
model_with_sideinfo.fit(X=ratings, U=encoded_policy_side_info, I=area_side_info)

# pred = model_with_sideinfo.predict(user=test["UserId"], item=test["ItemId"])
# rmse = np.sqrt(mean_squared_error(pred, test["Rating"]))
# plot_actual_vs_predicted(test, pred, "cmf_model")
# --> 169 raise ValueError(msg_err)

# ValueError: Input contains NaN.

# (Note that, since the side info has variables in a different scale,
# even though the weights sum up to 1, it's still not the same as the earlier model w.r.t. the regularization parameter
# - this type of model requires more hyperparameter tuning too.)


# 3. Content-based model ---------------------------------------------------

# Content-based model (コンテンツベースモデル) はCMFとは異なります。ご指摘の通り、Content-based modelは、推薦対象のアイテムの特徴と、ユーザーの過去の行動や嗜好に基づいて推薦を生成する手法です。

# Content-based modelの特徴

# アイテムの特徴に焦点を当てる: アイテムの属性情報（映画のジャンル、監督、俳優など）を分析し、ユーザーが過去に好んだアイテムと類似した特徴を持つアイテムを推薦します。
# ユーザーの嗜好を学習: ユーザーの過去の行動（評価、購入履歴など）から、ユーザーがどのようなアイテムを好むのかを学習し、推薦に反映させます。
# 協調フィルタリングとの違い: 他のユーザーの行動を考慮しないため、コールドスタート問題（新規ユーザーや新規アイテムに対する推薦）に強いという利点があります。
# CMFとの違い
# 潜在因子: CMFは、ユーザーとアイテムを潜在因子空間に写像し、その空間における近さを基に推薦を生成します。一方、Content-based modelは、アイテムの特徴を直接的に利用し、潜在因子空間は使用しません。
# 属性情報の利用: CMFは、ユーザーとアイテムの属性情報を潜在因子空間に反映させることで、推薦精度を向上させます。Content-based modelは、アイテムの属性情報を直接的に推薦に利用します。
# コールドスタート問題: CMFは、属性情報を利用することでコールドスタート問題を緩和できますが、Content-based modelは、アイテムの特徴に基づいて推薦するため、コールドスタート問題にさらに強いです。
# ご提示いただいた式について

# X ≈ (UC)(ID)^T + μ
# この式は、Content-based modelの一種を表しています。

# UC: ユーザー属性行列Uと属性因子行列Cの積。ユーザーを属性因子空間に写像します。
# ID: アイテム属性行列Iと属性因子行列Dの積。アイテムを属性因子空間に写像します。
# (UC)(ID)^T: ユーザーとアイテムの属性因子空間における近さを表します。
# この式では、ユーザーとアイテムの相互作用行列Xを、ユーザーとアイテムの属性情報のみに基づいて近似しています。つまり、特定のユーザーやアイテムに対する自由なパラメータ（バイアスなど）を含んでいません。

# Content-based modelの例

# ニュース推薦: ユーザーが過去に読んだ記事のキーワードやカテゴリに基づいて、類似した記事を推薦する。
# 音楽推薦: ユーザーが過去に聴いた曲のジャンルやアーティストに基づいて、類似した曲を推薦する。
# 商品推薦: ユーザーが過去に購入した商品や閲覧した商品のカテゴリ、ブランド、価格帯などに基づいて、類似した商品を推薦する。
# まとめ

# Content-based modelは、アイテムの特徴とユーザーの嗜好に基づいて推薦を生成する手法であり、CMFとは異なるアプローチです。CMFは潜在因子空間を用いるのに対し、Content-based modelはアイテムの特徴を直接的に利用します。どちらの手法が優れているかは、データの特性や推薦の目的に依存します。

from cmfrec import ContentBased

model_content_based = ContentBased(k=40, maxiter=0, user_bias=False, item_bias=False)
model_content_based.fit(X=train,
                        U=encoded_policy_side_info,
                        I=area_side_info)

pred = model_content_based.predict(user=test["UserId"], item=test["ItemId"])
rmse = np.sqrt(mean_squared_error(pred, test["Rating"]))
plot_actual_vs_predicted(test, pred, "content_based_model")

# 4. Non-personalized model ------------------------------------------------

# This is an intercepts-only version of the classical model, which estimates one parameter per user and one parameter per item, and as such produces a simple rank of the items based on those parameters. It is intended for comparison purposes and can be helpful to check that the recommendations for different users are having some variability (e.g. setting too large regularization values will tend to make all personalzied recommended lists similar to each other).
from cmfrec import MostPopular

model_non_personalized = MostPopular(user_bias=True, implicit=False)
model_non_personalized.fit(ratings)

# ==========
# Examining top-N recommended lists
# ==========

# This section will examine what would each model recommend to the user with ID 948.
# This is the demographic information for the user:

user_side_info.loc[user_side_info["UserId"] == 948].T.where(lambda x: x > 0).dropna()
#                         947
# UserId                  948
# Gender_M               True
# Age_56                 True
# Occupation_programmer  True
# Zipcode_43056          True

# These are the highest-rated movies from the user:

(
    ratings
    .loc[lambda x: x["UserId"] == 948]
    .sort_values("Rating", ascending=False)
    .assign(Movie=lambda x: x["ItemId"].map(movie_id_to_title))
    .head(10)
)

# These are the lowest-rated movies from the user:
#         UserId  ItemId  Rating                                  Movie
# 146721     948    3789       5                 Pawnbroker, The (1965)
# 146889     948    2665       5    Earth Vs. the Flying Saucers (1956)
# 146871     948    2640       5                        Superman (1978)
# 146872     948    2641       5                     Superman II (1980)
# 147105     948    2761       5                 Iron Giant, The (1999)
# 146875     948    2644       5                         Dracula (1931)
# 146878     948    2648       5                    Frankenstein (1931)
# 147097     948    1019       5    20,000 Leagues Under the Sea (1954)
# 146881     948    2657       5  Rocky Horror Picture Show, The (1975)
# 146884     948    2660       5   Thing From Another World, The (1951)

(
    ratings
    .loc[lambda x: x["UserId"] == 948]
    .sort_values("Rating", ascending=True)
    .assign(Movie=lambda x: x["ItemId"].map(movie_id_to_title))
    .head(10)
)
#         UserId  ItemId  Rating                                              Movie
# 147237     948    1247       1                               Graduate, The (1967)
# 147173     948      70       1                         From Dusk Till Dawn (1996)
# 146768     948     748       1                                Arrival, The (1996)
# 147135     948      45       1                                  To Die For (1995)
# 146812     948     780       1                      Independence Day (ID4) (1996)
# 146813     948     788       1                        Nutty Professor, The (1996)
# 146814     948    3201       1                            Five Easy Pieces (1970)
# 147118     948     356       1                                Forrest Gump (1994)
# 146821     948    3070       1  Adventures of Buckaroo Bonzai Across the 8th D...
# 146822     948    1617       1                           L.A. Confidential (1997)

# Will exclude already-seen movies
exclude = ratings["ItemId"].loc[ratings["UserId"] == 948]
exclude_cb = exclude.loc[lambda x: x.isin(item_sideinfo_pca["ItemId"])]

# Recommended lists with those excluded
recommended_non_personalized = model_non_personalized.topN(user=948, n=10, exclude=exclude)
recommended_no_side_info = model_no_sideinfo.topN(user=948, n=10, exclude=exclude)
recommended_with_side_info = model_with_sideinfo.topN(user=948, n=10, exclude=exclude)
recommended_content_based = model_content_based.topN(user=948, n=10, exclude=exclude_cb)

from collections import defaultdict

# aggregate statistics
avg_movie_rating = defaultdict(lambda: 0)
num_ratings_per_movie = defaultdict(lambda: 0)
for i in ratings.groupby('ItemId')['Rating'].mean().to_frame().itertuples():
    avg_movie_rating[i.Index] = i.Rating
for i in ratings.groupby('ItemId')['Rating'].agg(lambda x: len(tuple(x))).to_frame().itertuples():
    num_ratings_per_movie[i.Index] = i.Rating

# function to print recommended lists more nicely
def print_reclist(reclist):
    list_w_info = [str(m + 1) + ") - " + movie_id_to_title[reclist[m]] +\
        " - Average Rating: " + str(np.round(avg_movie_rating[reclist[m]], 2))+\
        " - Number of ratings: " + str(num_ratings_per_movie[reclist[m]]) for m in range(len(reclist))]
    print("\n".join(list_w_info))

print("Recommended from non-personalized model")
print_reclist(recommended_non_personalized)
print("----------------")
print("Recommended from ratings-only model")
print_reclist(recommended_no_side_info)
print("----------------")
print("Recommended from attributes-only model")
print_reclist(recommended_content_based)
print("----------------")
print("Recommended from hybrid model")
print_reclist(recommended_with_side_info)