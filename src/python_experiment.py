import pickle
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import numpy as np, pandas as pd, re

# load pure_premium data
# df = pd.read_csv("data/pure_premium.csv", index_col=0)
# df_org = pd.read_csv("data/brvehins.csv", index_col=0)
# cols_to_use = ["VehModel", "Area", "ExposTotal", "ClaimTotal"]
# df = df_org[cols_to_use].groupby(["VehModel", "Area"]).agg({"ExposTotal": "sum", "ClaimTotal": "sum"}).reset_index()
# df["pure_premium"] = np.where(df["ExposTotal"] > 0, df["ClaimTotal"] / df["ExposTotal"], np.nan)

# all_models = df["VehModel"].unique()
# all_areas = df["Area"].unique()

# df = df.dropna()
# df = df[['VehModel', 'Area', 'pure_premium']]
# df = df.reset_index(drop=True)
# df = df.rename(columns={"VehModel": "UserId", "Area": "ItemId", "pure_premium": "Rating"})
# df["Rating"] = df["Rating"].astype(float)

# # split df into train and test
# # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ratings = df.copy()
# users_train, users_test = train_test_split(ratings["UserId"].unique(), test_size=0.2, random_state=1)
# items_train, items_test = train_test_split(ratings["ItemId"].unique(), test_size=0.2, random_state=2)

# ratings_train, ratings_test1 = train_test_split(ratings.loc[ratings["UserId"].isin(users_train) &
#                                                             ratings["ItemId"].isin(items_train)],
#                                                 test_size=0.2, random_state=123)

# users_train = ratings_train["UserId"].unique()
# items_train = ratings_train["ItemId"].unique()
# ratings_test1 = ratings_test1.loc[ratings_test1["UserId"].isin(users_train) &
#                                   ratings_test1["ItemId"].isin(items_train)]

# m_classic = CMF(k=40).fit(ratings_train)
# from sklearn.metrics import mean_squared_error

# pred_ratingsonly = m_classic.predict(ratings_test1["UserId"], ratings_test1["ItemId"])
# print("RMSE type 1 ratings-only model: %.3f [rho: %.3f]" %
#       (np.sqrt(mean_squared_error(ratings_test1["Rating"],
#                                   pred_ratingsonly,
#                                   squared=True)),
#        np.corrcoef(ratings_test1["Rating"], pred_ratingsonly)[0,1]))

# //compare pred_ratingsonly with actual ratings
# plt.scatter(ratings_test1["Rating"], pred_ratingsonly)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs Predicted")
# plt.show()

# # https://github.com/david-cortes/cmfrec/

# from cmfrec import CMF

# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# model_no_sideinfo = CMF( # all default parameters
#     k=40,
#     lambda_=10.0,
#     method='als',
#     use_cg=True,
#     user_bias=True,
#     item_bias=True,
#     center=False, #True,
#     add_implicit_features=False,
#     scale_lam=False,
#     scale_lam_sideinfo=False,
#     scale_bias_const=False,
#     k_user=0,
#     k_item=0,
#     k_main=0,
#     w_main=1.0,
#     w_user=1.0,
#     w_item=1.0,
#     w_implicit=0.5,
#     l1_lambda=0.0,
#     center_U=True,
#     center_I=True,
#     maxiter=800,
#     niter=10,
#     parallelize='separate',
#     corr_pairs=4,
#     max_cg_steps=3,
#     precondition_cg=False,
#     finalize_chol=True,
#     NA_as_zero=False,
#     NA_as_zero_user=False,
#     NA_as_zero_item=False,
#     nonneg=True, #nonneg=False,
#     nonneg_C=False,
#     nonneg_D=False,
#     max_cd_steps=100,
#     precompute_for_predictions=True,
#     include_all_X=True,
#     use_float=True,
#     random_state=1,
#     verbose=False,
#     print_every=10,
#     handle_interrupt=True,
#     produce_dicts=False,
#     nthreads=-1,
#     n_jobs=None,
# )
# model_no_sideinfo.fit(train_df)

# # get predictions of train_df
# new_data = test_df.copy()
# predictions = model_no_sideinfo.predict(new_data["UserId"], new_data["ItemId"])
# # predictions = model_no_sideinfo.predict(new_data)
# print(predictions)

# plt.scatter(new_data["Rating"], predictions)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.xlim(-300, 30000)
# plt.ylim(-300, 30000)
# plt.title("Actual vs Predicted")
# plt.show()

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(df.drop('pure_premium', axis=1), df['pure_premium'], test_size=0.2, random_state=42)


# https://cmfrec.readthedocs.io/en/stable/

import numpy as np
import cmfrec

# Sample data (user-item interactions)
# Format: user_id, item_id, rating
data = np.array([
    [0, 0, 5],
    [0, 1, 3],
    [1, 0, 4],
    [1, 2, 2],
    [2, 1, 5],
    [2, 2, 1],
    [3, 0, 3],
    [3, 3, 4],
    [4, 1, 2],
    [4, 3, 5],
])

# Create a sparse matrix (CSR format)
n_users = data[:, 0].max() + 1
n_items = data[:, 1].max() + 1


# documentation: https://cmfrec.readthedocs.io/en/latest/

from scipy.sparse import csr_matrix
rows = data[:, 0]
cols = data[:, 1]
values = data[:, 2]

X = csr_matrix((values, (rows, cols)), shape=(n_users, n_items))


# tutorial notebook (movie lens dataset):
# https://nbviewer.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb
# includes model with side information (should be useful for our case)

ratings_path = "/Users/spectee/research/act-cmfreq/data/ratings.dat"
ratings = pd.read_csv(ratings_path, sep="::", header=None, engine='python')
ratings.columns = ["user_id", "item_id", "rating", "timestamp"]



movies_path = "/Users/spectee/research/act-cmfreq/data/movies.dat"
//load dat_path and movies_path

# Initialize and fit the model
model = cmfrec.CMF(method="als", k=40, lambda_=1e+1)
model.fit(X)

# Make predictions
user_id = 0
item_id = 2
prediction = model.predict(user_id, item_id)
print(f"Prediction for user {user_id}, item {item_id}: {prediction}")

# Get predictions for all users and items (dense matrix)
predictions = model.predict(np.arange(n_users), np.arange(n_items).reshape(1,-1)) #must be a 2d array for item ids
print("\nPredictions for all users and items:\n", predictions)

# Get top-N recommendations for a user
top_n = 3
recommendations = model.topN(user_id, top_n=top_n)
print(f"\nTop {top_n} recommendations for user {user_id}: {recommendations}")

#Example with confidence weights
data_conf = np.array([
    [0, 0, 5, 1.0],
    [0, 1, 3, 0.8],
    [1, 0, 4, 1.0],
    [1, 2, 2, 0.5],
    [2, 1, 5, 1.0],
    [2, 2, 1, 0.6],
    [3, 0, 3, 0.9],
    [3, 3, 4, 1.0],
    [4, 1, 2, 0.7],
    [4, 3, 5, 1.0],
])

rows_conf = data_conf[:, 0]
cols_conf = data_conf[:, 1]
values_conf = data_conf[:, 2]
confidences = data_conf[:, 3]

X_conf = csr_matrix((values_conf, (rows_conf, cols_conf)), shape=(n_users, n_items))
C = csr_matrix((confidences, (rows_conf, cols_conf)), shape=(n_users, n_items))

model_conf = cmfrec.CMFrec(n_factors = 2)
model_conf.fit(X_conf, C)

prediction_conf = model_conf.predict(user_id, item_id)
print(f"\nPrediction with confidence for user {user_id}, item {item_id}: {prediction_conf}")

# Example with explicit biases
model_biased = cmfrec.CMFrec(n_factors=2, use_bias=True)
model_biased.fit(X)

prediction_biased = model_biased.predict(user_id, item_id)
print(f"\nPrediction with bias for user {user_id}, item {item_id}: {prediction_biased}")





# create a recommendation system model with matrix factorization
from cmfrec import CMF

# create a model
model = CMF(k=10, k_main=10, k_user=10, k_item=10, reg_param=1e-3, n_iter=100, random_seed=123)

# fit the model
model.fit(X_train, y_train)




# Split the data into features and target
X = data.drop('pure_premium', axis=1)
y = data['pure_premium']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions

