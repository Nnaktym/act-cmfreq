"""test cmfrec in python with reference to the following official examples
https://nbviewer.org/github/david-cortes/cmfrec/blob/master/example/load_data.ipynb
https://nbviewer.org/github/david-cortes/cmfrec/blob/master/example/cmfrec_movielens_sideinfo.ipynb"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmfrec import CMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def plot_actual_vs_predicted(
    actual: np.ndarray,
    pred: np.ndarray,
    fig_name: str,
    rmse: float = None,
    max_limit: float = 2500,
    fig_dir: str = "figs/",
):
    """Plot actual vs predicted ratings and show the figure."""
    assert len(actual) == len(pred), "Lengths of test and pred should be the same"
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, pred, color="black", alpha=0.9)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.plot([0, max_limit], [0, max_limit], color="red")
    plt.xlim(0, max_limit)
    plt.ylim(0, max_limit)
    plt.grid(True)
    title_text = f"Actual vs Predicted: {fig_name}"
    if rmse is not None:
        title_text += f" RMSE: {rmse}"
    plt.title(title_text)
    plt.savefig(f"{fig_dir}{fig_name}.png")
    plt.close()
    return None


def create_df_with_full_user_item(
    original_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    user_col="VehModel",
    item_col="Area",
):
    """Create a full dataframe with all users and items."""
    all_users = original_data[user_col].unique()
    all_items = original_data[item_col].unique()
    full_df = pd.DataFrame()
    full_df[user_col] = np.repeat(all_users, len(all_items))
    full_df[item_col] = np.tile(all_items, len(all_users))
    full_df = full_df.merge(processed_data, on=[user_col, item_col], how="left")
    return full_df


def encode_side_info(side_info: pd.DataFrame, keep_cols: list):
    """Encode side info."""
    enc = OneHotEncoder()
    encoded_side_info = enc.fit_transform(side_info.drop(keep_cols, axis=1)).toarray()
    encoded_side_info = pd.DataFrame(
        encoded_side_info,
        columns=enc.get_feature_names_out(),
    )
    encoded_side_info.loc[:, keep_cols] = side_info[keep_cols]
    return encoded_side_info


if __name__ == "__main__":
    # 0. Settings -------------------------------------------------------------
    processed_data_path = "data/all_data.csv"  # from brazil_data_analysis_R.ipynb
    original_data_path = "data/brvehins_org.csv"  # from brazil_data_analysis_R.ipynb
    population_density_path = "data/brazil_population_density.csv"
    pop_density_quantile = 0.7
    k = 50  # number of latent factors
    lambda_ = 10  # regularization parameter

    cols_to_use = ["VehModel", "Area", "pure_premium"]
    user_side_info_cols = ["VehModel", "VehGroup"]
    item_side_info_cols = ["Area", "pop_density_class"]
    ratings_cols = ["pure_premium"]

    # 1. Load and preprocess data ---------------------------------------------

    # brazil ins data in original format
    original_data = pd.read_csv(original_data_path)

    # brazil ins data in processed format
    processed_data = pd.read_csv(processed_data_path, index_col=0)

    # import population density data by area (will be used as item side info)
    population_density = pd.read_csv(population_density_path)
    threshold_of_density = population_density["density_km2"].quantile(
        pop_density_quantile
    )
    population_density.loc[:, "pop_density_class"] = np.where(
        population_density["density_km2"] > threshold_of_density, "urban", "rural"
    )

    # create mapping for VehModel to VehGroup (will be used as user side info)
    model_group_mapping = original_data[["VehModel", "VehGroup"]].drop_duplicates()
    model_group_mapping = model_group_mapping.set_index("VehModel")["VehGroup"]

    # create full dataframe with all users and items
    df = create_df_with_full_user_item(original_data, processed_data)
    df = df.join(model_group_mapping, on="VehModel", how="left")
    df = pd.merge(df, population_density, on="Area", how="left")

    # datasets for ratings, user side info, and item side info
    ratings = df[cols_to_use]
    ratings = ratings.rename(
        columns={"VehModel": "UserId", "Area": "ItemId", "pure_premium": "Rating"}
    )

    user_side_info = df[user_side_info_cols]
    user_side_info = user_side_info.rename(columns={"VehModel": "UserId"})

    item_side_info = df[item_side_info_cols]
    item_side_info = item_side_info.rename(columns={"Area": "ItemId"})

    # split into train and test
    train, test = train_test_split(ratings, test_size=0.2, random_state=123)

    # 2. Classical model (simple MF) ------------------------------------------

    # ユーザーとアイテムの相互作用を表す行列を、ユーザー因子行列とアイテム因子行列に分解することで、ユーザーの嗜好とアイテムの特徴を表現
    # Rのノートブックと同じ。異なるのは乱数のみ。
    #
    # X ≈ AB^T + μ + b_A + b_B
    # X: ユーザーとアイテムの相互作用行列 (評価値など)
    # A: ユーザー因子行列
    # B: アイテム因子行列
    # μ: 全体の平均値
    # b_A: ユーザーバイアス
    # b_B: アイテムバイアス

    model_no_sideinfo = CMF(
        k=20,
        lambda_=10.0,
        method="als",
        center=False,
        nonneg=True,
    )
    model_no_sideinfo.fit(train.dropna())  # )

    pred = model_no_sideinfo.predict(
        user=test.dropna()["UserId"], item=test.dropna()["ItemId"]
    )
    rmse = np.sqrt(mean_squared_error(pred, test.dropna()["Rating"]))
    plot_actual_vs_predicted(
        actual=test.dropna()["Rating"],
        pred=pred,
        rmse=np.round(rmse, 2),
        fig_name="classical_model",
    )

    # 3. Collective Matrix Factorization (CMF) -------------------------------

    # https://www.researchgate.net/publication/354733006_Improving_cold-start_recommendations_using_item-based_stereotypes
    # ユーザーとアイテムの相互作用に加えて、ユーザー属性とアイテム属性も考慮することで、より精度の高い推薦を可能にする
    # CMFでは、上記の式に加えて、ユーザー属性とアイテム属性も低ランク近似により潜在因子空間で表現する

    # U ≈ AC^T + μ_U
    # I ≈ BD^T + μ_I
    # U: ユーザー属性行列
    # I: アイテム属性行列
    # C: ユーザー属性因子行列
    # D: アイテム属性因子行列
    # μ_U: ユーザー属性の列平均
    # μ_I: アイテム属性の列平均

    # CMFの利点
    # コールドスタート問題の緩和: 新規ユーザーや新規アイテムに対しても、属性情報を利用して推薦を生成できる
    # データスパース性の問題の緩和: ユーザーとアイテムの相互作用が少ない場合でも、属性情報を利用することで、より効果的な推薦が可能
    # 推薦精度の向上: ユーザーとアイテムの属性情報を考慮することで、よりユーザーの嗜好に合ったアイテムを推薦できる

    # 具体例 映画レコメンデーション:
    # ユーザー属性: 年齢、性別、職業、居住地など
    # アイテム属性: ジャンル、監督、俳優、公開年など

    # 今回:
    # ユーザー（車種）属性: 車種グループ
    # アイテム（地域）属性: 人口密度グループ（都市部、地方部）

    encoded_user_side_info = encode_side_info(user_side_info, ["UserId"])
    encoded_item_side_info = encode_side_info(item_side_info, ["ItemId"])

    model_with_sideinfo = CMF(
        k=k,
        lambda_=lambda_,
        method="als",
        center=False,
        nonneg=True,
        w_main=0.5,
        w_user=0.25,
        w_item=0.25,
    )

    valid_idx = train.dropna().index
    model_with_sideinfo.fit(
        X=ratings.iloc[valid_idx],
        U=encoded_user_side_info.iloc[valid_idx],
        I=encoded_item_side_info.iloc[valid_idx],
    )
    pred = model_with_sideinfo.predict(
        user=test.dropna()["UserId"], item=test.dropna()["ItemId"]
    )
    rmse = np.sqrt(mean_squared_error(pred, test.dropna()["Rating"]))
    plot_actual_vs_predicted(
        actual=test.dropna()["Rating"],
        pred=pred,
        rmse=np.round(rmse, 2),
        fig_name="cmf_model",
    )
