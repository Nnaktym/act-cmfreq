import pickle
import pandas as pd


# preprocess movie lens data
# https://nbviewer.org/github/david-cortes/cmfrec/blob/master/example/load_data.ipynb

import numpy as np, pandas as pd, re

ratings = pd.read_table(
    'data/ml-1m/ratings.dat',
    sep='::', engine='python',
    names=['UserId','ItemId','Rating','Timestamp']
)
ratings = ratings.drop("Timestamp", axis=1)
ratings.head()

print("Number of users: %d" % ratings["UserId"].nunique())
print("Number of items: %d" % ratings["ItemId"].nunique())
print("Number of ratings: %d" % ratings["Rating"].count())


movie_titles = pd.read_table(
    'data/ml-1m/movies.dat',
    sep='::', engine='python', header=None, encoding='latin_1',
    names=['ItemId', 'title', 'genres']
)
movie_titles = movie_titles[['ItemId', 'title']]

movie_titles.head()

movie_id_to_title = {i.ItemId: i.title for i in movie_titles.itertuples()}


# loading the tag genome
movies = pd.read_csv('data/ml-25m/movies.csv')
movies = movies[['movieId', 'title']]
movies = pd.merge(movies, movie_titles)
movies = movies[['movieId', 'ItemId']]

tags = pd.read_csv('data/ml-25m/genome-scores.csv')
tags_wide = tags.pivot(index='movieId', columns='tagId', values='relevance')
tags_wide.columns=["tag"+str(i) for i in tags_wide.columns]

item_side_info = pd.merge(movies, tags_wide, how='inner', left_on='movieId', right_index=True)
item_side_info = item_side_info.drop('movieId', axis=1)
item_side_info.head()

# Dimensionality reduction for the tag genome through PCA

from sklearn.decomposition import PCA

pca_obj = PCA(n_components = 50)
item_sideinfo_reduced = item_side_info.drop("ItemId", axis=1)
item_sideinfo_pca = pca_obj.fit_transform(item_sideinfo_reduced)

item_sideinfo_pca = pd.DataFrame(
    item_sideinfo_pca,
    columns=["pc"+str(i+1) for i in range(item_sideinfo_pca.shape[1])]
)
item_sideinfo_pca['ItemId'] = item_side_info["ItemId"].to_numpy()
item_sideinfo_pca = item_sideinfo_pca[["ItemId"] + item_sideinfo_pca.columns[:50].tolist()]
item_sideinfo_pca.head()

# Loading the states data

zipcode_abbs = pd.read_csv("data/states.csv", low_memory=False)
zipcode_abbs_dct = {z.State: z.Abbreviation for z in zipcode_abbs.itertuples()}
us_regs_table = [
    ('New England', 'Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont'),
    ('Middle Atlantic', 'Delaware, Maryland, New Jersey, New York, Pennsylvania'),
    ('South', 'Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, Missouri, North Carolina, South Carolina, Tennessee, Virginia, West Virginia'),
    ('Midwest', 'Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin'),
    ('Southwest', 'Arizona, New Mexico, Oklahoma, Texas'),
    ('West', 'Alaska, California, Colorado, Hawaii, Idaho, Montana, Nevada, Oregon, Utah, Washington, Wyoming')
    ]
us_regs_table = [(x[0], [i.strip() for i in x[1].split(",")]) for x in us_regs_table]
us_regs_dct = dict()
for r in us_regs_table:
    for s in r[1]:
        us_regs_dct[zipcode_abbs_dct[s]] = r[0]

# loading user demographic information
users = pd.read_table(
    'data/ml-1m/users.dat',
    sep='::', engine='python', encoding='cp1252',
    names=["UserId", "Gender", "Age", "Occupation", "Zipcode"]
)
# users["Zipcode"] = users["Zipcode"].map(lambda x: int(re.sub("-.*", "", x)))
# users = pd.merge(users, zipcode_info, on='Zipcode', how='left')
# users['Region'] = users["Region"].fillna('UnknownOrNonUS')

occupations = {
    0:  "\"other\" or not specified",
    1:  "academic/educator",
    2:  "artist",
    3:  "clerical/admin",
    4:  "college/grad student",
    5:  "customer service",
    6:  "doctor/health care",
    7:  "executive/managerial",
    8:  "farmer",
    9:  "homemaker",
    10:  "K-12 student",
    11:  "lawyer",
    12:  "programmer",
    13:  "retired",
    14:  "sales/marketing",
    15:  "scientist",
    16:  "self-employed",
    17:  "technician/engineer",
    18:  "tradesman/craftsman",
    19:  "unemployed",
    20:  "writer"
}
users['Occupation'] = users["Occupation"].map(occupations)
users['Age'] = users["Age"].map(lambda x: str(x))
users.head()

user_side_info = pd.get_dummies(users[['UserId', 'Gender', 'Age', 'Occupation', 'Zipcode']])
user_side_info.head()

print("Number of users with demographic information: %d" %
      user_side_info["UserId"].nunique())

pickle.dump(ratings, open("data/ratings.p", "wb"))
pickle.dump(item_sideinfo_pca, open("data/item_sideinfo_pca.p", "wb"))
pickle.dump(user_side_info, open("data/user_side_info.p", "wb"))
pickle.dump(movie_id_to_title, open("data/movie_id_to_title.p", "wb"))

