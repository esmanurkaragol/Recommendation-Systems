
#GÖREV1:

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

#adım 1:
movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")

#adım2:
df = movie.merge(rating, how="left", on="movieId")
df.head()

#adım3:
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
df.shape
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape

#adım4:
user_movie_df = common_movies.pivot_table( index=["userId"], columns=["title"], values="rating" )
user_movie_df.head()

#adım 5
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("datasets/movie.csv")
    rating = pd.read_csv("datasets/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df
user_movie_df = create_user_movie_df()

#GÖREV 2

#ADIM1
random_user = 100170

#adım2
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()

#adım3
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched

#GÖREV3
