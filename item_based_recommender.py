###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması (satırlarda users, sütunlarda movie olacak şekilde veri yapısını oluşturacağız. )
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.head()
df.shape

#27262 tane film var
df["title"].nunique()

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())


#1000den az yorum alanları dışarıda bırakmak istiyorum. bunları rare_movies değişkeninde tutuyorum.
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
# ~ bunların dışındakileri seç
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()

#3159 tane film var artık.
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
#corrwith fonksiyonu ile ilgili filmin (movie_name) korelasyonuna bakılır.
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
#yukarıda biz iflmlerin yıl bilgisine kadar yazdık sürekli böyle df gezecek miyim?
#bunun için ilk yol olarak rastgele filmler seçebilirsin. bir seri oluştuuryorum.df sütunlarında gezip bir sample (örneklem) alıyorum. gelen değerin sadece string bölümünü istediğim için values[0] giriyorum;
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

#işlemi fonksiyonlaştır. bir keyword girilir. girilen keywordün yakalandığı diğer filmler getirilir.
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)



######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





