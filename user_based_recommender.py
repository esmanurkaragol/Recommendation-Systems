############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

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

#user movie indexinde kullanıcılar var bu nedenle bu indexten ratgele  "sample(1, ...)" 1 tane kullanıcı seçmek için örneklem oluştur.
#random_state = 45 diyerek de kullanıcı ıd'sını veriyoruz. bu kullanıcı üzerinden çalışıcaz.
import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
random_user   #biraz önce random user adında bir kullanıcı oluşturmuştuk.
user_movie_df
#random_user ı seç. böylece veriseti bu kullanıcıya göre indirgenmiş olacaktır.
random_user_df = user_movie_df[user_movie_df.index == random_user]

#na olmayan değerleri getirir. böylece kullanıcığının izlediği filmleri gelecektir.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

#satırlarda random_user kullanıcısını sec, sutunlarda ise izlemiş olduğu ilgili filmi bul. bize bu filme kaç puan verdiği bilgisini verir.
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

#kullanıcı toplamda kaç film izlemiş.
len(movies_watched)


#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
#user movie df ne izlenen filmler[movies_watched] listesini sorarak veri setini indirgeyelim. veiriyi özelleştiriyoruz böylece.
movies_watched_df = user_movie_df[movies_watched]

#her bir kullanıcının dolu olan satırlarını saydır. (notnull=boş olmayan)
user_movie_count = movies_watched_df.T.notnull().sum()

#userId bilgisini değişkene çevirmek istiyorum. artık bunlar index olmaktan kurtuldu.
user_movie_count = user_movie_count.reset_index()

#1.değişkenin adı=userId; ikinci değişkenin adı =movie_count olsun diyorum.
user_movie_count.columns = ["userId", "movie_count"]

#sinan 33 film izlemişti. bende şimdi sinanın izlemiş olduğu filmlerden en az 20 tanesini izleyen kullanıcıları listelemek istiyorum. böylece veri daha da indirgenecektir.
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

#sinanın izlediği 33 filmi izleyen kullanıcı sayısını bulmak için;
user_movie_count[user_movie_count["movie_count"] == 33].count()

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

#ben sinanla en az %60 oranında aynı filmi izlesinler diye en az 20 tane aynı filmi izlemiş olmalarını bekliyorum.
# yalnız bunu daha programatik bir şekilde yapman lazım. bunun için len ile movies_watchesın boyutuna bak.bunun %60 hesapla.
# ve burada 20 yazmak yerine gel bu oranı (perc)gir.
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

#movies_watched_df.index.isin(...)= izlenen filmin indexlerinde ... ara
#isin(users_same_movies) =kullanıcıyla aynı filmleri izleyenleri ara,orada olanları seç.
#random_user_df[movies_watched] ile birleştiriyoruz (concat). sinan ile diğer kullanıcıların veriisni birleştiriyoruz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

#[3203 rows (kullanıcı) x 33 columns (filmler)]. sütünlara kullanıcıları almalıyız
final_df.shape

#kullanıcıları sütunlara çekmek için;
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

#user_id_1 = sinanı yani ilgili kullanıcıyı al. ve corelasyon sutununda %65 den byk olanları (benim kendi varsayımım) getir diyorum.
#user_id_2 = sinan ile benzer davranışları olan diğer kullanıcıları al. ve corr seç. en sonda indexi resetle.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

#user_id_' değişkenini user_ıd olarak değiştir diyorum.
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#top_users df ile rating.csv dosyasını birleştiriyorum.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

#sinanı çıkar...!= random_user
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################
#eğer eliimizde corr değerleri olmasaydı sinanla benzer bu kullanıcıların movie_id lerine göre groupby alıp, ratinglerinin ortlamasını alıp sıralardım.
# bu durumda ratinge göre sıralayıp yorumlama yaparsan bütün userların sinanla olan ilişkisnin aynı olduğunu varsayacaksın.
#bu nednele corr göre sıralayıp, ratingleri mi değerlendirsek; o zaman da bazı ratingler byk kimileri küçük.
#burada yapacağın işlem hem cor hem de ratinglerin etkisini eş zamanlı olarak incelemeli. bunun için 'weighted_rating' adında değişken oluşturalım.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

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

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5 bilgileri dışarıdan verilir:
#ratio=x --> sinan ile yuzde x ortak film izlemiş olsun
#corr_th--> sinan ile ortak filmleri izlemiş olabilir okey ama beğendi mi beğenmedi mi? bunun da ön tanımlı değerini 0.65 olarak giriyoruz.
def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


