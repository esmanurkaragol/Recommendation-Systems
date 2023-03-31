#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması



#VERİ SETİNDEKİ DEĞİŞKENLER
# BUDGET = FİLMLERİN BÜTÇELERİ
# GENRES = FİLMLERİN TÜRLERİ
# HOMEPAGE= FİLMLERİN ANASAYFALARI
# İD = FİLMLERİN VERİ SETİ İÇİNDEKİ İD LERİ
# İMDB_İD = FİLMLEİRN İMBD DEKİ ID LERİ
# LANGUAGE = FİLMLERİN DİLLERİ
# ORIGINAL_TITLE = FİLMLERİN ORİJİNAL İSİMLERİ
# OVERVİEW = FİLMLERİN AÇIKLAMALARI



#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("C:/Users/esman/PycharmProjects/recommender_system/datasets/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin low_memory=false girilir.
df.head()
df.shape

df["overview"].head()

#metinlerin içinde ölçüm değeri taşımayan kelimeleri (and, the, on, in...gibi) veri setinden çıkarmak istiyorum.
#neden? çünkü bu değerler boş gözlemdir ve ölçüm değeri taşımazlar.
tfidf = TfidfVectorizer(stop_words="english")


# df[df['overview'].isnull()]
#eksik değerleri " " boşluk ile değiştir.
df['overview'] = df['overview'].fillna('')
#fit_transform = fit edilen değerleri kalıcı olarak değiştirir.
tfidf_matrix = tfidf.fit_transform(df['overview'])

#satırlarda açıklamalar var
#sütunlarda ise eşsiz kelimeler var.
#satır ve sütun kesişimlerinde ise tf-df scoreları vardır
tfidf_matrix.shape

#title isimlerdi demek ki 45466 tane isim var.
df['title'].shape

#sütunlardaki tüm kelimelerin isimlerini getir diyorum.
tfidf.get_feature_names()

#toarray ile tfidf in durumunu gözlemleyelim
tfidf_matrix.toarray()



#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

#cosine_similarity içerisine benzerliğini hesaplamak istediğimiz matris girilir. tek veya iki argüman girilebilir.
cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)
#burada (45466,45466) değeri = 45466 tane filmin/ dokümanın/ ürünün/ açıklamanın olduğunu gösterir
cosine_sim.shape

#1.indexdeki filmin diğer tüm filmlerle benzerlik skorunu verir. amam pek anlaşılır bir çıktı vermez. bu nedenle title bilgisini kullan.
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

#pd.series içine index bilgileri ve filmlerin isimleri girilir.
indices = pd.Series(df.index, index=df['title'])

#indices içinde index bilgilerine git ve bunları saydır. kaç filmden kaç tane olduğu bilgisi gelecektir.
indices.index.value_counts()


#title larda çoklama var. mesela cinderalla filminden 11 tane varsa sen bana en son çekilen cinderellayı bırak, diğerlerini uçur diyorum. böylece çoklamalardan kurtulurum.
#duplicated ile veriler gezilir veriler 1 kere görülünce tutar, ikinci gördüğünde ise bu duplice der. yani çoklama. true döner.duplicated fonksiyonunun ön tanımlı değeri keep= first dür. ilk gördüğünü tut, diğerini sil demek. eğer sen tam tersini yapmak istiyorsan keep = "last" girilir.
#çoklama olanları tutar böylece. ama ben çoklama olmayanları istiyorum bu nedenle ~ işareti başa getirilir.
#böylece her bir title tekilleştirildi.
indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]
#sherlock holmes ile diğer filimler arasındaki benzerlik skorlarına erişmek için;
cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
#evet öneriler geldi:
df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title bilgisine karşılık gelen index'ini yakalama
    movie_index = indices[title]
    # title'a gore yakaladığımız movie_index kullanıralak benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

#sherlock holmes izleyenlere önerilecek filmleri getir.
content_based_recommender("Sherlock Holmes", cosine_sim, df)

#the matrix izleyenlere önerilecek filmleri getir.
content_based_recommender("The Matrix", cosine_sim, df)

#the godfather izleyenlere önerilecek filmleri getir.
content_based_recommender("The Godfather", cosine_sim, df)

#the dark knight rises izleyenlere önerilecek filmleri getir.
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

#cosine_similarity hesapladığımız kısmı fonksiyonlaştıralım.
def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
#film id leri [bu filmi izleyenlere önereceğin filmlerin id lerini gir]
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3

#bu işlemleri büyük verilerde yapmak istediğinde en çok izlenen 100 filmi sql veri tabanına alırsın.
# bu işlemleri yaaprsın, her film için bir öenri setleri oluşturusun.
# ve bunları izleyenlere neleri önereceğini de sql tablo formunda tutarsın.