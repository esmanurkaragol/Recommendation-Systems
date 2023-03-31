#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning (dışsal parametreleri optimize etme)
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

#movie ID df içinde movie_ids var mı? varsa bunları seç (isin). Buradaki ıd lere göre bir veri seti oluşturuyorum.
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

#pivot table methodunu kullan
user_movie_df = sample_df.pivot_table(index=["userId"],     # satırları userıd leri
                                      columns=["title"],    # sütunlara titleları yani filmleri
                                      values="rating")      #kesişim değeri olarak da ratingleri

user_movie_df.shape

#surprise kütüphanesi için ratinglerin skalasını vermemiz gerekiyor. bu nedenle reader methodu ile bir bilgilendirme verilir.
reader = Reader(rating_scale=(1, 5))

#surprise kütüphanesinin istediği forma veriyi getirelim;
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################
#modeller belirli bir eğitim seti üzerinde kurulur. daha sonra modelin görmediği başka bir test seti üzerinde test edilir. böylece ispatlanmış olunur.
#train_set üzerinde model kurucaz.
#test_set üzerinden test edicez. bakalım kurmuş olduğum model hiç görmediği veride iyi bir şekilde çalışıyor mu? bunu test etmek için test_set üzerinde modeli test edeceğiz.
#test_size=.25 --> yüzde 25 i test _set olsun. kalan yuzde 75 ise train_setine bölünüsin. bunun için surprise kütüphanesinde yer alan train_test_split methodu  kullanılır.
trainset, testset = train_test_split(data, test_size=.25)

#SVD model nesnesi getirilir. bu matrix_factorization yöntemini kullanacağımız fonksiyondur.
svd_model = SVD()
#train_set üzerinden öğren
svd_model.fit(trainset)             #burada p ve q ağırlıklarını bulduk
#şimdi test_set üzerinde uygula
predictions = svd_model.test(testset)   #bulmuş olduğun p,q ağırlıklarını kullanarak test setinde tahminlerde bulunç

#accuracy aracılığıyla tahminlerdeki sapmaları öğreniriz
#rmse = Root Mean Squared Error--Kök Ortalama Karekök Hatası
accuracy.rmse(predictions)
#çıkan rmse değeri = 0.92 . burada diyelim ki hasan bey filme 4 puann verdi bende bunun tahmini için 4+0.92 veya 4-0.92 diyebilirim.
#bir tahminde bulunduğumda yapmam gereken ortalama hatadır.

#kullanıcının kaç puan verdiğini tahmşn et;
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)

sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning = modelin tahmin performansını arttırmaya çalışmaktır
##############################

#olası bazı parametre setlerini dışarıdan vermek demek --> parametre grid oluşturmak demktir.
#bir sözlük aracılığıyla epochs ve lr değerleri girilir.
#olası kombinasyon sayısını daha fazla yaparak daha iyi sonuçlar elde edeceksindir.
#bu iki değerin olası tüm kombinasyonlarını dene.
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}


#svd = model nesnem
#rmse = Root Mean Squared Error
#mae= Mean Absolute Error --- “Hatanın mutlak ortalaması”
gs = GridSearchCV(SVD,
                  param_grid,                #girmiş olduğun olası tüm parametre çiftlerini dene
                  measures=['rmse', 'mae'],  #bu denemeler neticesinde hatamı rase ve mae metrikleri ile değerlendir. bu metrikler bağlamında başarıyı değerlendirir.
                  cv=3,                      #çapraz doğrulama yapmak için cv=3--> 3 katlı çapraz doğrulama yap. yani veri setini 3 e böl. 2 parçasıyla model kur. 1 parçasıyla test et. sonra kalan diğer 2 siyle model kur 1ini dışarı çıkar vs..
                  n_jobs=-1,                 #işlemcileri full performansıyla kullan demek.
                  joblib_verbose=True)      #işlemler gerçekleşirken bana raporlama yap.
gs.fit(data) #fit et yani modelle

gs.best_score['rmse']  #0.93
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

# ** 2 yıldız sonrasında ilgili sözlük yapısını gönderirsen, bu modeli yeni değerleri ile oluşturacaktır.
svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






