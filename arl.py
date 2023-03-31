############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("C:/Users/esman/PycharmProjects/recommender_system/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

#veriyi  yukarıda ki gibi okuduğunda sıkıntı çıkarırsa  "openyxl" indir ve öyle çalıştır.
# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")


#klasik veri ön işleme yapalım önce. sonra arl için özel veri işleme yapacağız.
#describe sadece sayısal değişkenleri betimler. daha okunabilir olması adına da transpozunu alıyorum.
df.describe().T   #min. değeerler, price eksi değerlerde ama bu olamaz. veri setinde iade edilen ürünlerden kaynaklıdır.
df.isnull().sum() #eksik değerleri gör
df.shape


#eksik değerleri uçuracağız. invoice da C olanları uçuracağız. Birde aykırı değerleri kırpacağız.
#bunun için fonksiyon tanımlayacağız.
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

df.describe().T
df.isnull().sum()
df.shape

#eşik değerleri hesağlayacak fonksiyon
#dataframe[variable] = datafame içinde olan değişkenlerden [variable] adında olanı seç. bunun çeyrek değerini (.quantile(0.01)) hesapla. ve bu değeri quartile1 olarak tut.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    #üst çeyrek değerden alt çeyrek değeri çıkartığımızda iqr değeri geldi. yani dağılımı, değişimi geldi. "değişkenin değişimini" ifade etmektedir.
    interquantile_range = quartile3 - quartile1
    #alt üst aşağı limitleri belirle
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #ilgili değişkende lowlimitden aşağı olanları getir ve bunları low limitle değiştir.
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df.head()
#buradaki invoicelar bizim sepetimiz olacak, işlemler olacak. sütunlarda ise ürünler olacak.
# faturada ilgili ürünün olup olmama durumunu göre 1-0 lar yazacak
# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

#veri setini indirgemek için sadece fransa üzerinden işlemleri yapıcam.
df_fr = df[df['Country'] == "France"]
#invoice ve description bazında groupby al. Ve hangi üründen kaçar tane alındığı bilgisine ulaşmak için quantity sum al.
#bir fatura içinde alınan ürünleri ve kaçar tane alındığı bilgisini verdi bana.
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

#ama biz satırlarda invoice, sütünlarda ürünler olsun istiyorum. bunun için gruopby aldıktan sonra pivot veya unstack() fonksiyonunu kullanmalıyız.
#unstack() ile ürün isimlerini değişken isimlerine çeviriyorum.
#iloc ile de satırlardan ve sütünlerden index 0 -5 arasını getir diyorum.
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

#eksik değerlerin yerine 0 yazması için --> fillna(0)
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

#apply fonksiyonuna satır ve sütün bilgisi verilir ve ilgili yerde gezer. ancak applymap tüm gözlemleri gezer.
# applymap fonksiyonu, 0 dan büyük bir değer gördüğünde 1 yazar, görmezse 0.
df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


#invoice-product df matrisini oluşturacak bir fonksiyon yazalım.
#ve bu fonksiyona bir özellik verelim. istediğimiz değişkene göre verileri bana getirsin.
#id false olarak ön tanımlı değerini giriyorum.
def create_invoice_product_df(dataframe, id=False):
    if id: #id=true ise StockCode göre işlemi yap.
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else: #id=false ise Descriptiona göre işlemi yap.
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

#df içinde herhangi bir id nın ismini öğrenmek için bu fonksiyonu kullan.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################
#apriori fonskiyonu ile "olası tüm ürün birlikteliklerinin support değerlerini (olasılıklarını)" bulalım.
frequent_itemsets = apriori(fr_inv_pro_df,      #df verilir
                            min_support=0.01,   #çalışmanın başında tanımladığın bir eşik değer varsa bunu gir.
                            use_colnames=True)  #girdiğin df deki değişkenleirn isimlerini kullanmak istiyorsan colnames True girilir.
#supporta göre büyükten küçüğe sırala, ascensing=false --> azalan
frequent_itemsets.sort_values("support", ascending=False)

#antecedents = önceki ürün
#consequents = ikinci ürün
#antecedent support= ilk ürünün tek başına gözlemlenme olasılığı
#consequent support= ikinci ürünün tek başına gözlemlenme olasılığı
#support = antecedents ve consequents ürünlerinin birlikte gözükme olasılığı
#confidence = x ürünü alındığında y nin alınma olasılığını verir.
#lift = x ürünü alındığında y ürünün alınma olasılığı z kat artar.
#leverage = kaldıraç etkisi demektir. leverage supportu yüksek olanlara öncelik verir. bundan dolayı ufak bir yanlılığı vardır.
#conviction = y ürünü olmadan x ürününün beklenen değeri, frekansıdır. veya x ürünü olmadan y nin frekansı.
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
#birkaç tane olası kombinasyonlar üzerinde değerlendirmeler yapılır. birden fazla koşul olduğunda köşeli parantez içine girilir --> rules[...&...&...]
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
#her şey tamma ama ürünlerin adını göremiyorsun bu nedenle check_id fonksiyonunu çağır.
check_id(df_fr, 21086)


#confidenca göre azalan bir şeklide sırala. (iki ürün birlikte alındığında 3.ürünün olasılığına bak ve yorumla)
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 4. Çalışmanın Scriptini Hazırlama
##################şimdiye kadar yaptığın işlemleri bir araya getir.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

#gelen id leri sorup sonuç almak üzere bu fonksiyonu yazmıştık.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

#bu fonksiyonda ülkeyi değiştirerek yaptığın işlemleri farklı ülkelerde de gerçekleştirebilirsin.
def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

#veri ön işleme yap
df = retail_data_prep(df)
#kuralları getir.
rules = create_rules(df)
rules.head()

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

#benim için lift öenmli olduğu için lifte göre sıralıyorum.
sorted_rules = rules.sort_values("lift", ascending=False)
#sıraladığımda gidicek 22492 ürünü bulacak. bu ürünün indexinde karşılık gelen consequents(2.ürün) değerine bak. ve bu değere sahip olan ürünü öner


recommendation_list = []
#sorted_rules=sıralnamış kurallar
#enumerate methodu --> bütün satırları product gezecek.
# aynı zamanda index bilgileirni de gezmek istiyorum. onu da i gezer.
for i, product in enumerate(sorted_rules["antecedents"]):
   #ürünler ikili üçlü vs birlikte olduğundan dolayı buralarda gezebilmek için önce bunları liste çevirmek lazım
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22326)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)  #1 tane ürün tavsiyesinde bulun
arl_recommender(rules, 22492, 2)  #2 tane ürün tavsiyesinde bulun
arl_recommender(rules, 22492, 3)  #3 tane ürün tavsiyesinde bulun