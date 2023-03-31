import pandas as pd
import datetime as dt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#Görev 1
#ADIM1
df_ = pd.read_csv("C:/Users/esman/PycharmProjects/recommender_system/datasets/armut_data.csv")
df = df_.copy()
df.head()

#ADIM2
df["Hizmet"] = [str(col[1]) + "_" + str(col[2]) for col in df.values]
df.head(5)


#ADIM3
df.dtypes #createdate object tipinde bunu datetime çevir.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime("%Y - %m")
df.head(5)
df["ID"] = [str(col[0]) + "_" + str(col[5]) for col in df.values]
df.head(5)

#GÖREV 2
#ADIM1
#PIVOT TABLE fonksiyonunu kullanarak yapamadım --> ?????  df.pivot_table(index = "ID", columns= "Hizmet", values="counts", aggfunc="...")
#matrix yapısını oluşturuyorum ve binary encode ile dolduuryorum.
df = df.groupby(["ID", "Hizmet"])["Hizmet"].count().  unstack(). fillna(0).applymap(lambda x: True if x > 0 else False)



#ADIM2 --> support, confidience, lift, leverage
frequent_itemsets = apriori(df,
                            min_support=0.01,
                            use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(10)
rules = association_rules(frequent_itemsets, \
                          metric="support",\
                          min_threshold=0.01)
rules.head

#ADIM3:

