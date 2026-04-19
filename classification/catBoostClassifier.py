import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)

df = pd.read_csv("hospital.csv")
df = df.drop(columns="patient_id")

df = df.fillna("Unknown") #Tüm veri setindeki NaN (boş) değerleri "Unknown" metni ile dolduruyoruz.
# Artık CatBoost float yerine bir metin göreceği için hata vermeyecek.

le = LabelEncoder()
df["readmission_risk"] = le.fit_transform(df["readmission_risk"])

y = df["readmission_risk"]
x = df.drop(columns="readmission_risk")

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=42 )

categorical_features = x.select_dtypes(include = ["object", "string"]).columns.tolist() #pandas'a bana sadece veri tipi 'object' olan sütunları getir diyoruz

cb = CatBoostClassifier(
    learning_rate = 0.1,
    loss_function = 'MultiClass',
    n_estimators = 1000,
    depth=6
)

model = cb.fit(x_train, y_train, cat_features=categorical_features)
sc = model.score(x_test, y_test)
print(sc)


