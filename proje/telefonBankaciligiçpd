import pandas as pd 
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



df = pd.read_csv('banka.csv')
df = df[['metin', 'kategori']]

trStopWords = stopwords.words('turkish')
cv = CountVectorizer(stop_words=trStopWords, max_features=1000) ##örneğin 'yapmamalı' daki ama yı da almasın diye
# bu olmasaydı:
# for word in trStopWords:
    #word = " " + word + " "  
    #df['sorgu'] = df['sorgu'].str.replace(word, " ")
    
x = cv.fit_transform(df['metin']).toarray()
y = df['kategori']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, train_size=0.7)

rf = RandomForestClassifier()
model = rf.fit(x_train,y_train)

mesaj = input("Lütfen mesajınızı giriniz: ")
tahminVektoru = cv.transform([mesaj]).toarray()

sonuc = model.predict(tahminVektoru)
skor = model.score(x_test,y_test)

print("Tahmin edilen kategori: ", sonuc[0])
print("Modelin doğruluk skoru: ", skor)
