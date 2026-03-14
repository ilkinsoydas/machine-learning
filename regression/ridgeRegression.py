#Ridge Regression overfitting durumları için kullanılır
#Ridge regression sayesinde bias ve varyans arasındaki dengeyi sağlayabiliriz
#Ridge regressionda katsayılar üzerinde regresyon yapılıyor
#katsayılar küçülür ama sıfır olmaz, features öz nitelik azalmaz
#diğer adı l2 

#y = a1*x1 +a2*x2 + .... + b alfa*(katsayılarToplamı)**2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv('student_scores.csv')
print(df.head(3))

y = df['Scores']
x = df['Hours']

plt.style.use('fivethirtyeight') #FiveThirtyEight veri gazeteciliği sitesinin kullandığı grafik stilidir.
plt.figure(figsize=(14, 14)) #grafiğin boyutu.
plt.scatter(x,y) #scatter plot (dağılım grafiği) çizer.
plt.show()

lr = LinearRegression()
model = lr.fit(x.values.reshape(-1,1), y) #-1  → satır sayısını otomatik hesapla, 1   → 1 feature var

print(model.score(x.values.reshape(-1,1), y)) #modelin doğruluk skorunu verir

alfalar = [1,10,20,100,200] #alfa arttıkça katsayılar azaldı
for a in alfalar: 
    r = Ridge(alpha=a) #alpha değeri, Ridge Regression'ın ceza teriminin gücünü belirler. Daha yüksek alpha değerleri, katsayıların daha fazla küçülmesine neden olur.
    modelr = r.fit(x.values.reshape(-1,1), y)
    skor = modelr.score(x.values.reshape(-1,1), y)
    print("Skor", skor)
    print("Katsayı", modelr.coef_) #modelin katsayılarını verir
