from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("Student_Marks.csv")

print(df.head(3))

df.columns = ['sinif', 'saat', 'puan']
print(df.head(3))

y = df[['puan']]
x = df[['sinif', 'saat']].values #values ile ->  Ben sadece 1. sütun ve 2. sütunla eğitiliyorum, isimler umurumda değil

lm = LinearRegression()
model = lm.fit(x,y) #veriden bir şeyler öğrenme özelliği

#y = ax + b

model.coef_ #herbir sayının katsayısı
model.intercept_ #sabit

print(model.predict([[4,5]])) #öngörü ile veriler üzerinden sonuç üretme


