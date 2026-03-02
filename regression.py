from sklearn.linear_model import LinearRegression
import pandas as pd

#y=ax+b

data = pd.read_csv("Student_Marks.csv")
print(data)

y = data[["Marks"]] #tahmin etmek istiyoruz
x = data[["number_courses", "time_study"]]

data.info() #data hakkında bilgi verir

l = LinearRegression() #yukarıda sadece bunu çağırmış olmasaydık daha uzun yazmamız gerekirdi
model = l.fit(x,y) 
a = model.predict([[4,4]])
print(a)

print(data[["Marks"]].max()) #en yüksek notu verir

print(model.score(x,y)) #modelin doğruluk oranını verir
