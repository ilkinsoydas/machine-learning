import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("/Users/ilkinsoydas/Desktop/projeler/vscode/Python/Audi_A1_listings.csv")

print(df.head(3))

df = df[["Year", "Type", "Mileage(miles)","Engine", "PS", "Transmission", "Fuel", "Number_of_Owners", "Price(£)"]]

df.columns = ["yil", "kasa", "mil",  "motor" ,"ps", "vites", "yakit", "sahip", "fiyat"]

df["motor"] = df["motor"].astype(str).str.replace("L", "")
df["motor"] = pd.to_numeric(df["motor"], errors="coerce") #eğer motor sütununda "L" harfi dışında bir şey varsa onu NaN yapar

df = pd.get_dummies(df, columns=["kasa", "vites", "yakit"], drop_first=True)

y = df[["fiyat"]]
x = df.drop("fiyat", axis=1)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.70, random_state=22 ) #verinin %70ini öğrenmek için, %30unu test etmek icin kullan demek,22 seed numarası, veriyi her böldüğümüzde aynı satırların ayrılmasını sağlar

lm = LinearRegression()
model = lm.fit(x_train, y_train)
print(model.score(x_test, y_test))


#lm = LinearRegression()
#model = lm.fit(x.values, y)
#print(model)
#print(model.score(x.values, y))

print(df.head(3))

print(model.predict([[2016, 3000, 1.0, 90, 5, 0, 1]]))