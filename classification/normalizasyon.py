import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("../price.csv")

df.info()

df = df[["Rcmnd cruise Knots", "Stall Knots dirty", "Fuel gal/lbs", "Eng out rate of climb", "Takeoff over 50ft", "Price"]]
print(df.head(3))

y = df["Price"]
x = df.drop(columns=["Price"])

#Y için normalizasyon yapılmaz
#outlier(aykırı değer) etkisini azaltır
#model performansı artar

ss = StandardScaler()
x2 = ss.fit_transform(x)
print(x2)

x2 = pd.DataFrame(x2 )
print(x2.head(3))

print(x2[2].mean())
print(x2[2].std())

mm = MinMaxScaler()
x3 = mm.fit_transform(x)
x3 = pd.DataFrame(x3)

print(x3.head(3))
print(x3[0].max())
print(x3[0].min())

mm2 = MinMaxScaler(feature_range=(0,10))
x4 = mm2.fit_transform(x)
x4 = pd.DataFrame(x4)
print(x4.head(3))

print(x4[2].max())
print(x4[2].min())


