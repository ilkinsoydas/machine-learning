import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

df = pd.read_csv("insurance.csv")  
print(df.head(3))

df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print(df.head(3))

y = df[['charges']]
x = df.drop('charges', axis=1) 

lm = LinearRegression()
model = lm.fit(x.values, y.values) #formülü bulur

print(model.predict([[19, 26, 0, 1, 1, 0, 0, 1]]))

df_hata = pd.DataFrame()

df_hata['y'] = y

y_tahmin = model.predict(x.values) #formülü uygular
print(y_tahmin)

df_hata['tahmin'] = y_tahmin

print(df_hata)

df_hata['error'] = y - y_tahmin

df_hata['squared_error'] = df_hata['error'] ** 2  #mse

df_hata['abs_error'] = np.abs(df_hata['error'])  #mae

df_hata['percent_error'] = np.abs((y - y_tahmin) / y)  #mape
print(df_hata.head(10))

print(df_hata.mean)

#son kütüphaneler kullanıldıktan sonra:

print(mean_squared_error(y, y_tahmin))

print(mean_absolute_error(y, y_tahmin))

print(mean_absolute_percentage_error(y, y_tahmin))