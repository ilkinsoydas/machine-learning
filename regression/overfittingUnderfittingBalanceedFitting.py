#UnderFitting: Az öğrenme
#OverFitting: Aşırı öğrenme
#BalancedFitting: Dengeli öğrenme

import seaborn as sns #seaborn veri görselleştirme kütüphanesidir. Hazır veri setleri vardır
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = sns.load_dataset('diamonds')

df = pd.get_dummies(df, columns = ['cut', 'color', 'clarity'], drop_first=True)
print(df.head(3))

y = df['price']
x = df.drop('price', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.76, random_state = 19)

lm = LinearRegression()
model = lm.fit(x_train, y_train)

#ikisi arasındaki fark ne kadar azsa o kadar uyumlu öğrenmştir, ne kadar fazlaysa o kadar overfitting
print(model.score(x_test, y_test))
print(model.score(x_train, y_train))
