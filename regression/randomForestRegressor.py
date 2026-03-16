import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("../personel.csv")
print(df.head(3))

df = pd.get_dummies(df, columns=["sex","smoker", "region"], drop_first=True)

y = df["charges"]
x = df.drop(columns=["charges"])

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=22, test_size=0.7)
lr = LinearRegression()
model = lr.fit(x_train,y_train)
print(model.score(x_test,y_test))

rf = RandomForestRegressor(n_estimators=200) #kaç ağaç kullanılmalı?
model = rf.fit(x_train, y_train)
print(model.score(x_test, y_test))

denemex = [[31, 20.2, 0, 0, 0, 1, 0, 1]]
deneme = pd.DataFrame(denemex, columns=x.columns)
prediction = model.predict(deneme)
print(prediction)














