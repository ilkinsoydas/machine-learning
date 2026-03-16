import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")
print(df.head(3))

y = df["target"]
x = df.drop(columns=["target"])

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42, test_size=0.7)
dt = DecisionTreeClassifier()
model = dt.fit(x_train, y_train)
print(model.score(x,y))

dt = DecisionTreeClassifier()
model = dt.fit(x_train, y_train)
print(model.score(x_test, y_test))

rf = RandomForestClassifier(n_estimators=200 )
model = rf.fit(x_train,y_train)
print(model.score(x_test,y_test))

rf = XGBClassifier()
model = rf.fit(x_train,y_train)
print(model.score(x_test,y_test))

print(df.shape) 

insan = df.sample().drop("target", axis=1).values
print(model.predict(insan))





