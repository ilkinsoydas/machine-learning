import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


pd.set_option("display.max_columns", None)

df = pd.read_csv("students.csv")
df = df.drop(columns="Student_ID")

df = pd.get_dummies(df, columns=["Gender", "Department"], drop_first=True)
#print(df.head(3))

y = df["Depression"]
x = df.drop(columns="Depression")

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)

rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)
print(model.score(x_test, y_test))

df["Depression"].value_counts()
sonuc = model.predict(x_test) 
dfs = pd.DataFrame()
dfs["sonuc"] = sonuc
dfs["gercek"] = y_test.values

print(dfs["sonuc"].value_counts())
print(dfs["gercek"].value_counts())
print(dfs)

dfs["truebil"] = (dfs["gercek"] == True) & (dfs["sonuc"] == True)
print(dfs["truebil"].value_counts())

ps = precision_score(dfs["gercek"], dfs["sonuc"])
print(ps)

rs = recall_score(dfs["gercek"], dfs["sonuc"])
print(rs)

f1s = f1_score(dfs["gercek"], dfs["sonuc"])
print(f1s)


