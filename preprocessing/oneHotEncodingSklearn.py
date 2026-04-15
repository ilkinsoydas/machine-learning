import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../churn.csv')

df.drop(columns=["RowNumber","Surname", "CustomerId"], inplace=True)

ohe = OneHotEncoder()
xd = ohe.fit_transform(df[["Geography", "Gender"]]).toarray()
ohe.get_feature_names_out()

xd = pd.DataFrame(xd) #sayı yığınını bir tabloya çevirir
xd.columns = ohe.get_feature_names_out() #Hangi sütuna ne isim verdin

print(df.head(3))
df = df.drop(columns=["Geography", "Gender"])

df[xd.columns] = xd 
print(df.head(3))

y = df["Exited"]
x = df.drop(columns=["Exited"])

tree = DecisionTreeClassifier()
model = tree.fit(x,y)
print(model.score(x,y)) #overfitting



