import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.ensemble import RandomForestClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../customer.csv', encoding="latin-1")
print(df.head())

df = pd.get_dummies(df, columns=["sales_channel", "trip_type", "flight_day", "route", "booking_origin"],drop_first=True)
y = df["wants_extra_baggage"]
x = df.drop(columns=["wants_extra_baggage"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.85, random_state=42)

rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)
print(model.score(x_test, y_test))

crossval = cross_val_score(model, x, y, cv = 4)
crossval.mean()
print(crossval)




