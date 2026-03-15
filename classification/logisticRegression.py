import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #veri grafiklerini çizmeye yarar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('../UCI_Credit_Card.csv')
print(df.head(3))

df.drop("ID", axis=1)
y = df["default.payment.next.month"]
x = df.drop("default.payment.next.month", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.77, random_state=6)

log = LogisticRegression()   
model = log.fit(x_train, y_train)

print(model.score(x_test, y_test))

denemex = np.array(x.iloc[1903])

deneme = pd.DataFrame([denemex], columns = x.columns)
print(model.predict(deneme))

print(y.iloc[1903])