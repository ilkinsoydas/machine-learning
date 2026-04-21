import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

pd.set_option("display.max_columns", None)

df = pd.read_csv("../turbinee.csv")
df = df.drop(columns="timestamp")
print(df.head(3))

vc = df["is_anomaly"].value_counts()
plt.pie(vc.values, autopct = "%1.1f%%")
plt.show()

x = df.drop(columns="is_anomaly")
y = df["is_anomaly"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.88, random_state=42, stratify=y)
iso = IsolationForest(
    contamination=0.021,
    n_estimators=1000
    )
model = iso.fit(x_train)

tahmin = pd.DataFrame()
tahmin["y_pred"] = model.predict(x_test)
tahmin["y_pred"] = np.where(tahmin["y_pred"] == 1, 0, 1)
print(tahmin["y_pred"].value_counts())

tahmin["y"] = y_test.values
print(tahmin)

print(f1_score(tahmin["y"], tahmin["y_pred"]))
print(precision_score(tahmin["y"], tahmin["y_pred"]))
print(recall_score(tahmin["y"], tahmin["y_pred"]))











