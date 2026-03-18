#dummy bağımlı değişkenlerde kullanılması zor

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("tour.csv")
print(df.head(3))

le = LabelEncoder()
le.fit(df["Team"])

x = le.transform(df['Team'])
print(x)

print(le.classes_)

print(le.inverse_transform(x))

le2 = LabelEncoder()
print(le2.fit_transform(df['Team']))
