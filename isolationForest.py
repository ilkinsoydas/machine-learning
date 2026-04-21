import numpy as np
import pandas as pd
import random

df = pd.DataFrame()
df["sensor"] = [4, 3, 4, 5,6,11,2,3,5]

deger = random.choice(range(df["sensor"].min(), df["sensor"].max()))
print(deger)
alt = []
ust = []

for i in df["sensor"]:
    if i<deger:
        alt.append(i)
    elif i>deger:
        ust.append(i)
        

print("alt:", alt)
print("ust:", ust)











