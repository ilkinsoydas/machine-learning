import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame()
df['cumleler'] = ['Merhaba dünya', 'Python programlama', 'Makine öğrenimi', 'Makine']
cv = CountVectorizer(max_features=4)
a = cv.fit_transform(df['cumleler'])

x = a.toarray()
print(x)

cvGet = cv.get_feature_names_out()
print(cvGet)