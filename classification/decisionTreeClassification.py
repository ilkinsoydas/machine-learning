import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Heart Attack Data Set.csv')
print(df.head(3))

y = df['target']
x = df.drop("target", axis = 1)

tree = DecisionTreeClassifier()
model = tree.fit(x,y)

print(model.score(x, y))
# train test split ile

x_train,x_test, y_train, y_test = train_test_split(x,y, random_state=16, train_size=0.7 )

tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)
print(model.score(x_test, y_test))

#train test split yokkken 1 çıkmıştı, kullandıktan sonra 0.80.. çıktı, yani model aşırı öğrenmeye maruz kalmış çıkarımını yapabiliriz

denemex = [[31,1,2,130,240,0,0,150,0,2,0,0,2]]
deneme = pd.DataFrame(denemex, columns=x.columns)
prediction = model.predict(deneme)
print(prediction)

dot = export_graphviz(model, feature_names=x.columns, filled=True, rounded=True, special_characters=True)
gorsel = graphviz.Source(dot)
gorsel.render("tree", format="png", view=True)




