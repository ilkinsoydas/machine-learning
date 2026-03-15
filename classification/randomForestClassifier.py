import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Heart Attack Data Set.csv')
df.head(3)

y = df["target"]
x = df.drop("target", axis=1)

tree = DecisionTreeClassifier()
model = tree.fit(x,y)
print(model.score(x,y))

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=16, train_size=0.7)
tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)
print(model.score(x_test,y_test))

denemex = [[31,1,2,130,240,0,0,150,0,2,0,0,2]]
deneme = pd.DataFrame(denemex, columns = x.columns)
prediction = model.predict(deneme)
print(prediction)

dot = export_graphviz(model, feature_names=x.columns, filled=True)
gorsel = graphviz.Source(dot)
gorsel.render("tree", format="png", view=True)


forest = RandomForestClassifier()
model = forest.fit(x_train, y_train)
print(model.score(x,y))

forest = RandomForestClassifier(n_estimators=200, max_depth=4 ) #n_estimators ağaç sayısı
print(model.score(x_test, y_test))



