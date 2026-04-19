import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)

df = pd.read_csv("../card.csv")
#print(df["fraud"].unique())

y = df["fraud"]
x = df.drop(columns=["fraud"])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=43) #veri seti arttıkça train size büyümeli
log = LogisticRegression()
model = log.fit(x_train, y_train)
print(model.score(x_test, y_test))

print(df["fraud"].value_counts())

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

tree = DecisionTreeClassifier()
model_tree = tree.fit(x_train, y_train)
print(model_tree.score(x_test, y_test))

y_tree_pred = model_tree.predict(x_test)
confusion_matrix_tree = confusion_matrix(y_test, y_tree_pred)
print(confusion_matrix_tree)

a = pd.DataFrame()
a["true"] = y_test
a["pred"] = y_pred

print(a[a["true"] == 1].value_counts())














