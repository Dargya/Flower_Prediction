from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.6)

ran_clf = RandomForestClassifier(n_estimators=10)

ran_clf.fit(X_train, y_train)

predictions = ran_clf.predict(X_test)

print(accuracy_score(predictions, y_test))


# Picking the model

import pickle
with open('ranfor.pkl','wb') as model_pkl:
    pickle.dump(ran_clf,model_pkl)
