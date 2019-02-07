import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

def addTrainingLables(filename):
    data = pd.read_csv(filename)
    labels = []
    for i in range(0, len(data) - 1):
        labels.append(0 if (data["Close"].values[i] > data["Close"].values[i+1]) else 1)
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)
    data["Label"] = trainingLables
    data = data[:-1]
    return data


df = addTrainingLables('../Data/S&P5Years.csv')
df.set_index('Date', inplace=True)


X = df.drop('Label', axis=1)
y = df['Label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# split = int(0.8 * len(X))
#
# X_train = X[:split]
# X_test = X[split:]
# y_train = y[:split]
# y_test = y[split:]

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit_transform(X_train)

scalar.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Data fit now predicting")

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import graphviz
dot_data = tree.export_graphviz(clf, out_file='tree.dot', feature_names=X.columns.values, class_names=['0', '1'],
                                filled=True, rounded=True, special_characters=True)
graphviz.Source(dot_data)
