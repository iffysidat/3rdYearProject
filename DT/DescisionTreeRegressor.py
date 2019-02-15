import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree


#Function to read data
def addTrainingLables(filename):
    data = pd.read_csv(filename)
    labels = []
    for i in range(0, len(data) - 1):
        labels.append(0 if (data["Close"].values[i] > data["Close"].values[i+1]) else 1)
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)
    data["Label"] = trainingLables
    data = data[:-1]
    return data

# Get data and add training labels
df = pd.read_csv('../Data/AAPL.csv')

# Remove date column
dataWithoutDate = np.delete(np.array(df), 0, 1)
print("Data without date")
print(dataWithoutDate)

actualValues = np.array(dataWithoutDate.take(3, 1), dtype=np.float64)
print("ACTUAL VALUES")
print(actualValues)

X = np.array(np.delete(dataWithoutDate, 3, 1), dtype=np.float64)
print("X")
print(X)

# Train test split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, actualValues, test_size=0.2)
# split = int(0.8 * len(X))
#
# X_train = X[:split]
# X_test = X[split:]
# y_train = actualValues[:split]
# y_test = actualValues[split:]
#
#Preprocess and fit data using scalar
scalar = MinMaxScaler()
scalar.fit(X_train)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print("Y_PREDICT")
print(y_predict)
print("Y_TEST")
print(y_test)

print("RMSE")
rms = sqrt(mean_squared_error(y_test, y_predict))
print(rms)

import graphviz
dot_data = tree.export_graphviz(clf, out_file='tree.dot', feature_names=['Open', 'High', 'Low', 'Adj Close', 'Volume'],
                                filled=True, rounded=True, special_characters=True)
graphviz.Source(dot_data)

