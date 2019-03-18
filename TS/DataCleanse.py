#Scratch file to build a function that will convert time series data in data that can be used for supervised learning
import numpy
import pandas
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from ta import *
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from math import sqrt
desired_width=320

pandas.set_option('display.width', desired_width)

numpy.set_printoptions(linewidth=desired_width)

pandas.set_option('display.max_columns',100)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def parser(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d')

def addTrainingLables(data, shift):
    labels = []
    for i in range(0, len(data) - 1):
        labels.append(0 if (data.iloc[i, 0] > data.iloc[i, -1]) else 1)
    labels = shift * [666] + labels
    trainingLables = pandas.DataFrame(columns=["Label"], data=labels)
    data["Label"] = trainingLables
    data = data[:-1]
    return data

series = read_csv('../Data/S&P15YearsCLoseNoDate.csv')
# summarize first few rows
#print(series)


series = utils.dropna(series)

series['ATR'] = average_true_range(series["High"], series["Low"], series["Close"], n=14, fillna=False)
# series['MACD'] = macd(series['Close'], n_fast=12, n_slow=26, fillna=False)
# series['MFI'] = money_flow_index(series['High'], series['Low'], series['Close'], series['Volume'], n=14, fillna=False)
series = utils.dropna(series)
print(series)
values = series.values

shift_days = 10
data = series_to_supervised(values, shift_days, 1)
data.rename(columns={"var1(t)": "Close"}, inplace=True)

data = data.loc[:, :"Close"]

#-----------------------------------------------------------------------------------------------------------------------
# ONLY USE WHEN CLASSIFYING
# data = addTrainingLables(data, shift_days)
#-----------------------------------------------------------------------------------------------------------------------

train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]

#train.plot(subplots=True)
#pyplot.show()

closeTrain = train["Close"]
closeTest = test["Close"]

train = train.drop(["Close"], axis=1)
test = test.drop(["Close"], axis=1)

#-----------------------------------------------------------------------------------------------------------------------
# ONLY USE WHEN CLASSIFYING
# trainLabels = train["Label"]
# testLabels = test["Label"]
#
# train = train.drop(["Label"], axis=1)
# test = test.drop(["Label"], axis=1)
# #-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#  MLP PART (REGRESSION)
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.fit_transform(train)
# X_test = scalar.transform(test)
#
# mlp = MLPRegressor(max_iter=20000)
#
# parameter_space = {
#     'hidden_layer_sizes': [(6, 7, 8)],
#     'activation': ['relu'],
#     'solver': ['adam'],
#     'alpha': [0.05],
#     'learning_rate': ['constant'],
# }
#
# clf = GridSearchCV(mlp, parameter_space, cv=3)
# clf.fit(X_train, closeTrain)
# clf = clf.best_estimator_
#
# y_predict = clf.predict(X_test)
#
# print("Y_PREDICT")
# print(y_predict)
# print("CLOSETEST")
# print(closeTest)
#
#
# print('Best parameters found:\n', clf)
# print("RMSE")
# rms = sqrt(mean_squared_error(closeTest, y_predict))
# print(rms)
#
# # import pickle
# # with open("mlpdata.pkl", "wb") as f:
# #     pickle.dump((y_predict, closeTest), f)
# plt.plot(range(len(closeTest)), closeTest, c="c", label="Raw Data")
# plt.plot(range(len(closeTest)), y_predict, c="m", label="Predictions")
# plt.xlabel("Extrapolation Length (Days)")
# plt.ylabel("Stock Price (USD)")
# plt.legend()
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#  MLP PART (Classifier)

# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.fit_transform(train)
# X_test = scalar.transform(test)
#
# mlp = MLPClassifier(max_iter=20000)
#
# parameter_space = {
#     'hidden_layer_sizes': [(80, 80, 80, 80)],
#     'activation': ['tanh'],
#     'solver': ['adam'],
#     'alpha': [0.05],
#     'learning_rate': ['constant'],
# }
#
# clf = GridSearchCV(mlp, parameter_space, cv=3)
# clf.fit(X_train, trainLabels)
# clf = clf.best_estimator_
#
#
# # Predict values
# y_predict = clf.predict(X_test)
#
# # print("PREDCITED")
# # print(y_predict)
# # print("TESTLABELS")
# # print(testLabels)
#
# # Print Classification report nd confusion matrix
# from sklearn.metrics import classification_report, confusion_matrix
# print('Best parameters found:\n', clf)
# print(classification_report(testLabels, y_predict))
# print(confusion_matrix(testLabels, y_predict))
# print("PERCENTAGE ACCURACY")
# print(accuracy_score(testLabels, y_predict) * 100)

#-----------------------------------------------------------------------------------------------------------------------
# SVM PART (REGRESSION)

scalar = MinMaxScaler()
scalar.fit(train)

X_train = scalar.fit_transform(train)
X_test = scalar.transform(test)

svr = SVR()

parameter_space = {
    'C': [100],
    'kernel': ['linear']
}
clf = GridSearchCV(svr, parameter_space, cv=3)
clf.fit(X_train, closeTrain)

y_predict = clf.predict(X_test)

print("Y_PREDICT")
print(y_predict)
print("CLOSETEST")
print(closeTest)

print('Best parameters found:\n', clf.best_params_)
print("RMSE")
rms = sqrt(mean_squared_error(closeTest, y_predict))
print(rms)
# #
# plt.plot(range(len(closeTest)), closeTest, c="c", label="Raw Data")
# plt.plot(range(len(closeTest)), y_predict, c="m", label="Predictions")
# plt.xlabel("Extrapolation Length (Days)")
# plt.ylabel("Stock Price (USD)")
# plt.legend()
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#  SVM PART (Classifier)
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.fit_transform(train)
# X_test = scalar.transform(test)
#
# svc = SVC(kernel="linear")
#
# parameter_space = {
#     'C': [100],
#     'kernel': ['linear']
# }
# clf = GridSearchCV(svc, parameter_space, cv=3)
# clf.fit(X_train, trainLabels)
# clf = clf.best_estimator_
#
#
# # Predict values
# y_predict = clf.predict(X_test)
#
# print("PREDCITED")
# print(y_predict)
# print("TESTLABELS")
# print(testLabels)
#
# # Print Classification report nd confusion matrix
# from sklearn.metrics import classification_report, confusion_matrix
# print('Best parameters found:\n', clf)
# print(classification_report(testLabels, y_predict))
# print(confusion_matrix(testLabels, y_predict))
# print("PERCENTAGE ACCURACY")
# print(accuracy_score(testLabels, y_predict) * 100)

#-----------------------------------------------------------------------------------------------------------------------
# Decision Tree Part (REGRESSION)

# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.fit_transform(train)
# X_test = scalar.transform(test)
#
# clf = DecisionTreeRegressor()
# clf.fit(X_train, closeTrain)
#
# y_predict = clf.predict(X_test)
#
# print("Y_PREDICT")
# print(y_predict)
# print("CLOSETEST")
# print(closeTest)
#
# print("RMSE")
# rms = sqrt(mean_squared_error(closeTest, y_predict))
# print(rms)
#
# plt.plot(range(len(closeTest)), closeTest, c="c", label="Raw Data")
# plt.plot(range(len(closeTest)), y_predict, c="m", label="Predictions")
# plt.xlabel("Extrapolation Length (Days)")
# plt.ylabel("Stock Price (USD)")
# plt.legend()
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#  Decision Tree PART (Classifier)
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.fit_transform(train)
# X_test = scalar.transform(test)
#
# dtc = DecisionTreeClassifier()
#
# parameter_space = {
#     'max_depth': [None],
#     'min_samples_split': [2],
# }
# clf = GridSearchCV(dtc, parameter_space, cv=3)
# clf.fit(train, trainLabels)
# clf = clf.best_estimator_
#
#
# # Predict values
# y_predict = clf.predict(test)
#
# print("PREDCITED")
# print(y_predict)
# print("TESTLABELS")
# print(testLabels)
#
# # Print Classification report nd confusion matrix
# from sklearn.metrics import classification_report, confusion_matrix
# print(classification_report(testLabels, y_predict))
# print(confusion_matrix(testLabels, y_predict))
# print("PERCENTAGE ACCURACY")
# print(accuracy_score(testLabels, y_predict) * 100)
#
# from sklearn import tree
#
# dot_data = tree.export_graphviz(clf, out_file='newtree.dot', feature_names=train.columns.values, class_names=['0', '1'],
#                                 filled=True, rounded=True, special_characters=True)


