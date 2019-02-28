#Scratch file to build a function that will convert time series data in data that can be used for supervised learning
import numpy
import pandas
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
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

series = read_csv('../Data/S&P5YearsCLoseNoDate.csv')
# summarize first few rows
print(series)
values = series.values
data = series_to_supervised(values, 2, 1)
data.rename(columns={"var1(t)": "Close"}, inplace=True)
#, "var1(t-1)": "Previous Close", "var2(t)": "High", "var2(t-1)": "Previous High",
#                     "var3(t)": "Low", "var3(t-1)": "Previous Low", "var4(t)": "Volume", "var4(t-1)": "Previous Volume"}
#            , inplace=True)
print(data)
print(list(data))

train_size = int(len(data) * 0.8)

train, test = data[0:train_size], data[train_size:len(data)]

print("TRAIN")
print(train)
print("TEST")
print(test)
print(len(train))
print(len(test))

#train.plot(subplots=True)
#pyplot.show()

closeTrain = train["Close"]
closeTest = test["Close"]

train = train.iloc[]

#-----------------------------------------------------------------------------------------------------------------------
#  MLP PART (REGRESSION)
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.transform(train)
# X_test = scalar.transform(test)
#
# mlp = MLPRegressor(hidden_layer_sizes=(3, 4, 5), max_iter=200000)
# mlp.fit(X_train, closeTrain)
#
# y_predict = mlp.predict(X_test)
#
# print("Y_PREDICT")
# print(y_predict)
# print("CLOSETEST")
# print(closeTest)
#
# print("RMSE")
# rms = sqrt(mean_squared_error(closeTest, y_predict))
# print(rms)

#-----------------------------------------------------------------------------------------------------------------------
#  MLP PART (Classifier)
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.transform(train)
# X_test = scalar.transform(test)
#

#-----------------------------------------------------------------------------------------------------------------------
# SVM PART (REGRESSION)
#
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.transform(train)
# X_test = scalar.transform(test)
#
# svr = SVR(kernel="linear")
# svr.fit(X_train, closeTrain)
#
# y_predict = svr.predict(X_test)
#
# print("Y_PREDICT")
# print(y_predict)
# print("CLOSETEST")
# print(closeTest)
#
# print("RMSE")
# rms = sqrt(mean_squared_error(closeTest, y_predict))
# print(rms)
#-----------------------------------------------------------------------------------------------------------------------
# Decision Tree Part (REGRESSION)
#
# scalar = MinMaxScaler()
# scalar.fit(train)
#
# X_train = scalar.transform(train)
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



