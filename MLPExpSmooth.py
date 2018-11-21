import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Read in data
# Remove Date Column
# Add Training labels column
# Numpy array those babies and stick them in X/y variables
# Train test split
# Preprocess data using scalar
# Train data using fit
# Predict test data and analyse

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

#Get data and add training labels
df = addTrainingLables('^GSPC (6).csv')
print(list(df))

train = df[0:16803]
test = df[16803:]

df.Timestamp = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df.Timestamp

train.Timestamp = pd.to_datetime(train.Date, format='%Y-%m-%d')
train.index = train.Timestamp

test.Timestamp = pd.to_datetime(test.Date, format='%Y-%m-%d')
test.index = test.Timestamp

# Visualise the data
train.Close.plot(figsize=(15, 8), title= 'Close prices train', fontsize=14)
test.Close.plot(figsize=(15,8), title= 'Close prices test', fontsize=14)
plt.show()

#1 Implement naive forecast
dd = np.asarray(train.Close)

from sklearn.metrics import mean_squared_error
from math import sqrt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Close'])).fit(smoothing_level=0.1, optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Close, y_hat_avg.SES))
print("SES RMSE", rms)

# import statsmodels.api as sm
# sm.tsa.seasonal_decompose(train.Close, freq=1000 ).plot()
# result = sm.tsa.stattools.adfuller(train.Close)
# plt.show()
#
# y_hat_avg = test.copy()
# fit1 = Holt(np.asarray(train['Close'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
# y_hat_avg['Holt_Linear'] = fit1.forecast(len(test))
# plt.figure(figsize=(16,8))
# plt.plot(train['Close'], label='Train')
# plt.plot(test['Close'], label='Test')
# plt.plot(y_hat_avg['Holt_Linear'], label='Holt_Linear')
# plt.legend(loc='best')
# plt.show()
#
# rms = sqrt(mean_squared_error(test.Close, y_hat_avg.Holt_Linear))
# print("Holt Linear RMSE", rms)
#
# y_hat_avg = test.copy()
# fit1 = ExponentialSmoothing(np.asarray(train['Close']), seasonal_periods=7, trend='add', seasonal='add').fit()
# y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
# plt.figure(figsize=(16,8))
# plt.plot(train['Close'], label='Train')
# plt.plot(test['Close'], label='Test')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
# plt.legend(loc='best')
# plt.show()
#
# rms = sqrt(mean_squared_error(test.Close, y_hat_avg.Holt_Winter))
# print("Holt Winter RMSE", rms)
#
# print(train.head)
#
# y_hat_avg = test.copy()
# fit1 = sm.tsa.statespace.SARIMAX(train.Close, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
# y_hat_avg['SARIMA'] = fit1.predict(dynamic=True)
# plt.figure(figsize=(16,8))
# plt.plot(train['Close'], label='Train')
# plt.plot(test['Close'], label='Test')
# plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
# plt.legend(loc='best')
# plt.show()
#
# print(y_hat_avg['SARIMA'])
# rms = sqrt(mean_squared_error(test.Close, y_hat_avg.SARIMA))
# print("SARIMA", rms)

# #Remove date column
# dataWithoutDate = np.delete(np.array(df), 0, 1)
#
# #Define X set which is the data without the training labels
# X = np.array(np.delete(dataWithoutDate,6,1), dtype=np.float64)
#
# #Define training labels separately
# labels = np.array(dataWithoutDate.take(6, 1),dtype=np.float64)
#
# #Train test split data 80/20
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
#
# print(X_train)
# #Preprocess and fit data using exponential smoothing
# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
# model = SimpleExpSmoothing(np.asarray(X_train['Close'])).fit()
#
# print(X_train)


#Preprocessing step
#X_train = scalar.transform(X_train)
#X_test = scalar.transform(X_test)

#Instantiaate classifier and fit data to model
#mlp = MLPClassifier(max_iter=20000)
#mlp.fit(X_train, y_train)

#Predict values
#y_predict = mlp.predict(X_test)

#Print Classification report nd confusion matrix
#from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(y_test, y_predict))
#print(confusion_matrix(y_test, y_predict))


