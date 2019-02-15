# load and plot dataset
import numpy
from math import sqrt
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
# load dataset
def parser(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d')

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.drop(0)
    return df

def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def fit_model(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
    return model

series = read_csv('../Data/S&P5YearsCLose.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())
# line plot
series.plot()
pyplot.show()

#from statsmodels.tsa.stattools import adfuller
#X = series.values
#result = adfuller(X)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#    print('\t%s: %.3f' % (key, value))

lag = 1

raw_values = series.values
print("RAW")
print(raw_values)
diff_values = difference(raw_values, 1)
print("DIFFERENCE")
print(diff_values)
supervised = timeseries_to_supervised(diff_values, lag)
print("SUPERVISED")
print(supervised)

supervised_values = supervised.values[lag:, :]
print("SUPERVISED VALUES")
print(supervised_values)

#from statsmodels.tsa.stattools import adfuller
#X = supervised_values[:, 0]
#result = adfuller(X)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#    print('\t%s: %.3f' % (key, value))

supervised.plot()
pyplot.show()

train, test = supervised_values[0:-280], supervised_values[-280:]
print("TRAIN")
print(train)
print("TEST")
print(test)

scaler, train_scaled, test_scaled = scale(train, test)

print("TRAIN SCALED")
print(train_scaled)
print("TEST SCALED")
print(test_scaled)

batch_size = 4
train_trimmed = train_scaled[2:, :]
model = fit_model(train_trimmed, batch_size, 2000, 1)

test_reshaped = test_scaled[:, 0:-1]

output = model.predict(test_reshaped, batch_size=batch_size)

predictions = list()
for i in range(len(output)):
    yhat = output[i, 0]
    X = test_scaled[i, 0:-1]
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    predictions.append(yhat)
print("PREDICITIONS")
print(predictions)


rmse = sqrt(mean_squared_error(raw_values[-280:], predictions))
print('%d) Test RMSE: %.3f' % (1, rmse))