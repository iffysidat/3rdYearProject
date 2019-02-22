#Scratch file to build a function that will convert time series data in data that can be used for supervised learning
import numpy
import pandas
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
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

series = read_csv('../Data/S&P5YearsCLoseNODate.csv')
# summarize first few rows
print(series)
values = series.values
data = series_to_supervised(values, 1, 1)
data.rename(columns={"var1(t)": "Close", "var1(t-1)": "Previous Close", "var2(t)": "High", "var2(t-1)": "Previous High",
                     "var3(t)": "Low", "var3(t-1)": "Previous Low", "var4(t)": "Volume", "var4(t-1)": "Previous Volume"}
            , inplace=True)
print(data)
print(list(data))

train_size = int(len(data) * 0.8)

train, test = data[0:train_size], data[train_size:len(data)]

print(len(train))
print(len(test))

train.plot(subplots=True)
pyplot.show()

print("TRAIN")
#print(train)