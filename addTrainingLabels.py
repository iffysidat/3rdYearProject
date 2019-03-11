import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime

desired_width=320

pd.set_option('display.width', desired_width)

numpy.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',100)

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


def addTrainingLables(data, shift):
    labels: list[int] = []
    for i in range(0, len(data)):

        if data.iloc[i, 0] > data.iloc[i, -1]:
            labels.append(0)
        else:
            labels.append(1)

    labels = shift * [666] + labels
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)
    data["Label"] = trainingLables
    #data = data[:-1]
    print(labels)
    print(len(data))
    return data

series = read_csv('Data/S&P5YearsCLoseNoDate.csv')
# summarize first few rows
print(series)
values = series.values
shift_days = 10
data = series_to_supervised(values, shift_days, 1)
data.rename(columns={"var1(t)": "Close"}, inplace=True)


print(data)
data = data.loc[:, :"Close"]
data = addTrainingLables(data, shift_days)
print(data)


# fitnessMeasure = []
# layers = []
# neurons = []
# numEvents = len(y_test)
# for numLayers in np.add(range(4),2):
#     hiddenLayers = []
#     row = []
#     rowLay = []
#     rowNeur = []
#     for numNeurons in (np.add(range(20),8)):
#         hiddenLayers = [numNeurons] * numLayers
#         mlp = MLPClassifier(hidden_layer_sizes=hiddenLayers, max_iter=20000)
#         print(hiddenLayers)
#         mlp.fit(X_train, y_train)
#
#         y_predict = mlp.predict(X_test)
#         numCorrect = (np.subtract(y_test,y_predict) == 0).sum()
#         fracCorrect = numCorrect / numEvents
#         row.append(fracCorrect)
#         rowLay.append(numLayers)
#         rowNeur.append(numNeurons)
#     fitnessMeasure.append(row)
#     layers.append(rowLay)
#     neurons.append(rowNeur)
# #
# from sklearn.metrics import classification_report,confusion_matrix
# print(classification_report(y_test, y_predict))
# print(confusion_matrix(y_test, y_predict))
#
# import matplotlib.pyplot as plt
# plt.hist(np.subtract(y_predict, y_test))
# plt.contourf(neurons, layers, fitnessMeasure)#, levels = np.linspace(0,1,11))
# plt.xlabel("Number of neurons per layer")
# plt.ylabel("Number of layers")
# plt.title("MLP performance measured by fractional prediction accuracy")
# cbar = plt.colorbar();
# plt.savefig("fitnessContour.pdf")
# plt.show()
