import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

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
df = addTrainingLables('../Data/AAPL.csv')
df.set_index('Date', inplace=True)
print(df.tail())
#Remove date column
#dataWithoutDate = np.delete(np.array(df), 0, 1)

df = df[['Close']]

forecast_out = int(30)
df['Prediction'] = df[['Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
split = int(0.8 * len(X))

X_train = X[:split]
print(X_train)
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print("Confidence:", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)


# #Define X set which is the data without the training labels
# X = np.array(np.delete(df,5,1), dtype=np.float64)
#
# #Define training labels separately
# labels = np.array(df.take(5, 1),dtype=np.float64)
#
# #Train test split data 80/20
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
#
# #Preprocess and fit data using scalar
# scalar = MinMaxScaler()
# scalar.fit_transform(X_train)
#
# #Preprocessing step
#
# X_test = scalar.fit_transform(X_test)
#
# #Instantiaate classifier and fit data to model
# mlp = MLPClassifier(max_iter=20000)
# mlp.fit(X_train, y_train)
#
# #Predict values
# y_predict = mlp.predict(X_test)
#
# #Print Classification report nd confusion matrix
# from sklearn.metrics import classification_report,confusion_matrix
# print(classification_report(y_test, y_predict))
# print(confusion_matrix(y_test, y_predict))


