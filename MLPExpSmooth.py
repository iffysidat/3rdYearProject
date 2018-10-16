import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
df = addTrainingLables('^GSPC (6).csv')

#Remove date column
dataWithoutDate = np.delete(np.array(df), 0, 1)

#Define X set which is the data without the training labels
X = np.array(np.delete(dataWithoutDate,6,1), dtype=np.float64)

#Define training labels separately
labels = np.array(dataWithoutDate.take(6, 1),dtype=np.float64)

#Train test split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

print(X_train)
#Preprocess and fit data using exponential smoothing
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
model = SimpleExpSmoothing(np.asarray(X_train['Close'])).fit()

print(X_train)


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


