import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import pandas as pd

# Read in data
# Remove Date Column
# Add Training labels column
# Numpy array those babies and stick them in X/y variables
# Train test split
# Preprocess data using scalar
# Train data using fit
# Predict test data and analyse

# Function to read data
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
df = addTrainingLables('S&P5Years.csv')

# Remove date column
dataWithoutDate = np.delete(np.array(df), 0, 1)

# Define X set which is the data without the training labels
X = np.array(np.delete(dataWithoutDate, 1, 1), dtype=np.float64)
print(len(X))
# Define training labels separately
labels = np.array(dataWithoutDate.take(6, 1),dtype=np.float64)

# Train test split data 80/20
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = labels[:split]
y_test = labels[split:]

# Preprocess and fit data using scalar
scalar = StandardScaler()
scalar.fit(X_train)

# Preprocessing step
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Instantiate classifier and fit data to model
mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 250, 125), max_iter=5000000)
mlp.fit(X_train, y_train)

# Predict values
y_predict = mlp.predict(X_test)
print(y_predict)

# Print Classification report nd confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))


