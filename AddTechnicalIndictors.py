import pandas as pd
from ta import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

df = pd.read_csv("S&P15Years.csv", sep=',')

df = utils.dropna(df)

df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume")

#Function to read data
def addTrainingLables(dataframe):
    labels = []
    for i in range(0, len(dataframe) - 1):
        labels.append(0 if (dataframe["Close"].values[i] > dataframe["Close"].values[i+1]) else 1)
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)
    dataframe["Label"] = trainingLables
    dataframe = dataframe[:-1]
    return dataframe

#Get data and add training labels
df = addTrainingLables(df)

print(df.shape)
df.dropna(inplace=True)
print(df.shape)

#Remove date column
dataWithoutDate = np.delete(np.array(df), 0, 1)

#Define X set which is the data without the training labels
#temp = dataWithoutDate
#X = temp.drop(columns=['Label'])
X = np.array(np.delete(dataWithoutDate, 64, 1), dtype=np.float64)

#Define training labels separately
#labels = temp['Label']
labels = np.array(dataWithoutDate.take(64, 1),dtype=np.float64)

#Train test split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

#Preprocess and fit data using scalar
scalar = MinMaxScaler()
scalar.fit_transform(X_train)

#Preprocessing step

X_test = scalar.fit_transform(X_test)

#Instantiaate classifier and fit data to model
mlp = MLPClassifier(max_iter=20000)
mlp.fit(X_train, y_train)

#Predict values
y_predict = mlp.predict(X_test)

#Print Classification report nd confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
