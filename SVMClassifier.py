import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def addTrainingLables(filename):
    data = pd.read_csv(filename)
    labels = []
    for i in range(0, len(data) - 1):
        labels.append(0 if (data["Close"].values[i] > data["Close"].values[i+1]) else 1)
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)
    data["Label"] = trainingLables
    data = data[:-1]
    return data


df = addTrainingLables('S&P5Years.csv')
df.set_index('Date', inplace=True)
print(df.head())

X = df.drop('Label', axis=1)
y = df['Label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



