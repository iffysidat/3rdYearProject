import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd

def addTrainingLables(filename):
    data = pd.read_csv(filename)
    print(data)
    print(data["Close"].values[0])

    labels = []
    for i in range(0, len(data) - 1):
        labels.append(0 if (data["Close"].values[i] > data["Close"].values[i+1]) else 1)
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)

    print(trainingLables)
    data["Label"] = trainingLables
    print(data)
    data = data[:-1]
    print(data)



addTrainingLables("^GSPC (5).csv")


fitnessMeasure = []
layers = []
neurons = []
numEvents = len(y_test)
for numLayers in np.add(range(4),2):
    hiddenLayers = []
    row = []
    rowLay = []
    rowNeur = []
    for numNeurons in (np.add(range(20),8)):
        hiddenLayers = [numNeurons] * numLayers
        mlp = MLPClassifier(hidden_layer_sizes=hiddenLayers, max_iter=20000)
        print(hiddenLayers)
        mlp.fit(X_train, y_train)

        y_predict = mlp.predict(X_test)
        numCorrect = (np.subtract(y_test,y_predict) == 0).sum()
        fracCorrect = numCorrect / numEvents
        row.append(fracCorrect)
        rowLay.append(numLayers)
        rowNeur.append(numNeurons)
    fitnessMeasure.append(row)
    layers.append(rowLay)
    neurons.append(rowNeur)
#
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

import matplotlib.pyplot as plt
plt.hist(np.subtract(y_predict, y_test))
plt.contourf(neurons, layers, fitnessMeasure)#, levels = np.linspace(0,1,11))
plt.xlabel("Number of neurons per layer")
plt.ylabel("Number of layers")
plt.title("MLP performance measured by fractional prediction accuracy")
cbar = plt.colorbar();
plt.savefig("fitnessContour.pdf")
plt.show()
