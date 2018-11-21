from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from ta import *

df = pd.read_csv("S&PNov2016-Nov2017.csv")

#df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume")

#Function to read data
def addTrainingLables(dataframe):
    labels = []
    for i in range(0, len(dataframe) - 1):
        labels.append(0 if (dataframe["Close"].values[i] > dataframe["Close"].values[i+1]) else 1)
    trainingLables = pd.DataFrame(columns=["Label"], data=labels)
    dataframe["Label"] = trainingLables
    dataframe = dataframe[:-1]
    return dataframe

df = addTrainingLables(df)
df.Timestamp = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df.Timestamp

df['Close'].plot()
plt.show()

# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(np.asarray(df['Close'])).fit(smoothing_level=0.2, optimized=False)
fcast1 = fit1.forecast(12)

y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()
# plot
fcast1.plot(marker='o', color='blue', legend=True)
fit1.fittedvalues.plot(marker='o',  color='blue')




