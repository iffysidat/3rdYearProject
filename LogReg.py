import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("^GSPC (6).csv")
dataWithoutDate = data.drop(columns=["Date"])

print(dataWithoutDate)


