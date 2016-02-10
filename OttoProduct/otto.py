
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from scipy.stats import randint as sp_randint

import xgboost as xgb
import sklearn.preprocessing as skpreprocess

train_csv = pd.read_csv("train.csv",index_col = "id")
train_datas = train_csv.iloc[:,xrange(93)]
train_labels = train_csv.loc[:,"target"]

train_labels.value_counts().plot(kind="bar")

labelencoder = skpreprocess.LabelEncoder()
train_targets = labelencoder.fit_transform(train_labels)

