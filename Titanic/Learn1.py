
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpreprocess
import sklearn.linear_model as sklinear
import sklearn.cross_validation as skcv

titanic = pd.DataFrame.from_csv("train_processed.csv")
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]

Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

# scale the train data
scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

# fit a LR classifier
lr = sklinear.LogisticRegression()
lr.fit(Xtrain_scaled,ytrain)
train_accuracy = lr.score(Xtrain_scaled,ytrain)


