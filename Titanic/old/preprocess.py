
import numpy as np
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')

traindata = pd.read_csv("train.csv")
traindata.Age.hist(bins=50)
traindata.Age.plot(kind="kde")
traindata.Fare.hist(bins=50)
traindata.Fare.plot(kind="kde")




testdata = pd.read_csv("test.csv")
traindata_proc = pd.DataFrame.from_csv("train_processed.csv")

testdata.Age = testdata.Age.fillna(traindata_proc.Age.median())

# fare
train_fare_median = traindata_proc.Fare.median()
traindata_proc.Fare.mean()
testdata.Fare = testdata.Fare.fillna(train_fare_median)

testdata.Embarked.groupby(testdata.Embarked).size()
