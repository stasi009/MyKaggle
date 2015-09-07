
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.preprocessing as skpreprocess
import sklearn.decomposition as skdecompose
import sklearn.manifold as skmanifold
import sklearn.linear_model as sklinear
import sklearn.cross_validation as skcv

titanic = pd.DataFrame.from_csv("train_processed.csv")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

# scale the train data
scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

#
lrcv = sklinear.LogisticRegressionCV(Cs=30)
lrcv.fit(Xtrain_scaled,ytrain)
lrcv.score(Xtrain_scaled,ytrain)
lrcv.C_

#
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]
Xtest_scaled = scaler.transform(Xtest)

predictions = lrcv.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submission.csv", index=False)