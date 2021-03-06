﻿
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# --------------------------- load train data
titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

# feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","Ticket-7digit","Ticket-A","Ticket-C","Ticket-F","Ticket-Others","Ticket-P","Ticket-S","Ticket-W"]
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","GiveHelp","RecvHelp"]
Xtrain = titanic_train[feature_names]
ytrain = titanic_train["Survived"]

# --------------------------- load test data
titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
Xtest = titanic_test[feature_names]

# --------------------------- scale train data
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

# --------------------------- scale test data
Xtest_scaled = scaler.transform(Xtest)

# --------------------------- LR
lrcv = LogisticRegressionCV(Cs=30,cv=10)
lrcv.fit(Xtrain_scaled,ytrain)

lrcv.C_
lrcv.score(Xtrain_scaled,ytrain)

predict_train = lrcv.predict(Xtrain_scaled)
wrong_samples = titanic_train[ predict_train != ytrain ]
wrong_samples.to_csv("wrong_lr.csv",index_col="PassengerId")

def pretty_print_coef(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)     for coef, name in lst)
pretty_print_coef(lrcv.coef_.ravel(),feature_names,True)

coefs = pd.DataFrame({"names":feature_names,"coefs":lrcv.coef_.ravel()},columns=["names","coefs"])
coefs["rank"] = np.abs(coefs.coefs)
coefs.sort_index(by="rank",inplace=True,ascending=False)
del coefs["rank"]

# --------------------------- predict
predictions = lrcv.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv("submit_lr.csv", index=False)