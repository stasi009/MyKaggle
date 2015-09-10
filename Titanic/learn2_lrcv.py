
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# --------------------------- load train data
titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","Ticket-7digit","Ticket-A","Ticket-C","Ticket-F","Ticket-Others","Ticket-P","Ticket-S","Ticket-W"]
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

def pretty_print_coef(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)     for coef, name in lst)
pretty_print_coef(lrcv.coef_.ravel(),feature_names,True)

# --------------------------- predict
predictions = lrcv.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv("submit_lr.csv", index=False)