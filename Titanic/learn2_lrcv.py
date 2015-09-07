
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.preprocessing as skpreprocess
import sklearn.linear_model as sklinear


titanic = pd.DataFrame.from_csv("train_processed.csv")

# feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale"]

Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]


scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)



lrcv = sklinear.LogisticRegressionCV(Cs=30,cv=10)
lrcv.fit(Xtrain_scaled,ytrain)
lrcv.score(Xtrain_scaled,ytrain)
lrcv.C_

# ------------- display how score change vs. C
scores = np.asarray( lrcv.scores_[1] )
average_scores = scores.mean(axis=0)

log_cs = np.log10(lrcv.Cs_)
plt.plot(log_cs,average_scores)
plt.axvline(np.log10(lrcv.C_))


def pretty_print_coef(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)     for coef, name in lst)

pretty_print_coef(lrcv.coef_.ravel(),feature_names,True)


# In[48]:

titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]
Xtest_scaled = scaler.transform(Xtest)


# In[49]:

predictions = lrcv.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_lr_fewfeatures.csv", index=False)

