
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.preprocessing as skpreprocess
import sklearn.linear_model as sklinear
import sklearn.ensemble as skensemble
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

# --------------------------- load train data
titanic_train = pd.DataFrame.from_csv("train_processed.csv")

#feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale"]
Xtrain = titanic_train[feature_names]
ytrain = titanic_train["Survived"]

# --------------------------- load test data
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]

# --------------------------- scale train data
scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

# --------------------------- scale test data
Xtest_scaled = scaler.transform(Xtest)

# --------------------------- prepare to hold the result
train_results = {"PassengerId":titanic_train.PassengerId}
train_results["Actual"] = ytrain

# --------------------------- LR
lrcv = sklinear.LogisticRegressionCV(Cs=30,cv=10)
lrcv.fit(Xtrain_scaled,ytrain)
lrcv.score(Xtrain_scaled,ytrain)
lrcv.C_
lr_results = lrcv.predict(Xtrain_scaled)
train_results["LR"] = lr_results

# --------------------------- RF
rf = skensemble.RandomForestClassifier(n_estimators=200, criterion="entropy",oob_score=True)
rf.fit(Xtrain,ytrain)
rf.oob_score_
train_results["RF"] = rf.predict(Xtrain)

# --------------------------- SVC
svc = LinearSVC(dual=False)
Cs = np.logspace(-4,4)
searchcv = GridSearchCV(estimator=svc, param_grid=dict(C = Cs),n_jobs=1,cv=10)
searchcv.fit(Xtrain_scaled,ytrain)    

searchcv.best_score_                                  
searchcv.best_estimator_
train_results["SVC"] = searchcv.predict(Xtrain_scaled)

# --------------------------- GBDT
import pickle
inf = open("gbdt2.pkl","rb")
gbdt = pickle.load(inf)
inf.close()

gbdt.score(Xtrain,ytrain)
train_results["GBDT"] = gbdt.predict(Xtrain)

# --------------------------- Form DataFrame
rsltframe = pd.DataFrame(train_results)
rsltframe = rsltframe.set_index("PassengerId")
rsltframe.to_csv("train_results.csv",index_label="PassengerId")

# --------------------------- check training error
def train_score(col):
    temp = (col == rsltframe.Actual).astype(int)
    return temp.sum()/(float(len(temp)))
train_scores = rsltframe.apply(train_score)

# --------------------------- check training error
lr_wrong_ids = rsltframe.index[ rsltframe.LR != rsltframe.Actual ]



