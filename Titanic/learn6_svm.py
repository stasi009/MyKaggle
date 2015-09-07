
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.preprocessing as skpreprocess
import sklearn.linear_model as sklinear
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC


titanic = pd.DataFrame.from_csv("train_processed.csv")

#feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale"]
Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)


svc = LinearSVC(dual=False)
Cs = np.logspace(-4,4)

# cannot use "n_jobs=-1", because multiprocessing cannot run within IPython interactive environment under windows
searchcv = GridSearchCV(estimator=svc, param_grid=dict(C = Cs),n_jobs=1,cv=10)
searchcv.fit(Xtrain_scaled,ytrain)    

searchcv.best_score_                                  
searchcv.best_estimator_


titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]
Xtest_scaled = scaler.transform(Xtest)

predictions = searchcv.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_linear_svc.csv", index=False) 