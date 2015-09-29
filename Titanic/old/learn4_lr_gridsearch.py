
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.preprocessing as skpreprocess
import sklearn.linear_model as sklinear
from sklearn.grid_search import GridSearchCV


titanic = pd.DataFrame.from_csv("train_processed.csv")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

lr = sklinear.LogisticRegression()
Cs = np.logspace(-4,4,50)
penalty = ['l1' , 'l2']

# cannot use "n_jobs=-1", because multiprocessing cannot run within IPython interactive environment under windows
gridsearch = GridSearchCV(estimator=lr, param_grid=dict(C = Cs,penalty=penalty),n_jobs=1,cv=10)
gridsearch.fit(Xtrain_scaled,ytrain)      

for params, mean_score, scores in gridsearch.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

  
gridsearch.best_score_                                  
gridsearch.best_estimator_.C 
gridsearch.best_estimator_.penalty   


titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]
Xtest_scaled = scaler.transform(Xtest)

predictions = gridsearch.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_lr_gridsearch.csv", index=False) 


