﻿
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","Ticket-7digit","Ticket-A","Ticket-C","Ticket-F","Ticket-Others","Ticket-P","Ticket-S","Ticket-W"]
Xtrain = titanic_train[feature_names]
ytrain = titanic_train["Survived"]

param_dist = {"n_estimators":  sp_randint(1000,5000),                
              "max_depth": [2,3, 4,5,6,7,8,9,None],              
              "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier(oob_score=True,verbose=1)
searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,n_iter=200)
searchcv.fit(Xtrain,ytrain)    

searchcv.best_score_                                  
searchcv.best_estimator_
searchcv.best_params_

import pickle
outfile = open('rf.pkl', 'wb')
pickle.dump(searchcv.best_estimator_,outfile)
outfile.close()

rf = searchcv.best_estimator_
sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),reverse=True)


predict_on_train = rf.predict(Xtrain)
wrong_samples = titanic_train[predict_on_train != ytrain]
wrong_samples.to_csv("wrong_rf.csv",index_col="PassengerId")

# ---------------------------------------------------- #
titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
Xtest = titanic_test[feature_names]

predictions = rf.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv("submit_rf.csv", index=False)
