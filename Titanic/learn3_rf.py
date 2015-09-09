
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint


titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")


feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","TicketGroup"]
Xtrain = titanic_train[feature_names]
ytrain = titanic_train["Survived"]

param_dist = {"n_estimators":  sp_randint(100,1500),                
              "max_depth": [2,3, 4,5,None],              
              "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier(oob_score=True,verbose=1)
searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,n_iter=200,verbose=1)
searchcv.fit(Xtrain,ytrain)    

searchcv.best_score_                                  
searchcv.best_estimator_
searchcv.best_params_

rf = searchcv.best_estimator_
sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),reverse=True)

# ---------------------------------------------------- #
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]

predictions = rf.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_rf.csv", index=False)
