
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint


titanic = pd.DataFrame.from_csv("train_processed.csv")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale"]
# feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

param_dist = {"n_estimators=200":  sp_randint(100,1500),                
              "max_depth": [2,3, 4,None],              
              "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier(oob_score=True)
searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,n_iter=200)
searchcv.fit(Xtrain,ytrain)    

sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),reverse=True)

# ---------------------------------------------------- #
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]

predictions = rf.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_rf_fewfeature.csv", index=False)
