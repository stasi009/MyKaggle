
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.ensemble as skensemble
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint


titanic = pd.DataFrame.from_csv("train_processed.csv")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale"]
# feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]



loss = ['deviance', 'exponential']
learning_rate = np.logspace(-4,3)
n_estimate_dist = sp_randint(100,1000)
max_depth_dist = sp_randint(1,5)
param_dist = dict(loss=loss,
                learning_rate=learning_rate,
                n_estimators=n_estimate_dist,
                max_depth=max_depth_dist)


gbdt = skensemble.GradientBoostingClassifier()
n_iter_search = 200
searchcv = RandomizedSearchCV(estimator=gbdt, param_distributions=param_dist,n_iter=n_iter_search)
searchcv.fit(Xtrain,ytrain)      

searchcv.best_score_                                  
searchcv.best_estimator_
searchcv.best_params_


# ---------------------------------------------------- #
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]

predictions = searchcv.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_gbdt2.csv", index=False)






import pickle
f = open('gbdt2.pkl', 'wb')
pickle.dump(searchcv.best_estimator_,f)
f.close()

import pprint
f = open('gbdt.pkl', 'rb')
clf = pickle.load(f)
pprint.pprint( clf)
f.close()
