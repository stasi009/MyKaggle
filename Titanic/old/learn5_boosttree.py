
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

# ---------------------- load the data
titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","Ticket-7digit","Ticket-A","Ticket-C","Ticket-F","Ticket-Others","Ticket-P","Ticket-S","Ticket-W"]
Xtrain = titanic_train[feature_names]
ytrain = titanic_train["Survived"]

# ---------------------- train
loss = ['deviance', 'exponential']
learning_rate = np.logspace(-5,2)
n_estimate_dist = sp_randint(1000,5000)
max_depth_dist = sp_randint(1,9)
param_dist = dict(loss=loss,
                learning_rate=learning_rate,
                n_estimators=n_estimate_dist,
                max_depth=max_depth_dist)

gbdt = GradientBoostingClassifier(verbose=1)
searchcv = RandomizedSearchCV(estimator=gbdt, param_distributions=param_dist,n_iter=200,verbose=1)
searchcv.fit(Xtrain,ytrain)      

searchcv.best_score_                                  
searchcv.best_estimator_
searchcv.best_params_


# ---------------------- predict
titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
Xtest = titanic_test[feature_names]

predictions = searchcv.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv("submit_gbdt.csv", index=False)






import pickle
inf = open('gbdt.pkl', 'rb')
gbdt = pickle.load(inf)
inf.close()

sorted(zip(map(lambda x: round(x, 4), gbdt.feature_importances_), feature_names),reverse=True)


titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
Xtest = titanic_test[feature_names]

predictions = gbdt.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv("submit_gbdt.csv", index=False)
