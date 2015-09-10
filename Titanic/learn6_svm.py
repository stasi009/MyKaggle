
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

# ------------------ prepare the data
titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","Ticket-7digit","Ticket-A","Ticket-C","Ticket-F","Ticket-Others","Ticket-P","Ticket-S","Ticket-W"]
Xtrain = titanic_train[feature_names]
ytrain = titanic_train["Survived"]

scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

# ------------------ train
svc = LinearSVC(dual=False)
Cs = np.logspace(-4,4)

# cannot use "n_jobs=-1", because multiprocessing cannot run within IPython interactive environment under windows
searchcv = GridSearchCV(estimator=svc, param_grid=dict(C = Cs),n_jobs=1,cv=10)
searchcv.fit(Xtrain_scaled,ytrain)    

searchcv.best_score_    
searchcv.best_params_                              
searchcv.best_estimator_

# ------------------ predict
titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
Xtest = titanic_test[feature_names]
Xtest_scaled = scaler.transform(Xtest)

predictions = searchcv.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv("submit_linear_svc.csv", index=False) 