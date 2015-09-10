
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

# remember to include below line, otherwise, runtime error
if __name__ == "__main__":

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
    searchcv = RandomizedSearchCV(estimator=gbdt, param_distributions=param_dist,n_iter=200,verbose=1,n_jobs=4)

    print "--------------------- RandomizedSearchCV begins"
    searchcv.fit(Xtrain,ytrain)      
    print "--------------------- RandomizedSearchCV ends"

    print "--------------------- begin save the trainer"
    import pickle
    f = open('gbdt.pkl', 'wb')
    pickle.dump(searchcv.best_estimator_,f)
    f.close()
    print "--------------------- trainer saved into file"
