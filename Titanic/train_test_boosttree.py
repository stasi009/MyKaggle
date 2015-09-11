
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","GiveHelp","RecvHelp"]

def train():
    # ---------------------- load the data
    titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")
    Xtrain = titanic_train[feature_names]
    ytrain = titanic_train["Survived"]

    # ---------------------- train
    loss = ['deviance', 'exponential']
    learning_rate = np.logspace(-5,2)
    n_estimate_dist = sp_randint(1000,4800)
    max_depth_dist = sp_randint(1,9)
    param_dist = dict(loss=loss,
                    learning_rate=learning_rate,
                    n_estimators=n_estimate_dist,
                    max_depth=max_depth_dist)

    gbdt = GradientBoostingClassifier(verbose=1)
    searchcv = RandomizedSearchCV(estimator=gbdt, param_distributions=param_dist,n_iter=214,verbose=1,n_jobs=4)

    print "--------------------- RandomizedSearchCV begins"
    searchcv.fit(Xtrain,ytrain)      
    print "--------------------- RandomizedSearchCV ends"

    with open('gbdt.pkl', 'wb') as f:
        pickle.dump(searchcv.best_estimator_,f)
    print "--------------------- GBDT saved into file"

def test():
    with open("gbdt.pkl","rb") as infile:
        gbdt = pickle.load(infile)

    titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
    Xtest = titanic_test[feature_names]

    predictions = gbdt.predict(Xtest)
    submission = pd.DataFrame({
            "PassengerId": titanic_test.index,
            "Survived": predictions
        })
    submission.to_csv("submit_gbdt.csv", index=False)

# remember to include below line, otherwise, 
# multiprocessing will cause runtime error under windows
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="train or test?")
    
    args = parser.parse_args()
    
    if args.job == "train":
        train()
    elif args.job == "test":
        test()
    else:
        raise ValueError("unknown command")

    
