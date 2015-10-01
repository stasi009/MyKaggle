
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

import common

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit"]

def train():
    train_df = pd.read_csv("train_processed.csv",index_col="PassengerId")
    Xtrain = train_df[feature_names]
    ytrain = train_df["Survived"]

    param_dist = {"n_estimators":  sp_randint(500,3000),                
                  "max_depth": [2,3, 4,5,6,None],              
                  "criterion": ["gini", "entropy"]}
    njobs = 4
    rf = RandomForestClassifier(oob_score=True,verbose=1,n_jobs=njobs)
    searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,n_iter=200,n_jobs=njobs)

    print "#################### search cv begins"
    searchcv.fit(Xtrain,ytrain)    
    print "#################### search cv ends"
    print "best score: ",searchcv.best_score_                                  
    print "best parameters: ",searchcv.best_params_

    common.dump_predictor('rf.pkl',searchcv.best_estimator_)
    print "*** RF saved into file"

def test():
    test_df = pd.read_csv("test_processed.csv",index_col="PassengerId")
    Xtest = test_df[feature_names]

    rf = common.load_predictor("rf.pkl")
    predictions = rf.predict(Xtest)

    common.make_submission(Xtest.index,predictions,"submit_rf.csv")

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
    