
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import common

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit"]

def train_cv():
    # ---------------------- load the data
    train_df = pd.read_csv("train_processed.csv",index_col="PassengerId")
    Xtrain = train_df[feature_names]
    ytrain = train_df["Survived"]

    # ---------------------- train
    loss = ['deviance', 'exponential']
    learning_rate = np.logspace(-5,1)
    n_estimate_dist = sp_randint(500,3000)
    max_depth_dist = sp_randint(1,6)
    param_dist = dict(loss=loss,
                    learning_rate=learning_rate,
                    n_estimators=n_estimate_dist,
                    max_depth=max_depth_dist)

    gbdt = GradientBoostingClassifier(verbose=1)
    searchcv = RandomizedSearchCV(estimator=gbdt, param_distributions=param_dist,n_iter=210,verbose=1,n_jobs=4)

    print "--------------------- RandomizedSearchCV begins"
    searchcv.fit(Xtrain,ytrain)      
    print "--------------------- RandomizedSearchCV ends"
    print "best score: ",searchcv.best_score_                                  
    print "best parameters: ",searchcv.best_params_

    common.dump_predictor('gbdt-cv.pkl',searchcv.best_estimator_)
    print "--------------------- GBDT saved into file"

def train_whole():
    train_df = pd.read_csv("train_processed.csv",index_col="PassengerId")
    Xtrain = train_df[feature_names]
    ytrain = train_df["Survived"]

    # ------------------------------ load
    # this estimator is trained on partial dataset, without using the valiation part
    prev_estimator = common.load_predictor("gbdt-cv.pkl")
    print "cross-validation score: %f"%(prev_estimator.score(Xtrain,ytrain))

    # ------------------------------ train
    # after we get the paramters, we should train another estimator with all data
    gbdt = GradientBoostingClassifier(verbose=1,
                                      loss=prev_estimator.loss,
                                      learning_rate = prev_estimator.learning_rate,
                                      n_estimators = prev_estimator.n_estimators,
                                      max_depth = prev_estimator.max_depth)
    print gbdt
    gbdt.fit(Xtrain,ytrain)
    print "training with all data, get score: ",gbdt.score(Xtrain,ytrain)

    # ------------------------------ save
    common.dump_predictor("gbdt.pkl",gbdt)

def test():
    test_df = pd.read_csv("test_processed.csv",index_col="PassengerId")
    Xtest = test_df[feature_names]

    gbdt = common.load_predictor("gbdt.pkl")
    predictions = gbdt.predict(Xtest)
    common.make_submission(Xtest.index,predictions,"submit_gbdt.csv")

# remember to include below line, otherwise, 
# multiprocessing will cause runtime error under windows
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="train or test?")
    
    args = parser.parse_args()
    if args.job == "train_cv":
        train_cv()
    elif args.job == "train_whole":
        train_whole()
    elif args.job == "test":
        test()
    else:
        raise ValueError("unknown command")

    
