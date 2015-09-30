

import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import sklearn.preprocessing as skpreprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

import common

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit"]

def train_cv():
    # --------------------- load train data
    train_df = pd.read_csv("train_processed.csv",index_col="PassengerId")
    ytrain = train_df["Survived"]
    Xtrain_scaled = skpreprocess.scale(train_df[feature_names])

    param_dist = {"n_neighbors":  np.arange(2,11)}
    knn = KNeighborsClassifier()
    searchcv = GridSearchCV(estimator=knn, param_grid=param_dist,n_jobs=4,cv=10)

    print "#################### search cv begins"
    searchcv.fit(Xtrain_scaled,ytrain)    
    print "#################### search cv ends"
    print "best score: ",searchcv.best_score_                                  
    print "best parameters: ",searchcv.best_params_

    common.dump_predictor('knn-cv.pkl',searchcv.best_estimator_)
    print "*** RF saved into file"

def train_whole():
    prev_knn = common.load_predictor("knn-cv.pkl")
    ()


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
    