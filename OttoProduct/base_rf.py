
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.externals import joblib

import commfuncs

def train(seed):
    tag = "rf%d"%seed
    num_cv = 4

    # ------------------------ load and prepare the data
    trainData = commfuncs.RawTrainData("raw_datas/train.csv")

    # ------------------------ prepare the estimator
    param_dist = {"n_estimators":  sp_randint(500,3000),              
                  "max_depth": [ 5,10,20,30,50,100,None],
                  "min_samples_split": sp_randint(2,20),
                  "max_features": ["auto","sqrt","log2",None], 
                  "criterion": ["gini", "entropy"],
                  "min_samples_leaf": [5,10,20,30,40,50]}
    njobs = -1
    rf = RandomForestClassifier(oob_score=True,verbose=1,n_jobs=njobs,random_state=seed)
    searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                  scoring = "log_loss",random_state=seed,                                  
                                  n_iter=200,n_jobs=njobs,cv=num_cv)

    # ------------------------ search the best parameters
    print "#################### search cv begins"
    searchcv.fit(trainData.Xtrain,trainData.ytrain)    
    print "#################### search cv ends"
    print "best score: ",searchcv.best_score_                                  
    print "best parameters: ",searchcv.best_params_

    # ------------------------ save the best estimator
    # use joblib to improve the disk efficiency when dumping the estimator
    joblib.dump(searchcv.best_estimator_,"%s.pkl"%tag)

    # ------------------------ cross-validation to generate predicted probabilities
    # ------------------------ preparing for stack generalization in next step
    logloss,cv_predicted_probs = commfuncs.cv_predict(searchcv.best_estimator_,tag,trainData,num_cv,seed)

    cv_predicted_probs.to_csv("meta_features/%s.csv"%tag,index_labels="index")
    print "cv logloss is: %3.2f"%(logloss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="random seed")
    
    args = parser.parse_args()
    train(int( args.seed ))

