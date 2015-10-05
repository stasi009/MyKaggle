
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import RandomizedSearchCV

import commfuncs

class Estimator(object):

    def __init__(self,feature_names,target_column):
        self.target_column = target_column
        self._feature_names = feature_names
        self._pklname = "gbdt_%s.pkl"%(target_column)

    def train(self,traindf,parm_dist,n_iter,n_jobs = 4):
        searchcv = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=parm_dist,n_iter=n_iter,verbose=1,n_jobs=n_jobs,cv=5)

        print "------------ RandomizedSearchCV begins for '%s'"%self.target_column
        Xtrain = traindf[self._feature_names]
        ytrain = traindf[self.target_column]
        searchcv.fit(Xtrain,ytrain)      
        print "------------ RandomizedSearchCV ends for '%s'"%self.target_column

        self._gbdt = searchcv.best_estimator_
        return searchcv.best_score_    

    def dump(self):
        commfuncs.dump_predictor(self._pklname,self._gbdt)

    def load(self):
        self._gbdt = commfuncs.load_predictor(self._pklname)

    def test(self,testdf):
        Xtest = testdf[self._feature_names]
        predicts = self._gbdt.predict(Xtest)
        return pd.Series( commfuncs.logcounts_to_counts(predicts),index = testdf.index)

def train(feature_names,estimators):
    traindf = pd.read_csv("train_extend.csv",index_col="datetime")

    parm_dist = dict(learning_rate=[0.001,0.005,0.01,0.02,0.05,0.1,0.3],                   
                     n_estimators=sp_randint(100,2000),                    
                     max_depth=sp_randint(3,6),
                     min_samples_leaf = range(1,10)
                     )
    n_iter = 300
    n_jobs = 6
    for estimator in estimators:
        best_cv_score = estimator.train(traindf,parm_dist,n_iter,n_jobs)
        print "************* '%s' got best CV score: %f"%(estimator.target_column,best_cv_score)
        estimator.dump()

def test(feature_names,estimators):
    testdf = pd.read_csv("test_extend.csv",index_col="datetime")

    predicts = [ e.test(testdf) for e in estimators]
    predicts = pd.concat(predicts,axis=1,keys = [e.target_column  for e in estimators])

    predicts["count"] = predicts.sum(axis=1)
    predicts[["count"]].to_csv("submit_gbdt.csv",index_label="datetime")

if __name__ == "__main__":

    feature_names = ['month',  'weekday', 'hour', 'season', 'holiday','workingday', 'weather', 'atemp', 'humidity', 'windspeed']
    estimators = [Estimator(feature_names,"log_casual"),Estimator(feature_names,"log_registered")]

    train(feature_names,estimators)
    print "---------------- train completed ----------------"

    test(feature_names,estimators)
    print "---------------- test completed ----------------"


    







    

