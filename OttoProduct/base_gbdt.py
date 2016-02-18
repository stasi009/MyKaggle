
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import commfuncs

# load train datas
trainData = commfuncs.RawTrainData("raw_datas/train.csv")
labels = trainData.unique_labels()

class Predictor(object):

    def __init__(self,param):
        param['objective'] = 'multi:softprob'# output probabilities
        param['eval_metric'] = 'mlogloss'
        param['nthread'] = 4
        param['silent'] = 1
        param['num_class'] = trainData.unique_labels().size
        self.param = param

    def cv_train(self):
        num_rounds = self.param["num_rounds"]
        early_stopping_rounds = self.param["early_stop_rounds"]
        num_cv = self.param["num_cv"]
    
        # ------------------- begin cross-validation
        logloss = 0.0
        folds_predicts = []

        kfolds = StratifiedKFold(y=trainData.ytrain,n_folds=num_cv)
        for k,(train_indices, validate_indices) in enumerate(kfolds):
            print "************** %d-th fold **************" % (k + 1)

            # prepare the data
            xg_train = xgb.DMatrix(trainData.Xtrain.iloc[train_indices,:], label=trainData.ytrain[train_indices])
            fold_xvalidate = trainData.Xtrain.iloc[validate_indices,:]
            fold_yvalidate = trainData.ytrain[validate_indices]
            xg_validate = xgb.DMatrix(fold_xvalidate,fold_yvalidate)
        
            # train
            watchlist = [(xg_train,'train'), (xg_validate, 'validate')]# early stop will check on the last dataset
            bst = xgb.train(self.param, xg_train, num_rounds, watchlist,early_stopping_rounds=early_stopping_rounds)

            logloss += bst.best_score
            print "best_iteration: %d" % (bst.best_iteration)
            print "best score: %3.2f" % (bst.best_score)

            # predict on validation set
            predict_on_validate = bst.predict(xg_validate,ntree_limit=bst.best_iteration)
            fold_logloss = log_loss(fold_yvalidate, predict_on_validate)
            print "logloss re-calculated: %3.2f" % fold_logloss
            if abs(fold_logloss - bst.best_score) > 1e-3: 
                raise Exception("my logloss not equal with xgboost's logloss")

            predict_on_validate = pd.DataFrame(predict_on_validate,index = fold_xvalidate.index, columns = labels)
            folds_predicts.append(predict_on_validate)

        self.logloss = logloss / num_cv
        self.cv_predict_probs = pd.concat(folds_predicts)
        print "Parameters: %s" % (self.param)
        print "average logloss: %3.2f" % (self.logloss)

    def refit_save(self,modelname):
        self.bst.save_model(modelname)

    def save_meta_features(self,fname):
        self.cv_predict_probs.to_csv("meta_features/%s" % fname,index_labels="id")



param = {}
param['max_depth'] = 6
param["num_rounds"] = 500
param["early_stop_rounds"] = 20
param["num_cv"] = 5

predictor = Predictor(param)
predictor.cv_train()



