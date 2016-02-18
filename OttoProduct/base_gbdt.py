
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

def experiment(param):
    # ------------------- prepare parameters
    param['objective'] = 'multi:softprob'# output probabilities
    param['eval_metric'] = 'mlogloss'
    param['nthread'] = 4
    param['silent'] = 1
    param['num_class'] = trainData.unique_labels().size
    num_rounds = param["num_rounds"]
    early_stopping_rounds=param["early_stop_rounds"]
    num_cv = param["num_cv"]
    
    # ------------------- begin cross-validation
    logloss = 0.0
    folds_predicts = []

    kfolds = StratifiedKFold(y=trainData.ytrain,n_folds=num_cv)
    for k,(train_indices, validate_indices) in enumerate(kfolds):
        print "************** %d-th fold **************"%(k+1)

        # prepare the data
        xg_train = xgb.DMatrix( trainData.Xtrain.iloc[train_indices,:], label=trainData.ytrain[train_indices])
        fold_xvalidate = trainData.Xtrain.iloc[validate_indices,:]
        fold_yvalidate = trainData.ytrain[validate_indices]
        xg_validate = xgb.DMatrix(fold_xvalidate,fold_yvalidate)
        
        # train
        watchlist = [ (xg_train,'train'), (xg_validate, 'validate') ]
        bst = xgb.train(param, xg_train, num_rounds, watchlist,early_stopping_rounds=early_stopping_rounds)

        logloss += bst.best_score
        print "best_iteration: %d"%( bst.best_iteration)
        print "best score: %3.2f"%(bst.best_score)

        # predict on validation set
        predict_on_validate = bst.predict(xg_validate,ntree_limit=bst.best_iteration)
        print "logloss re-calculated: %3.2f"%( log_loss(fold_yvalidate, predict_on_validate) )

        predict_on_validate = pd.DataFrame(predict_on_validate,index = fold_xvalidate.index, columns = labels)
        folds_predicts.append(predict_on_validate)

    return (logloss/num_cv),(pd.concat(folds_predicts))

param = {}
param['max_depth'] = 6
param["num_rounds"] = 500
param["early_stop_rounds"] = 20
param["num_cv"] = 5
logloss,cv_predict_probs = experiment(param)




