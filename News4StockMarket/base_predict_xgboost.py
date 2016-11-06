
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score,roc_curve
import bow_tfidf
seed = 999

Xtrain,ytrain = bow_tfidf.load_tfidf('train')
Xtest,ytest = bow_tfidf.load_tfidf('test')

train_matrix = xgb.DMatrix(Xtrain,ytrain)
test_matrix = xgb.DMatrix(Xtest,ytest)

params = {}
params['silent'] = 1
params['objective'] = 'binary:logistic'  # output probabilities
params['eval_metric'] = 'auc'
params["num_rounds"] = 300
params["early_stopping_rounds"] = 50
# params['min_child_weight'] = 2
params['max_depth'] = 6
params['eta'] = 0.1
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7

cv_results = xgb.cv(params,train_matrix,
                    num_boost_round = params["num_rounds"],
                    nfold = params.get('nfold',5),
                    metrics = params['eval_metric'],
                    early_stopping_rounds = params["early_stopping_rounds"],
                    verbose_eval = True,
                    seed = seed)



