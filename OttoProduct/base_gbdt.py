
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import commfuncs

seed = 999

# load train datas
trainData = commfuncs.RawTrainData("raw_datas/train.csv")

# text labels 
txt_labels = trainData.labelencoder.inverse_transform(np.unique(trainData.ytrain))

# split into train set and validation set
xtrain,xvalidate,ytrain,yvalidate = train_test_split(trainData.Xtrain,trainData.ytrain,test_size=0.3,random_state=seed)
xg_train = xgb.DMatrix( xtrain, label=ytrain)
xg_validate = xgb.DMatrix(xvalidate,yvalidate)

# setup parameters for xgboost
param = {}
param['objective'] = 'multi:softprob'# output probabilities
param['eval_metric'] = 'mlogloss'
param['max_depth'] = 6
param['nthread'] = 4
param['silent'] = 1
param['seed'] = seed
param['num_class'] = trainData.unique_labels().size

watchlist = [ (xg_train,'train'), (xg_validate, 'validate') ]
num_round = 500

bst = xgb.train(param, xg_train, num_round, watchlist,early_stopping_rounds=10 )
bst.best_iteration

# predict on test data
testdata = pd.read_csv("raw_datas/test.csv",index_col = "id")
xg_test = xgb.DMatrix(testdata)

prediction = bst.predict(xg_test,ntree_limit=bst.best_iteration)
prediction = pd.DataFrame(prediction,index = testdata.index, columns = txt_labels)

# save prediction into file
prediction.to_csv("gbdt1.csv",index_label="id")
