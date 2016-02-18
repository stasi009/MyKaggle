
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
testdata = pd.read_csv("raw_datas/test.csv",index_col = "id")

xg_entire_train = xgb.DMatrix("train.dmatrix")
xg_test = xgb.DMatrix("test.dmatrix")

class Predictor(object):

    def __init__(self,param):
        param['objective'] = 'multi:softprob'# output probabilities
        param['eval_metric'] = 'mlogloss'
        param['nthread'] = 4
        param['silent'] = 1
        param['num_class'] = trainData.unique_labels().size
        self.param = param

        self.folds_logloss = []
        self.folds_best_iterations = []

    def cv_train(self):
        num_rounds = self.param["num_rounds"]
        early_stopping_rounds = self.param["early_stop_rounds"]
        num_cv = self.param["num_cv"]
    
        # ------------------- begin cross-validation
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

            self.folds_logloss.append(bst.best_score)
            print "best score: %3.2f" % (bst.best_score)
            self.folds_best_iterations.append(bst.best_iteration)
            print "best_iteration: %d" % (bst.best_iteration)

            # predict on validation set
            predict_on_validate = bst.predict(xg_validate,ntree_limit=bst.best_iteration)
            fold_logloss = log_loss(fold_yvalidate, predict_on_validate)
            print "logloss re-calculated: %3.2f" % fold_logloss
            if abs(fold_logloss - bst.best_score) > 1e-3: 
                raise Exception("my logloss not equal with xgboost's logloss")

            predict_on_validate = pd.DataFrame(predict_on_validate,index = fold_xvalidate.index, columns = labels)
            folds_predicts.append(predict_on_validate)

        # ------------------- summary
        self.cv_predict_probs = pd.concat(folds_predicts)

        print "\n******************** TRAIN COMPLETED ********************"
        print "\nParameters: %s" % (self.param)
        print "\nfolds best iterations: %s, \nmean=%3.2f, median=%d, STD=%3.2f" % (self.folds_best_iterations,np.mean(self.folds_best_iterations),np.median(self.folds_best_iterations),np.std(self.folds_best_iterations))
        print "\nfolds logloss: %s \nmean=%3.2f, STD=%3.2f" % (self.folds_logloss,np.mean(self.folds_logloss),np.std(self.folds_logloss))

    def refit_save_model(self,num_rounds, modelname):
        # -------------------- refit on entire train dataset
        watchlist = [(xg_entire_train,'entire_train')]
        self.bst = xgb.train(self.param,xg_entire_train,num_rounds,watchlist)

        # -------------------- save model
        self.bst.save_model(modelname)

    def save_meta_features(self,fname):
        self.cv_predict_probs.to_csv("meta_features/%s" % fname,index_labels="id")

    def predict_on_test(self,filename):
        prediction = self.bst.predict(xg_test)
        prediction = pd.DataFrame(prediction,index = testdata.index, columns = labels)

        # save prediction into file
        prediction.to_csv(filename,index_label="id")


file_offset = 1

param = {}
param['max_depth'] = 10
param["num_rounds"] = 500
param["early_stop_rounds"] = 20
param["num_cv"] = 5
param["seed"] = 9

predictor = Predictor(param)
predictor.cv_train()

file_offset += 1
predictor.save_meta_features("xgb_cv%d.csv"%file_offset)
predictor.refit_save_model(153,"xgboost%d.xgb"%file_offset)
predictor.predict_on_test("xgb_predict%d.csv"%file_offset)



