
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

        self.folds_train_loss = []
        self.folds_val_loss = []
        self.folds_best_iterations = []

    def _fold_predict(self,bst,dmatrix,loglosses):
        predicts = bst.predict(dmatrix,ntree_limit=bst.best_iteration)
        logloss = log_loss(dmatrix.get_label(), predicts)
        loglosses.append(logloss)
        return logloss,predicts

    def _summarize_logloss(self,prefix,loglosses):
        txt_losses = ",".join(("%3.2f"%e for e in loglosses))
        print "\n%s logloss: [%s] \nmean=%3.2f, STD=%3.2f" % (prefix,txt_losses,np.mean(loglosses),np.std(loglosses))

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

            print "best score: %3.2f" % (bst.best_score)
            self.folds_best_iterations.append(bst.best_iteration)
            print "best_iteration: %d" % (bst.best_iteration)

            # predict on train set
            self._fold_predict(bst,xg_train,self.folds_train_loss)

            # predict on validation set
            val_loss,predict_on_validate = self._fold_predict(bst,xg_validate,self.folds_val_loss)
            print "logloss re-calculated: %3.2f" % val_loss
            if abs(val_loss - bst.best_score) > 1e-3: 
                raise Exception("my logloss not equal with xgboost's logloss")

            folds_predicts.append(pd.DataFrame(predict_on_validate,index = fold_xvalidate.index, columns = labels))

        # ------------------- summary
        self.cv_predict_probs = pd.concat(folds_predicts)

        print "\n******************** TRAIN COMPLETED ********************"
        print "\nParameters: %s" % (self.param)
        print "\nfolds best iterations: %s, \nmean=%3.2f, median=%d, STD=%3.2f" % (self.folds_best_iterations,np.mean(self.folds_best_iterations),np.median(self.folds_best_iterations),np.std(self.folds_best_iterations))
        self._summarize_logloss("train",self.folds_train_loss)
        self._summarize_logloss("validate",self.folds_val_loss)

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
param['max_depth'] = 6
param['eta'] = 0.3
param["num_rounds"] = 1000
param["subsample"] = 1
param["colsample_bytree"] = 1
param["early_stop_rounds"] = 15
param["num_cv"] = 5
# param["seed"] = 9

predictor = Predictor(param)
predictor.cv_train()

file_offset += 1
predictor.save_meta_features("xgb_cv%d.csv"%file_offset)
predictor.refit_save_model(153,"xgboost%d.xgb"%file_offset)
predictor.predict_on_test("xgb_predict%d.csv"%file_offset)



