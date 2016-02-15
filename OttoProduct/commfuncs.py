
import pickle
import pandas as pd
import numpy as np
import matplotlib.cm as colormap
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold

class RawTrainData(object):

    def __init__(self,filename):
        train_csv = pd.read_csv(filename,index_col = "id")
        self.Xtrain = train_csv.iloc[:,xrange(93)]
        self.labels = train_csv.loc[:,"target"]

        self.labelencoder = LabelEncoder()
        self.ytrain = self.labelencoder.fit_transform(self.labels)

    def unique_labels(self):
        return self.labels.unique()

    def boolindex_by_label(self,label):
        return np.asarray(self.labels == label)


def dump_predictor(filename,predictor):
    with open(filename, 'wb') as outfile:
        pickle.dump(predictor,outfile)

def load_predictor(filename):
    with open(filename,"rb") as infile:
        return pickle.load(infile)

def colors_iterator(num_lables):
    return iter(colormap.rainbow(np.linspace(0, 1, num_lables)))

def cv_predict(predictor,tag,trainData,num_cv,random_seed=None):
    folds_predicts = []
    logloss = 0.0

    kfolds = StratifiedKFold(y=trainData.ytrain,n_folds=num_cv)
    for train_indices, test_indices in kfolds:
        # ---------------------- fit on training fold
        fold_xtrain = trainData.Xtrain.iloc[train_indices,:]
        fold_ytrain = trainData.ytrain[train_indices]
        predictor.fit(fold_xtrain,fold_ytrain)

        # ---------------------- predict on test fold
        fold_xtest = trainData.Xtrain.iloc[test_indices,:]
        fold_ytest = trainData.ytrain[test_indices]
        predicted_probs = predictor.predict_proba(fold_xtest)

        # ---------------------- statistics on logloss
        logloss += log_loss(fold_ytest, predicted_probs)

        # ---------------------- assemble into DataFrame
        predicted_probs = pd.DataFrame(predicted_probs,
                                       index= fold_xtest.index,
                                       columns = [tag+"_"+label for label in  trainData.labelencoder.inverse_transform(predictor.classes_) ])
        folds_predicts.append(predicted_probs)

    cv_predict_probs = pd.concat(folds_predicts)
    logloss = logloss / num_cv

    return logloss,cv_predict_probs

        

























