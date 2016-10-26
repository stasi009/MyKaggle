
import itertools
import os.path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import bow_tfidf
import common
seed = 999

def crossval_predict(predictor, X, y, prefix, n_cv=5):
    if not np.array_equal( predictor.classes_, [0, 1]):
        raise Exception("classes labels NOT match")

    can_pred_proba = common.can_predict_probability(predictor)

    n_samples = X.shape[0]
    print "totally {} samples, divided into {} folds".format(n_samples, n_cv)

    if can_pred_proba:
        datas = np.full((n_samples, 2), np.NaN)
        headers = ["{}_{}".format(prefix, t) for t in ["proba", "log_proba"]]
        yvalidates = pd.DataFrame(datas, columns=headers,index=y.index)
    else:
        datas = np.full((n_samples, 1), np.NaN)
        header = "{}_label".format(prefix)
        yvalidates = pd.DataFrame(datas, columns=[header],index=y.index)

    folds = StratifiedKFold(y, n_folds=n_cv,shuffle=True,random_state=seed)
    for index, (train_index, test_index) in enumerate(folds):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        predictor.fit(Xtrain, ytrain)
        if can_pred_proba:
            ytest_probas = predictor.predict_proba(Xtest)
            pos_proba = ytest_probas[:, 1]  # probability for label=1 (Positive)
            yvalidates.iloc[test_index, 0] = pos_proba
            yvalidates.iloc[test_index, 1] = np.log(pos_proba)
        else:
            yvalidates.iloc[test_index, 0] = predictor.predict(Xtest)

        print "====== cross-validated on {}-fold ======".format(index + 1)

    return yvalidates

class Learner(object):
    def __init__(self):
        self.Xtrain, self.ytrain = bow_tfidf.load_sparse_dataset("train",'tfidf')
        self.Xvalid, self.yvalid = bow_tfidf.load_sparse_dataset("validate",'tfidf')
        self.Xtest, self.ytest = bow_tfidf.load_sparse_dataset('test','tfidf')
        print "train set {}, positive ratio: {:.2f}%".format(self.Xtrain.shape, self.ytrain.mean() * 100)
        print "validation set {}, positive ratio: {:.2f}%".format(self.Xvalid.shape, self.yvalid.mean() * 100)

        # ---------- prepare file to record statistics
        stats_file = "meta_features/model_stats.csv"
        stats_existed = os.path.exists(stats_file)
        self.stats_file = open(stats_file,'at')
        if not stats_existed: # first time
            self.stats_file.write('model,accuracy,auc,description\n')

    def predict_validation(self,predictor,tag):
        valid_pred_result = common.predict_proba_or_label(predictor,self.Xvalid,self.yvalid.index,tag)
        valid_pred_result.to_csv("meta_features/{}_valid.csv".format(tag), index_label='id')

        colname_proba = '{}_proba'.format(tag)
        if colname_proba in valid_pred_result:
            yvalid_pred_proba = valid_pred_result.loc[:,colname_proba]
            valid_auc = roc_auc_score(self.yvalid, yvalid_pred_proba)
            valid_accuracy = accuracy_score(self.yvalid,yvalid_pred_proba > 0.5)
        else:
            assert valid_pred_result.shape[1] == 1, 'only one label column'
            yvalid_pred_label = valid_pred_result.iloc[:,0]
            valid_accuracy = accuracy_score(self.yvalid,yvalid_pred_label)
            valid_auc = np.NaN

        performance_text = "{},{},{},{}\n".format(tag,valid_accuracy,valid_auc,str(predictor))
        self.stats_file.write(performance_text)
        print performance_text

    def run_once(self,predictor,tag):
        # ---------- fit on all training dataset to get the model
        predictor.fit(self.Xtrain, self.ytrain)
        common.simple_dump("meta_features/{}.pkl".format(tag), predictor)
        print "\tModel[{}] learnt and saved".format(tag)

        # ---------- predict on validation
        self.predict_validation(predictor,tag)
        print "\tModel[{}] predicted on validation".format(tag)

        # ---------- predict on test
        test_pred_result = common.predict_proba_or_label(predictor,self.Xtest,self.ytest.index,tag)
        test_pred_result.to_csv("meta_features/{}_test.csv".format(tag), index_label='id')
        print "\tModel[{}] predicted on test".format(tag)

        # ---------- generate meta features
        metafeatures = crossval_predict(predictor, self.Xtrain, self.ytrain, tag)
        metafeatures.to_csv("meta_features/{}_train.csv".format(tag), index_label='id')
        print "\tMeta-features generated from Model[{}]".format(tag)

    def run(self,predictors):
        for (predictor,tag) in predictors:
            print '\nModel[{}] is learning, ......'.format(tag)
            self.run_once(predictor,tag)

    def close(self):
        self.stats_file.close()

if __name__ == "__main__":
    Cs = np.logspace(-2,2,5)
    lrs = [(LogisticRegression(C=c,random_state=seed),'lr{}'.format(index+1)) for index,c in enumerate(Cs)]
    svcs = [(LinearSVC(C=c,random_state=seed),'linsvc{}'.format(index+1)) for index,c in enumerate(Cs)]

    nbs = [(MultinomialNB(),'nb')]

    n_trees = [10,50,100,500]
    rfs = [(RandomForestClassifier(n_estimators=ntree, random_state=seed, verbose=1, n_jobs=-1),
            'rf{}'.format(index+1)) for index,ntree in enumerate(n_trees)]

    models = itertools.chain(lrs,svcs,nbs,rfs)

    learner = Learner()
    learner.run(models)
    learner.close()









