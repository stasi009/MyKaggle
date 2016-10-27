
import numpy as np
import pandas as pd
import xgboost as xgb
import bow_tfidf
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import common
seed = 999

# ********************* load the data
Xtrain, ytrain = bow_tfidf.load_sparse_dataset("train", 'tfidf')
print "Train TF-IDF loaded"
train_matrix = xgb.DMatrix(Xtrain,ytrain)

Xvalid, yvalid = bow_tfidf.load_sparse_dataset("validate", 'tfidf')
print "Validation TF-IDF loaded"
valid_matrix = xgb.DMatrix(Xvalid,yvalid)

Xtest, ytest = bow_tfidf.load_sparse_dataset('test', 'tfidf')
print "Test TF-IDF loaded"
test_matrix = xgb.DMatrix(Xtest)

# ********************* #
def crossval_predict(tag,X,y,params,n_cv=5):

    n_samples = X.shape[0]
    print "totally {} samples, divided into {} folds".format(n_samples, n_cv)

    headers = ["{}_{}".format(tag, t) for t in ["proba", "log_proba"]]
    yvalidates = pd.DataFrame(np.full((n_samples, 2), np.NaN), columns=headers,index=y.index)

    folds = StratifiedKFold(y, n_folds=n_cv,shuffle=True,random_state=seed)
    for index, (train_index, test_index) in enumerate(folds):
        train_matrix = xgb.DMatrix(X[train_index],y[train_index])
        valid_matrix = xgb.DMatrix(X[test_index],y[test_index])

        watchlist = [(train_matrix, 'train'), (valid_matrix, 'validate')]
        bst = xgb.train(params, train_matrix, params['n_best_trees'], watchlist)

        pos_proba = bst.predict(valid_matrix, ntree_limit=params['n_best_trees'])
        yvalidates.iloc[test_index, 0] = pos_proba
        yvalidates.iloc[test_index, 1] = np.log(pos_proba)

        print "====== cross-validated on {}-fold ======".format(index + 1)

    return yvalidates

class Learner(object):

    def __init__(self,params):
        self.params = params
        self.params['silent'] = 1
        self.params['objective'] = 'binary:logistic'

        self.tag = self.params['tag']
        self.stats_file = common.ModelStatsFile()

    def train(self):
        cv_results = xgb.cv(self.params, train_matrix,
                            num_boost_round=self.params["num_rounds"],
                            nfold=self.params.get('nfold', 5),
                            metrics=self.params['eval_metric'],
                            early_stopping_rounds=self.params["early_stopping_rounds"],
                            verbose_eval=True,
                            seed=seed)

        self.n_best_trees = cv_results.shape[0]
        self.params['n_best_trees'] = self.n_best_trees
        print "n_best_trees: {}".format(self.n_best_trees)

        self.gbt = xgb.train(self.params, train_matrix, self.n_best_trees, [(train_matrix, 'train')])

    def predict(self,matrix,yindex,dsname):
        ypred_probas = self.gbt.predict(matrix, ntree_limit=self.n_best_trees)

        result = pd.DataFrame({'{}_proba'.format(self.tag): ypred_probas,
                               '{}_log_proba'.format(self.tag): np.log(ypred_probas)},
                              index=yindex)
        result.to_csv('meta_features/{}_{}.csv'.format(self.tag,dsname),index_label='id')
        print "Model[{}] predicted on '{}' dataset".format(self.tag,dsname)

        return result

    def evaluate(self,valid_pred_result):
        yvalid_pred_probas = valid_pred_result.loc[:,'{}_proba'.format(self.tag)]
        yvalid_pred = (yvalid_pred_probas > 0.5).astype(int)

        valid_accuracy = accuracy_score(yvalid, yvalid_pred)
        valid_auc = roc_auc_score(yvalid, yvalid_pred_probas)

        self.stats_file.log(self.tag,valid_accuracy,valid_auc,str(self.params))

    def run(self):
        self.train()

        valid_pred_result = self.predict(valid_matrix,yvalid.index,'validate')
        self.evaluate(valid_pred_result)

        self.predict(test_matrix,ytest.index,'test')

        metafeatures = crossval_predict(self.tag,Xtrain,ytrain,self.params)
        metafeatures.to_csv("meta_features/{}_train.csv".format(self.tag),index_label="id")

    def close(self):
        self.stats_file.close()

if __name__ == "__main__":
    params = {}
    params['tag'] = 'xgb5'
    params['silent'] = 1
    params['objective'] = 'binary:logistic'  # output probabilities
    params['eval_metric'] = 'auc'
    params["num_rounds"] = 400
    params["early_stopping_rounds"] = 30
    params['min_child_weight'] = 10
    params['max_depth'] = 6
    params['eta'] = 0.1
    params["subsample"] = 0.5
    params["colsample_bytree"] = 0.5

    learner = Learner(params)
    learner.run()
    learner.close()