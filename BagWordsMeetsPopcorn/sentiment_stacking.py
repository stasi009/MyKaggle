
import pandas as pd
import numpy as np
import glob
import xgboost as xgb
from sklearn.metrics import accuracy_score,roc_auc_score
import common
seed = 999

def combine_meta_features(dsname):
    files = glob.glob("meta_features/*_{}.csv".format(dsname))
    dfs = [pd.read_csv(fname,index_col='id') for fname in files]

    # read target
    dfs.append( pd.read_csv("vsm/{}_meta.csv".format(dsname),index_col='id') )

    combined_df = pd.concat(dfs,axis=1)

    # make sure all meta-features have the same order in features
    combined_df.sort_index(axis=1,inplace=True)
    combined_df.to_csv('stacking/meta_{}.csv'.format(dsname),index_label='id')

    print "combined [{}] has shape: {}".format(dsname,combined_df.shape)

def make_matrix(dsname,return_index=False):
    df = pd.read_csv('stacking/meta_{}.csv'.format(dsname),index_col='id',na_values=['None'])
    y = df['sentiment']

    del df['sentiment']
    matrix = xgb.DMatrix(df,y)

    return (matrix,df.index) if return_index else matrix

train_matrix = make_matrix("train")
valid_matrix = make_matrix("validate")
test_matrix,test_index = make_matrix('test',True)

class Learner(object):

    def __init__(self,params):
        self.params = params
        self.params['silent'] = 1
        self.params['objective'] = 'binary:logistic'
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
        print "n_best_trees: {}".format(self.n_best_trees)

        self.gbt = xgb.train(self.params, train_matrix, self.n_best_trees, [(train_matrix, 'train')])
        print "-------- finish training --------"

    def predict_validation(self):
        yvalid_pred_probas = self.gbt.predict(valid_matrix, ntree_limit=self.n_best_trees)
        yvalid_pred = (yvalid_pred_probas > 0.5).astype(int)

        valid_accuracy = accuracy_score(valid_matrix.get_label(), yvalid_pred)
        valid_auc = roc_auc_score(valid_matrix.get_label(), yvalid_pred_probas)
        self.stats_file.log('stacking',valid_accuracy,valid_auc,str(self.params))

    def predict_test(self):
        ytest_pred_probas = self.gbt.predict(test_matrix, ntree_limit=self.n_best_trees)
        ytest_pred = (ytest_pred_probas > 0.5).astype(int)

        df = pd.DataFrame({'sentiment': ytest_pred},index=test_index)
        df.to_csv("stacking/submission.csv",index_label='id')

    def run(self):
        self.train()
        self.predict_validation()
        self.predict_test()

    def close(self):
        self.stats_file.close()

if __name__ == "__main__":
    params = {'colsample_bytree': 0.6,
              'silent': 1,
              'eval_metric': 'error',
              'num_rounds': 300,
              'min_child_weight': 30,
              'subsample': 0.6,
              'eta': 0.05,
              'early_stopping_rounds': 30,
              'objective': 'binary:logistic',
              'max_depth': 6}
    learner = Learner(params)
    learner.run()
    learner.close()


