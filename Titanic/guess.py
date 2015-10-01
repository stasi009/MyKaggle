
import numpy as np
import pandas as pd
import common

def read_result(filename):
    results = pd.read_csv(filename,index_col="PassengerId")
    results = results.iloc[:,0]
    results[results == 0] = -1
    return results

methods = ["lr","svc","rf","gbdt","knn"]
results = [read_result(filename) for filename in ("submit_%s.csv"%m for m in methods)]
results = pd.concat(results,axis=1, keys=methods)

consensus = results.sum(axis=1)
consensus = (consensus == 5) | (consensus == -5)

consensus.value_counts()

consensus_result = results.loc[consensus,:].sum(axis=1)

nonconsensus_result = results.loc[-consensus,:].sum(axis=1)

neg_nonconsensus_result = -1 * nonconsensus_result

guess = pd.concat([consensus_result,neg_nonconsensus_result])

guess = (guess > 0).astype(int)

common.make_submission(guess.index,guess,"submit_guess.csv")