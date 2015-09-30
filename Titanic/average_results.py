
import numpy as np
import pandas as pd
import common

def read_result(filename):
    results = pd.read_csv(filename,index_col="PassengerId")
    results = results.iloc[:,0]
    results[results == 0] = -1
    return results

methods = ["lr","svc","rf","gbdt"]
results = [read_result(filename) for filename in ("submit_%s.csv"%m for m in methods)]
results = pd.concat(results,axis=1, keys=methods)

majority = results.sum(axis=1)
majority = (majority > 0).astype(int)

common.make_submission(majority.index,majority,"submit_average.csv")