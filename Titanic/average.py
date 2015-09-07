
import numpy as np
import pandas as pd

def read_result(filename):
    results = pd.read_csv(filename)
    results = results.set_index("PassengerId")
    results = results.iloc[:,0]
    results[results == 0] = -1
    return results
    
lr_results = read_result("submission_lr.csv")
rf_entropy_results = read_result("submission_rf_entropy.csv")
rf_gini_results = read_result("submission_rf_gini.csv")

results = pd.concat([lr_results,rf_entropy_results,rf_gini_results],axis=1, keys=["lr","rf_entropy","rf_gini"])

flag = results.lr + results.rf_entropy + results.rf_gini
flag = (flag > 0).astype(int)

flag.to_csv("submission_average.csv")