
import numpy as np
import pandas as pd

def read_result(filename):
    results = pd.read_csv(filename,index_col="PassengerId")
    results = results.iloc[:,0]
    results[results == 0] = -1
    return results
    
lr_results = read_result("submit_lr.csv")
rf_results = read_result("submit_rf.csv")
svc_results = read_result("submit_linear_svc.csv")

results = pd.concat([lr_results,rf_results,svc_results],axis=1, keys=["lr","rf","svc"])

flag = results.lr + results.rf + results.svc
flag = (flag > 0).astype(int)

flag.to_csv("submit_average.csv")