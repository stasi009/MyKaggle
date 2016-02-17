
import numpy as np
import pandas as pd
import xgboost as xgb
import commfuncs

def save_train_test_dmatrix():
    """
    load and save to file, in order to save preprocessing time
    """
    trainData = commfuncs.RawTrainData("raw_datas/train.csv")
    xg_train = xgb.DMatrix( trainData.Xtrain, label=trainData.ytrain)
    xg_train.save_binary("train.dmatrix")

    testdata = pd.read_csv("raw_datas/test.csv",index_col = "id")
    xg_test = xgb.DMatrix(testdata)
    xg_test.save_binary("test.dmatrix")


