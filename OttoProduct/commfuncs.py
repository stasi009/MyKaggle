
import pickle
import pandas as pd
import numpy as np

def dump_predictor(filename,learner):
    with open(filename, 'wb') as outfile:
        pickle.dump(learner,outfile)

def load_predictor(filename):
    with open(filename,"rb") as infile:
        return pickle.load(infile)

def cv_predict(predictor,Xtrain,ytrain):
    pass
