
import pickle

import numpy as np
import pandas as pd

def logcounts_to_counts(x):
    return np.exp(x) - 1

def dump_predictor(filename,learner):
    with open(filename, 'wb') as outfile:
        pickle.dump(learner,outfile)

def load_predictor(filename):
    with open(filename,"rb") as infile:
        return pickle.load(infile)

def make_submission(testindex,predictions,filename):
    submission = pd.DataFrame({
        "PassengerId": testindex,
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)