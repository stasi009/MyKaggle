
import pickle
import pandas as pd
import numpy as np
import matplotlib.cm as colormap
from sklearn.preprocessing import LabelEncoder

class RawTrainData(object):

    def __init__(self,filename):
        train_csv = pd.read_csv(filename,index_col = "id")
        self.Xtrain = train_csv.iloc[:,xrange(93)]
        self.labels = train_csv.loc[:,"target"]

        self.labelencoder = LabelEncoder()
        self.ytrain = self.labelencoder.fit_transform(self.labels)

    def unique_labels(self):
        return self.labels.unique()

    def boolindex_by_label(self,label):
        return np.asarray(self.labels == label)

def dump_predictor(filename,learner):
    with open(filename, 'wb') as outfile:
        pickle.dump(learner,outfile)

def load_predictor(filename):
    with open(filename,"rb") as infile:
        return pickle.load(infile)

def colors_iterator(num_lables):
    return iter(colormap.rainbow(np.linspace(0, 1, num_lables)))

def cv_predict(predictor,Xtrain,ytrain):
    pass
