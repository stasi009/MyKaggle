
import cPickle
import numpy as np
import pandas as pd

def simple_dump(filename, *objects):
    with open(filename, 'wb') as outfile:
        for obj in objects:
            cPickle.dump(obj, outfile)

def simple_load(filename,n_objs=1):
    objects = []
    with open(filename, "rb") as infile:
        for index in xrange(n_objs):
            objects.append(cPickle.load(infile))
        return objects

def can_predict_probability(predictor):
    try:
        predictor.__getattribute__('predict_proba')
        return True
    except AttributeError:
        return False

def predict_proba_or_label(predictor,X,index,prefix):
    try:
        y_probas = predictor.predict_proba(X)
        pos_proba = y_probas[:, 1]  # probability for label=1 (Positive)

        return pd.DataFrame({'{}_proba'.format(prefix): pos_proba,
                             '{}_log_proba'.format(prefix): np.log(pos_proba)},
                            index=index)
    except AttributeError:
        y_label = predictor.predict(X)
        return pd.DataFrame({'{}_label'.format(prefix): y_label},index=index)



