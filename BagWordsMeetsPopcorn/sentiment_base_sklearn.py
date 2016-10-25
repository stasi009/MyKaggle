
import numpy as np
import pandas as pd
from gensim import corpora,matutils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import bow_tfidf

def crossval_predict(predictor, X, y, result, prefix, n_cv=3):
    if not np.array_equal( predictor.classes_, [0, 1]):
        raise Exception("classes labels NOT match")

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)
    print "totally {} samples, divided into {} folds".format(n_samples, n_cv)

    if result == "label":
        datas = np.full((n_samples, 1), np.NaN)
        header = "{}_label".format(prefix)
        yvalidates = pd.DataFrame(datas, columns=[header])
    elif result == "probability":
        datas = np.full((n_samples, 2), np.NaN)
        headers = ["{}_{}".format(prefix, t) for t in ["proba", "log_proba"]]
        yvalidates = pd.DataFrame(datas, columns=headers)

    folds = StratifiedKFold(y, n_folds=n_cv,shuffle=True)
    for index, (train_index, test_index) in enumerate(folds):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        predictor.fit(Xtrain, ytrain)
        if result == "label":
            yvalidates.iloc[test_index, 0] = predictor.predict(Xtest)
        elif result == "probability":
            ytest_probas = predictor.predict_proba(Xtest)
            pos_proba = ytest_probas[:, 1]  # probability for label=1 (Positive)

            yvalidates.iloc[test_index, 0] = pos_proba
            yvalidates.iloc[test_index, 1] = np.log(pos_proba)
        else:
            raise Exception("unknown result type")

        print "====== cross-validated on {}-fold ======".format(index + 1)

    return yvalidates

def run_one_model(predictor,tag):
    Xtrain,ytrain = bow_tfidf.load_tfidf("train")
    print "train set's positive ratio: {}%".format(ytrain.mean()*100)

    # ---------- generate meta features
    yvalidates =

    # ---------- evaluate on validation set
    Xvalid,yvalid = bow_tfidf.load_tfidf("validate")
    print "validation set's positive ratio: {}%".format(yvalid.mean()*100)





