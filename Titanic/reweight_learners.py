
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.base import clone as skclone
import sklearn.cross_validation as skcv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import common

# ***************************** load train data
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit"]
train_df = pd.read_csv("train_processed.csv",index_col="PassengerId")

# ***************************** split the train data into train-set and validation set
Xtrain, Xvalidate, ytrain, yvalidate = skcv.train_test_split(train_df[feature_names], train_df["Survived"], test_size=0.25)

# ***************************** scale train data
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xvalidate_scaled = scaler.transform(Xvalidate)

# ***************************** fit base estimators
# first element is the method's name
# second element is whether it need scaling
methods = [("lr",True),("svc",True),("knn",True),("rf",False),("gbdt",False)]

Estimator = collections.namedtuple("Estimator",("estimator","name","need_scale"))
def fit_estimator(name,need_scale,y,X,scaledX):
    temp = common.load_predictor("%s.pkl"%name)
    estimator = skclone(temp)

    if need_scale:
        estimator.fit(scaledX,y)
    else:
        estimator.fit(X,y)

    return Estimator(estimator,name,need_scale)

base_estimators = [ fit_estimator(name,need_scale,ytrain,Xtrain,Xtrain_scaled) for (name,need_scale) in methods]

# ***************************** generate predictions on validation sets
def predict_features(base_estimators,X,scaledX):
    basepredicts = [ estimator.estimator.predict(scaledX) if estimator.need_scale else estimator.estimator.predict(X) \
        for estimator in base_estimators]
    return pd.DataFrame(np.asarray(basepredicts).T,
                        index = X.index,
                        columns = [estimator.name  for estimator in base_estimators])

# ***************************** fit advanced features to validation target 
validate_basepredicts = predict_features(base_estimators,Xvalidate,Xvalidate_scaled)
lrcv = LogisticRegressionCV(Cs=30,cv=10)
lrcv.fit(validate_basepredicts,yvalidate)
lrcv.score(validate_basepredicts,yvalidate)
common.make_coefs_frame(validate_basepredicts.columns,lrcv.coef_.ravel())

# fit again with whole data
basepredict_lr = LogisticRegression(C = lrcv.C_[0])
basepredict_lr.fit(validate_basepredicts,yvalidate)
basepredict_lr.score(validate_basepredicts,yvalidate)
common.make_coefs_frame(validate_basepredicts.columns,basepredict_lr.coef_.ravel())

# ***************************** test
test_df = pd.read_csv("test_processed.csv",index_col="PassengerId")
Xtest = test_df[feature_names]
Xtest_scaled = scaler.transform(Xtest)

test_basepredict = predict_features(base_estimators,Xtest,Xtest_scaled)
final_predictions = basepredict_lr.predict(test_basepredict)
common.make_submission(Xtest.index,final_predictions,"submit_reweight_learners.csv")





