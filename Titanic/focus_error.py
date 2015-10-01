
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import common

# ***************************** load train data
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit"]
train_df = pd.read_csv("train_processed.csv",index_col="PassengerId")

Xtrain = train_df[feature_names]
ytrain = train_df["Survived"]

# ***************************** scale train data
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

# ***************************** predict on train data
methods = [("lr",True),("svc",True),("knn",True),("rf",False),("gbdt",False)]

def fit_train_data(allresults,name,needscale):
    traindata = Xtrain_scaled if needscale else Xtrain

    estimator = common.load_predictor("%s.pkl"%name)
    print "[%s] training score: %f"%(name,estimator.score(traindata,ytrain))

    predictions = estimator.predict(traindata)
    allresults.append(predictions)

all_train_results = []
for (name,needscale) in methods:
    fit_train_data(all_train_results,name,needscale)

all_train_results = pd.DataFrame(np.asarray(all_train_results).T ,
                                 index = Xtrain.index,
                                 columns = [ m for m,_ in methods])


    

