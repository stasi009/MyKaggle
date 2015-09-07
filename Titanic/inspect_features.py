
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import sklearn.preprocessing as skpreprocess
import sklearn.decomposition as skdecompose
import sklearn.manifold as skmanifold
import sklearn.linear_model as sklinear

titanic = pd.DataFrame.from_csv("train_processed.csv")
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]

Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

def plot2d(dimreducer,X,y):
    X2d = dimreducer.fit_transform(X)

    plt.figure()
    
    # have to change to ndarray, because now ndarray only "boolean index" when input is bool array
    # cannot be array-like object
    X2d_survived = X2d[np.asarray( y == 1),:]
    label = "%d survived"%(X2d_survived.shape[0])
    plt.scatter(X2d_survived[:,0],X2d_survived[:,1],label=label,alpha=0.5,color="b")

    X2d_died = X2d[np.asarray( y == 0),: ]
    label = "%d died"%(X2d_died.shape[0])
    plt.scatter(X2d_died[:,0],X2d_died[:,1],label=label,alpha=0.5,color="r")

pca = skdecompose.PCA(n_components=2)
plot2d(pca,Xtrain_scaled,ytrain)

mds = skmanifold.MDS(n_components=2)
plot2d(mds,Xtrain_scaled,ytrain)


def cross_tab_plot(survived,feature,figsize=(10,4)):
    crosstab = pd.crosstab(survived,feature)
    print "**************** counts"
    print crosstab

    print "**************** Row decomposition"
    row_crosstab = crosstab.apply(lambda r: r/r.sum(),axis=1)
    print row_crosstab

    print "**************** Column decomposition"
    col_crosstab = crosstab.apply(lambda c: c/c.sum()).T
    print col_crosstab

    # ------------------ plot
    fig, axes = plt.subplots(1, 2,figsize=figsize)
    row_crosstab.plot(kind="bar",stacked=True,ax= axes[0])
    col_crosstab.plot(kind="bar",stacked=True,ax= axes[1])


cross_tab_plot(titanic.Survived,titanic.Sex)


cuts =  [0,10,20,50,90] 
# cuts = np.arange(0,95,10)
age_bins = pd.cut(titanic.Age,cuts,right=False)
age_bins.value_counts()
cross_tab_plot(titanic.Survived,age_bins.astype("string"))

    



        
