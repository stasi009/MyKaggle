
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.preprocessing as skpreprocess
import sklearn.decomposition as skdecompose
import sklearn.manifold as skmanifold
import sklearn.linear_model as sklinear
import sklearn.cross_validation as skcv

titanic = pd.DataFrame.from_csv("train_processed.csv")
feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]

Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

# scale the train data
scaler = skpreprocess.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

# fit a LR classifier
lr = sklinear.LogisticRegression()
lr.fit(Xtrain_scaled,ytrain)
train_accuracy = lr.score(Xtrain_scaled,ytrain)

# predict
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]
Xtest_scaled = scaler.transform(Xtest)

predictions = lr.predict(Xtest_scaled)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("submission.csv", index=False)









# visualize in 2D plane by PCA
pca = skdecompose.PCA(n_components=2)
Xtrain_2d = pca.fit_transform(Xtrain_scaled)
Xtrain_2d = pd.DataFrame(Xtrain_2d,columns=["pc1","pc2"])

survive_grps = Xtrain_2d.groupby(ytrain)
plt.figure()
for key,grp in survive_grps:
    label = "Survived" if key==1 else "Dead"
    color = "b" if key==1 else "r"
    plt.scatter(grp.pc1,grp.pc2,label=label,alpha=0.5,color=color)

# visualize in 2D plane by MDS
mds = skmanifold.MDS(n_components=2)
Xtrain_2d = mds.fit_transform(Xtrain_scaled)
Xtrain_2d = pd.DataFrame(Xtrain_2d,columns=["pc1","pc2"])

survive_grps = Xtrain_2d.groupby(ytrain)
plt.figure()
for key,grp in survive_grps:
    label = "Survived" if key==1 else "Dead"
    color = "b" if key==1 else "r"
    plt.scatter(grp.pc1,grp.pc2,label=label,alpha=0.5,color=color)






