
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.decomposition import PCA,KernelPCA

import commfuncs

def reduce2d_plot(reducer,trainData):
    Xtrain_2d = reducer.fit_transform(trainData.Xtrain)

    plt.figure()
    unique_labels = trainData.unique_labels()
    clr_it = common.colors_iterator(len(unique_labels))
    for label in unique_labels:
        bindex = trainData.boolindex_by_label(label)
        x2d = Xtrain_2d[bindex]
        plt.scatter(x2d[:,0],x2d[:,1],label=label,color=next(clr_it),alpha=0.5)
    # plt.legend(loc="best")

trainData = commfuncs.RawTrainData("raw_datas/train.csv")
trainData.Xtrain.describe()
trainData.labels.value_counts().plot(kind="bar")

reduce2d_plot(PCA(n_components=2),trainData)

# Kernel PCA depends on similarity matrix which is N*N
# here we have memory issue when dealing with that large matrix
kernel_pca = KernelPCA(n_components=2,kernel='rbf', gamma=15)
reduce2d_plot(kernel_pca,trainData)



