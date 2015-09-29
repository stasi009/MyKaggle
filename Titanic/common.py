
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def survive_cross_tab(df,featName,figsize=(10,4)):
    crosstab = pd.crosstab(df.Survived,df[featName])
    print "**************** counts"
    print crosstab

    print "**************** Survive across %s"%featName
    row_crosstab = crosstab.apply(lambda r: r/r.sum(),axis=1)
    print row_crosstab

    print "**************** %s across Survive"%featName
    col_crosstab = crosstab.apply(lambda c: c/c.sum()).T
    print col_crosstab

    # ------------------ plot
    fig, axes = plt.subplots(1, 2,figsize=figsize)
    row_crosstab.plot(kind="bar",stacked=True,ax= axes[0])
    col_crosstab.plot(kind="bar",stacked=True,ax= axes[1])

def make_submission(testindex,predictions,filename):
    submission = pd.DataFrame({
        "PassengerId": testindex,
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)
