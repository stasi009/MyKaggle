
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

wrong_lr = pd.read_csv("wrong_lr.csv",index_col="PassengerId")


featurenames = ["Survived","Pclass","Age","Sex","SibSp","Parch","Fare","Ticket"]


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

female = wrong_lr.loc[(wrong_lr.Sex == "female"),featurenames]
cross_tab_plot(female.Survived,female.Pclass)

male = wrong_lr.loc[(wrong_lr.Sex == "male"),featurenames]
cross_tab_plot(male.Survived,male.Pclass)

wrong_lr.Pclass.value_counts()

temp = wrong_lr.loc[wrong_lr.Pclass==3,featurenames]
cross_tab_plot(temp.Survived,temp.Sex)

wrong_lr.loc[(wrong_lr.Sex == "male") & (wrong_lr.Pclass==3),featurenames]


