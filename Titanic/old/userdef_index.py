
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def age_group(p):
    if p.Age > 0 and p.Age <= 14:
        return "child"
    elif p.Age > 50 and p.Age < 100:
        return "old"
    else:
        return "middle"

def give_help_index(p):
    agegrp = age_group(p)

    # children within this Age range have no ability to offer help
    if agegrp == "child":
        return 0

    # more relatives together, you have to offer more help
    index = 1 + p.SibSp + p.Parch # plus 1, to avoid 0

    if agegrp == "middle":
        index *= p.Age # during this period, the ability of giving help increase as age increase
    elif agegrp == "old":
        index /= p.Age # the older you are, the weaker you can offer help
    else:
        raise ValueError("Age out of range")

    if p.Sex == "male":
        index *=2 # men has the obligation to offer help

    return index

def receive_help_index(p):

    # more relatives you have, more possible to get help
    # younger you are, more possible to get help
    index = (1 + p.SibSp + p.Parch) / (1 + p.Age)

    agegrp = age_group(p)

    # no one cares about middle
    # children are easy to get help
    # old people are easy to get help, but not as much as children
    if agegrp == "child":
        index *= 10
    elif agegrp == "old":
        index *= 3

    if p.Sex == "female":
        index *= 2

    return index


def calculate_help_index(r):
    givehelp = give_help_index(r)
    recvhelp = receive_help_index(r)
    return pd.Series([givehelp,recvhelp],index=["GiveHelp","RecvHelp"])

titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")
train_help = titanic_train.apply(calculate_help_index,axis=1)

titanic_train = pd.concat([titanic_train,train_help],axis=1)
titanic_train.to_csv("train_processed.csv",index_label="PassengerId")

givehelp_grps = titanic_train.GiveHelp.groupby(titanic_train.Survived)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for survived,grp in givehelp_grps:
    # grp.hist(bins=50,ax=ax,label="Survived=%d"%survived)
    grp.plot(kind="kde",ax=ax,label="Survived=%d"%survived)
ax.legend(loc='best')

recvhelp_grps =  titanic_train.RecvHelp.groupby(titanic_train.Survived)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for survived,grp in recvhelp_grps:
    # grp.hist(bins=50,ax=ax,label="Survived=%d"%survived)
    grp.plot(kind="kde",ax=ax,label="Survived=%d"%survived)
ax.legend(loc='best')








titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
test_help = titanic_test.apply(calculate_help_index,axis=1)

titanic_test = pd.concat([titanic_test,test_help],axis=1)
titanic_test.to_csv("test_processed.csv",index_label="PassengerId")

