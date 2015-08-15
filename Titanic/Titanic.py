
import pandas as pd
import numpy as np

titanic = pd.read_csv("train.csv")

# fill in missing values
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# convert the Sex attribute into numeric attribute
titanic["Sex"].unique()
titanic.loc[ titanic["Sex"] == "male","Sex"] = 0
titanic.loc[ titanic["Sex"] == "female","Sex"] = 1

titanic["Embarked"].unique()


embarked = titanic["Embarked"]
embarked.unique()

grps = embarked.groupby(lambda index:embarked[index])
for index,(key,grp) in enumerate(grps):
    print "[%d]: %s has %d"%(index+1,key,len(grp))


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                             columns = ["a","b","c"],
                             index = ["record1","record2","record3"])