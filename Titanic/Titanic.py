
import pandas as pd
import numpy as np

titanic = pd.read_csv("train.csv")
titanic.describe()
np.median(titanic.Age)

# fill in missing values
titanic.Age = titanic.Age.fillna(titanic.Age.median())

# convert the Sex attribute into numeric attribute
titanic["Sex"].unique()
titanic.loc[ titanic["Sex"] == "male","Sex"] = 0
titanic.loc[ titanic["Sex"] == "female","Sex"] = 1

titanic["Embarked"].unique()


embarked = titanic.Embarked
embarked.unique()
embarked.groupby(lambda index:embarked[index]).size()




