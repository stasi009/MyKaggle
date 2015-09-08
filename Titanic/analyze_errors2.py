
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

titanic_train = pd.DataFrame.from_csv("train_processed.csv")
titanic_train = titanic_train.set_index("PassengerId")

titanic_train.Fare.describe()
titanic_train.Fare.hist(bins = 50)

comm_wrong_samples = pd.read_csv("common_wrong.csv",index_col="PassengerId")

condition = (comm_wrong_samples.Sex == "male") & (comm_wrong_samples.Pclass == 3)
wrong_male3 = comm_wrong_samples.loc[condition,:]
wrong_male3[["Age","SibSp","Parch","Ticket","Fare","Embarked"]]

# ******************************** #
cond_male3dead = (titanic_train.Sex == "male" ) & (titanic_train.Pclass == 3) & (titanic_train.Survived==0)
cond_male3live = (titanic_train.Sex == "male" ) & (titanic_train.Pclass == 3) & (titanic_train.Survived==1)

male3_live = titanic_train.loc[cond_male3live,:]

correct_live_male3 = ~(male3_live.index.isin(wrong_male3.index))
correct_live_male3 = male3_live[correct_live_male3]
correct_live_male3[["Age","SibSp","Parch","Ticket","Fare","Embarked"]]
