
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

titanic_train["TicketGroup"] = "Others"

prefix = "A"
prefix = "PC"
prefix = "STON"
prefix = "SOTON"
prefix = "C"
flags = titanic_train.Ticket.map(lambda e: e.startswith(prefix))
titanic_train.loc[flags,"TicketGroup"] = prefix

flags = titanic_train.Ticket.map(lambda e: len(e)==7 and e.isdigit()  )
titanic_train.loc[flags,"TicketGroup"] = "7digit"


titanic_train.loc[titanic_train.TicketGroup == "Unknown","TicketGroup"] = "Others"

titanic_train.TicketGroup.value_counts()

titanic_train.to_csv("train_processed.csv",index_label="PassengerId")


male3 = titanic_train.loc[(titanic_train.Sex == "male") & (titanic_train.Pclass==3),:]
male3.Survived.value_counts()
pd.crosstab(male3.Survived,male3.TicketGroup,margins=True)

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
cross_tab_plot(male3.Survived,male3.TicketGroup,(15,4))



titanic_test = pd.DataFrame.from_csv("test_processed.csv")
titanic_test= titanic_test.set_index("PassengerId")
titanic_test["TicketGroup"] = "Others"

prefix = "W"
flags = titanic_test.Ticket.map(lambda e: e.startswith(prefix))
titanic_test.loc[flags,"TicketGroup"] = prefix

flags = titanic_test.Ticket.map(lambda e: len(e)==4 and e.isdigit()  )
titanic_test.loc[flags,"TicketGroup"] = "4digit"

titanic_test.loc[titanic_test.TicketGroup=="Others","Ticket"]

titanic_test.TicketGroup.value_counts()
titanic_test.to_csv("test_processed.csv",index_label="PassengerId")

