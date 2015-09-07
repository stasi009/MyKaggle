
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn.ensemble as skensemble

titanic = pd.DataFrame.from_csv("train_processed.csv")

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale"]
# feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","EmbarkC","EmbarkQ","EmbarkS"]
Xtrain = titanic[feature_names]
ytrain = titanic["Survived"]

criterion = "entropy"
rf = skensemble.RandomForestClassifier(n_estimators=200, criterion=criterion,oob_score=True)
rf.fit(Xtrain,ytrain)
rf.oob_score_

sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),reverse=True)

# ---------------------------------------------------- #
titanic_test = pd.DataFrame.from_csv("test_processed.csv")
Xtest = titanic_test[feature_names]

predictions = rf.predict(Xtest)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("submit_rf_fewfeature.csv", index=False)
