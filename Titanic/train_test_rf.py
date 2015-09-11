
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

feature_names = ["Pclass","Age","SibSp","Parch","Fare","IsMale","Ticket-4digit","Ticket-5digit","Ticket-6digit","GiveHelp","RecvHelp"]

def train():
    titanic_train = pd.read_csv("train_processed.csv",index_col="PassengerId")

    Xtrain = titanic_train[feature_names]
    ytrain = titanic_train["Survived"]

    param_dist = {"n_estimators":  sp_randint(1000,4800),                
                  "max_depth": [2,3, 4,5,6,7,8,9,None],              
                  "criterion": ["gini", "entropy"]}

    njobs = 4
    rf = RandomForestClassifier(oob_score=True,verbose=1,n_jobs=njobs)
    searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,n_iter=200,n_jobs=njobs)

    print "#################### search cv begins"
    searchcv.fit(Xtrain,ytrain)    
    print "#################### search cv ends"

    with open('rf.pkl', 'wb') as outfile:
        pickle.dump(searchcv.best_estimator_,outfile)
    print "*** RF saved into file"

    print "best score: ",searchcv.best_score_                                  
    print "best parameters: ",searchcv.best_params_

    rf = searchcv.best_estimator_
    print "feature importance: "
    print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names),reverse=True)

def test():
    with open("rf.pkl","rb") as infile:
        rf = pickle.load(infile)

    titanic_test = pd.read_csv("test_processed.csv",index_col="PassengerId")
    Xtest = titanic_test[feature_names]

    predictions = rf.predict(Xtest)
    submission = pd.DataFrame({
            "PassengerId": titanic_test.index,
            "Survived": predictions
        })
    submission.to_csv("submit_rf.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="train or test?")
    
    args = parser.parse_args()
    
    if args.job == "train":
        train()
    elif args.job == "test":
        test()
    else:
        raise ValueError("unknown command")
    