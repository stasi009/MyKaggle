
import cPickle
import numpy as np
import pandas as pd
from sentence import Sentence
import text_utility

StopWords = text_utility.make_stop_words()

def tokenize_merge(row):
    allwords = []
    for text in row.iloc[1:].dropna():
        text = text.lstrip("b\'").lstrip("b\"").lstrip("b\'''")
        s = Sentence.from_raw(text,StopWords,neg_mark=True)
        allwords += s.words

    print allwords# show progress
    return allwords

def load_tokenize():
    alldata = pd.read_csv("datas/Combined_News_DJIA.csv")
    alldata['Date'] = pd.to_datetime(alldata.Date)
    alldata.set_index('Date',inplace=True)

    allwords = alldata.apply(tokenize_merge,axis=1)
    return pd.concat( [alldata.loc[:,'Label'],allwords],axis=1, keys=['label','words'])

if __name__ == "__main__":
    df = load_tokenize()

    cutoff_dt = pd.to_datetime('2015-01-01')
    for_training = df.index < cutoff_dt

    train_set = df.loc[for_training,:]
    test_set = df.loc[np.logical_not(for_training),:]

    with open("datasets.pkl",'wb') as outf:
        cPickle.dump(train_set,outf)
        cPickle.dump(test_set,outf)

    print "====== Tokenized and Dumped ======"


