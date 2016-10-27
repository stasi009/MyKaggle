
import pandas as pd
import pymongo
from review import Review

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

import nltk
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def sentence_stream(colname):
    client = pymongo.MongoClient()
    collection = client.popcorn[colname]

    cursor = collection.find({}, {'words':0 })
    for d in cursor:
        yield Review.from_dict(d)

    client.close()

def predict(colname):
    sent_stream = sentence_stream(colname)

    n_correct = 0
    results = []
    for index,r in enumerate(sent_stream):
        ss = sid.polarity_scores(r.sent.raw)
        ss['id'] = r.id
        results.append(ss)

        score = ss['compound']
        if r.sent.sentiment is not None:
            if (score > 0 and r.sent.sentiment==1) or (score < 0 and r.sent.sentiment==0):
                n_correct += 1

        if index % 300 == 0:
            print '{} sentences processed'.format(index+1)

    df = pd.DataFrame(results)
    df.set_index('id',inplace=True)
    df.to_csv("meta_features/vader_{}.csv".format(colname),index_label='id')
    print "{} in total, {} is correct, correct ratio={}%".format(index+1,n_correct,n_correct*100.0/(index+1))

if __name__ == "__main__":
    predict("test")