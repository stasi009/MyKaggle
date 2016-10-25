
"""
perform some Explory Data Analysis (EDA) in this script
"""
from pymongo import MongoClient
import bow_tfidf
from review import ReviewsDAL

def check_examples():
    dal = ReviewsDAL()
    r_stream = dal.load_words("train")
    for index in xrange(10):
        print "******* {} *******".format(index+1)
        r = next(r_stream)
        print r.sent.words

def calc_pos_ratio(colname):
    client = MongoClient()
    collect = client.popcorn[colname]
    # '_id':1, just put all records in the same group
    cursor = collect.aggregate([{"$group" : { "_id":1,
                                              "total":{"$sum":1},
                                              'n_pos':{'$sum':'$is_positive'},
                                              'pos_ratio': {'$avg': '$is_positive'}
                                              }}])
    return next(cursor)

def check_dictionary():
    clean_dict = bow_tfidf.clean_dictionary(no_below=10,no_above=0.8,keep_n=15000)

    # sort by decreasing doc-frequency
    clean_dict.save_as_text("processed/dictionary.txt",sort_by_word=False)


