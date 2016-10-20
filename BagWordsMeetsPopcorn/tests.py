
import nltk
from pymongo import MongoClient
from sentence import Sentence
from review import Review,ReviewsDAL
import text_utility

def test_sentence():
    stopwords = text_utility.make_stop_words()

    texts = [   "can't is a contraction",
                "she isn't my wife any more",
                "I am not in USA right now",
                "I'm a Chinese",
                "1630 NE Valley Rd, Pullman, WA, 99163, Apt X103",
                "I should've done that thing I didn't do",
                "I don't love her any more",
                "I want to divorce without hesitation",
                "bye, Pullman, bye, USA"]

    for index,text in enumerate(texts):
        sent = Sentence.from_raw(text,stopwords,True)
        print "\n******************** {}".format(index+1)

        print sent.raw
        print "===>"
        print sent.words

def test_review():
    txt = "I don't love her any more"
    r = Review('abc',txt)
    d = r.to_dict()

    cpyr = Review.from_dict(d)
    print cpyr.to_dict()

def test_load_review_words():
    dal = ReviewsDAL()
    r_stream = dal.load_reviews_words("unlabeled")

    for index in xrange(10):
        review = next(r_stream)
        print "*************** {} ***************".format(index+1)
        print "sentiment: {}".format(review.sent.sentiment)
        print "words: {}".format(review.sent.words)

    dal.close()

def test_load_review_raw():
    client = MongoClient()
    collection = client['popcorn']['unlabeled']

    cursor = collection.find({},{'text':1})
    for index in xrange(10):
        d = next(cursor)
        print d['text']+"\n"

    client.close()

def test_load_ids():
    dal = ReviewsDAL()
    ids = dal.load_ids("train")

