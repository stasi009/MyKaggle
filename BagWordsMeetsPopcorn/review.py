
from pymongo import MongoClient
from sentence import Sentence
import text_utility

class Review(object):

    StopWords = text_utility.make_stop_words()

    def __init__(self,id = None,text = None,is_positive=None):
        self.id = id
        self.sent = None if text is None else Sentence.from_raw(text,Review.StopWords,neg_mark=True)
        if self.sent is not None:
            self.sent.sentiment = is_positive

    def to_dict(self):
        return {'_id': self.id,'text': self.sent.raw,'words': self.sent.words,'is_positive': self.sent.sentiment }

    @staticmethod
    def from_dict(d):
        r = Review()
        r.id = d.get("_id",None)

        r.sent = Sentence()
        r.sent.raw = d.get('text',None)
        r.sent.words = d.get('words',None)
        r.sent.sentiment = d.get('is_positive',None)

        return r

class ReviewsDAL(object):

    def __init__(self,dbname = 'popcorn'):
        self._client = MongoClient()
        self._db = self._client[dbname]

    def close(self):
        self._client.close()

    def insert_many(self,colname,reviews):
        # cannot be iterator, but an non-empty list
        if len(reviews)>0:
            review_collection = self._db[colname]
            review_collection.insert_many([r.to_dict() for r in reviews])

    def load_one(self,colname,id):
        cursor = self._db[colname].find({'_id':id})
        d = list(cursor)
        assert len(d)==1,'id must be unique'
        return Review.from_dict(d[0])

    def load_ids(self,colname):
        # exclude id and raw text
        cursor = self._db[colname].find({}, {'_id':1})
        for d in cursor:
            yield d["_id"]

    def load_words(self,colname):
        """
        load reviews with just words
        """
        # exclude id and raw text
        cursor = self._db[colname].find({}, { 'text':0})
        for d in cursor:
            yield Review.from_dict(d)