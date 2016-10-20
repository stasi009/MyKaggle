
import itertools
import random
import os.path
from pymongo import MongoClient
from bs4 import BeautifulSoup
from review import Review,ReviewsDAL
from sentence import Sentence

def reviews_stream(filename,labeled):
    with open(filename,'rt') as inf:
        # read and discard the first line, which is header
        _ = inf.readline()

        for line in inf:
            segments = [s.strip(' \r\n\"') for s in line.split('\t')]
            assert (len(segments) == 3 if labeled else 2)

            id = segments[0]

            txt = segments[2 if labeled else 1]
            txt = BeautifulSoup(txt).get_text()  # Remove HTML

            is_positive = int(segments[1]) if labeled else None
            yield Review(id, txt, is_positive)

def read_save_mongodb(filename,labeled,colname,buffersize=300):
    r_stream = reviews_stream(filename,labeled)
    dal = ReviewsDAL()

    buffer = []
    for index,review in enumerate(r_stream):
        if index % buffersize == 0:
            dal.insert_many(colname,buffer)
            del buffer[:] # clear
            print "{} reviews saved into mongo[{}]".format(index,colname)

        buffer.append(review)

    dal.insert_many(colname,buffer)
    dal.close()

    print "----------- DONE -----------"
    print "totally {} reviews inserted into mongodb[{}]".format(index+1,colname)

def split_train_validation():
    """
    load samples from 'train' collection,
    draw some samples out, to use as validation set
    remove them from training set and insert those samples into 'validation' collection
    """
    random.seed(999)
    valid_ratio = 0.3

    dal = ReviewsDAL()
    train_ids = list(dal.load_ids("train"))
    total_train = len(train_ids)
    print "originally, there are {} reviews in train set".format(total_train)

    valid_ids = random.sample(train_ids,int(total_train * valid_ratio))
    print "randomly draw {} samples to use as validation".format(len(valid_ids))

    train_collect = dal._db['train']
    valid_collect = dal._db['validate']
    for index,valid_id in enumerate(valid_ids):
        # load from train collection
        cursor = train_collect.find({'_id':valid_id})
        review_dict = next(cursor)

        # insert into validation collection
        valid_collect.insert_one(review_dict)

        # remove from train collection
        result = train_collect.delete_one({'_id':valid_id})
        assert result.deleted_count == 1

        #
        if index % 100 == 0:
            print "{} reviews transferred from train to validation".format(index+1)
    print "*** totally {} reviews transferred from train to validation ***".format(index+1)

    print "now, train set has {} reviews".format(train_collect.count({}))
    print "now, validation set has {} reviews".format(valid_collect.count({}))

if __name__ == "__main__":
    raise Exception("!!! ATTENTION !!!\nthe script has run once. \nrun this script again will insert duplicates into database.")

    # split_train_validation()

    # read_save_mongodb("datas/labeledTrainData.tsv",True,'train')

    # read_save_mongodb("datas/testData.tsv",False,'test')

    # read_save_mongodb("datas/unlabeledTrainData.tsv",False,'unlabeled')





