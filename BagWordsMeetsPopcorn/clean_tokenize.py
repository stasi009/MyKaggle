
import itertools
import os.path
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
    print "----------- DONE -----------"

    dal.close()
    return index+1# return total


if __name__ == "__main__":
    raise Exception("!!! ATTENTION !!!\n"
                    "the script has run once. \n"
                    "all three sets have been inserted into database. \n"
                    "run this script again will insert duplicates into database.")

    # total = read_save_mongodb("datas/labeledTrainData.tsv",True,'train')

    # total = read_save_mongodb("datas/testData.tsv",False,'test')

    total = read_save_mongodb("datas/unlabeledTrainData.tsv",False,'unlabeled')

    print "{} reviews inserted into mongodb".format(total)






