
from gensim import corpora,models,matutils

def test_dictionary():
    dictionary = corpora.Dictionary.load('headlines.dict')
    print "originally, there are {} tokens".format(len(dictionary))

if __name__ == "__main__":
    pass