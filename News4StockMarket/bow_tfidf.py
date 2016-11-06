
import itertools
import logging
import cPickle
import pandas as pd
from gensim import corpora,models,matutils

def load_datasets():
    with open("datasets.pkl",'rb') as inf:
        train_set =  cPickle.load(inf)
        test_set = cPickle.load(inf)
        return (train_set,test_set)

def build_dictionary():
    train_set, _ = load_datasets()
    words_stream = (train_set.iloc[r,1] for r in xrange(train_set.shape[0]) )

    dictionary = corpora.Dictionary(words_stream)
    dictionary.save('headlines.dict')  # store the dictionary, for future reference
    dictionary.save_as_text("headlines_dict.txt", sort_by_word=False)

    print "======== Dictionary Generated and Saved ========"

def clean_dict_save():
    dictionary = corpora.Dictionary.load('headlines.dict')
    print "originally, there are {} tokens".format(len(dictionary))

    dictionary.filter_extremes(no_below=5, no_above=0.8)
    print "after filtering too rare, there are {} tokens".format(len(dictionary))

    dictionary.save('headlines.dict')
    # sort by decreasing doc-frequency
    dictionary.save_as_text("headlines_dict.txt", sort_by_word=False)
    print "##################### dictionary is cleaned, shrinked and saved"

def __bow(dictionary,df,tag):
    bow_stream = (dictionary.doc2bow(df.iloc[r, 1]) for r in xrange(df.shape[0]))
    corpora.MmCorpus.serialize("{}.bow".format(tag), bow_stream)

    df.loc[:,['label']].to_csv("{}_labels.csv".format(tag),index_label='date')

def build_bow_save():
    dictionary = corpora.Dictionary.load('headlines.dict')
    train_set, test_set = load_datasets()

    __bow(dictionary,train_set,'train')
    __bow(dictionary,test_set,'test')

    print "!!! DONE !!!"

def build_tfidf_save():
    dictionary = corpora.Dictionary.load("headlines.dict")
    model = models.TfidfModel(id2word=dictionary,dictionary=dictionary, normalize=True)
    model.save("headlines.tfidf_model")
    print 'TF-IDF model generated and saved.'

    train_set,test_set = load_datasets()
    for (df,tag) in itertools.izip([train_set,test_set],['train','test']):
        print "\n=========== BOW[{}] to TF-IDF, ......".format(tag)

        bow_stream = corpora.MmCorpus('{}.bow'.format(tag))
        tfidf_stream = model[bow_stream]
        corpora.MmCorpus.serialize('{}.tfidf'.format(tag), tfidf_stream)

        print "=========== TF-IDF[{}] saved ===========".format(tag)

    print "!!! DONE !!!"

def load_tfidf(tag):
    corpus = corpora.MmCorpus('{}.tfidf'.format(tag))
    # with documents as columns
    X = matutils.corpus2csc(corpus,printprogress=1)
    # transpose to make each document a row
    X = X.T

    y = pd.read_csv("{}_labels.csv".format(tag),index_col='date')
    y = y.iloc[:,0]# DataFrame to Series

    return X,y

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build_dictionary()
    # clean_dict_save()
    # build_bow_save()
    build_tfidf_save()