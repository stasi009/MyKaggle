
import logging
import os
import re
import itertools
import pandas as pd
import text_utility
from review import ReviewsDAL
from gensim import corpora,models,matutils
from topic_model import print_topic_distribution

def extend_sentiment_words_save():
    # since we ignore header, so we only skip 34 rows, not 35 rows
    poswords = pd.read_csv('datas/positive-words.txt',skiprows=34, header=None).values.flatten().tolist()
    print 'totally {} positive words'.format(len(poswords))

    negwords = pd.read_csv('datas/negative-words.txt',skiprows=34, header=None).values.flatten().tolist()
    print "totally {} negative words".format(len(negwords))

    neg_poswords = [w+text_utility.NEG_SUFFIX for w in poswords]
    neg_negwords = [w+text_utility.NEG_SUFFIX for w in negwords]

    poswords += neg_negwords
    negwords += neg_poswords

    pd.Series(poswords).to_csv("datas/extend_poswords.txt",index=False)
    pd.Series(negwords).to_csv("datas/extend_negwords.txt",index=False)

PosReview = 'POSREVIEW'
NegReview = 'NEGREVIEW'

class WordsStream(object):
    PosWords = frozenset( pd.read_csv('datas/extend_poswords.txt',header=None).values.flatten() )
    NegWords = frozenset( pd.read_csv('datas/extend_negwords.txt',header=None).values.flatten() )

    def __init__(self,colname):
        self._colname = colname

    def append_sentiment_words(self,words):
        extra_words = []
        for w in words:
            if w in WordsStream.PosWords:
                extra_words.append(PosReview)
                # print "+++ {}: Positive".format(w)

            if w in WordsStream.NegWords:
                extra_words.append(NegReview)
                # print "--- {}: Negative".format(w)

        words += extra_words

    def stream(self):
        dal = ReviewsDAL()
        review_stream = dal.load_words(self._colname)
        for index, r in enumerate(review_stream):

            self.append_sentiment_words(r.sent.words)
            yield r.sent.words

            if index % 300 == 0:
                print "{} examples loaded from mongodb[{}]".format(index + 1, self._colname)

        dal.close()

def test_words_stream():
    train_words_stream =  WordsStream('train').stream()
    for index,words in enumerate( train_words_stream):
        if index > 10:
            return

        print "*********** [{}] ***********".format(index+1)
        print words

def build_dictionary():
    train_words_stream =  WordsStream('train').stream()
    unlabeled_words_stream = WordsStream('unlabeled').stream()
    wstream = itertools.chain(train_words_stream, unlabeled_words_stream)

    dictionary = corpora.Dictionary(wstream)
    dictionary.save('topic_model/codeword_dictionary.dict')  # store the dictionary, for future reference
    print "======== Dictionary Generated and Saved ========"

def dict2text():
    dictionary = corpora.Dictionary.load("topic_model/codeword_dictionary.dict")
    dictionary.save_as_text("topic_model/codeword_dictionary.txt", sort_by_word=False)

class DictCleaner(object):
    def __init__(self):

        self._invalid_pattern = re.compile(r"[^a-zA-Z_\-]")

        self._extra_stop_words = set()
        # for a dataset containing movie reviews, 'movie' and 'film' are too frequent
        for w in ['movie', 'film']:
            self._extra_stop_words.add(w)
            self._extra_stop_words.add(w + text_utility.NEG_SUFFIX)

    def is_token_invalid(self,token):
        if self._invalid_pattern.search(token) is not None:
            return True

        if token in self._extra_stop_words:
            return True

        return False

    def clean(self,no_below=5, keep_n=100000):
        dictionary = corpora.Dictionary.load('topic_model/codeword_dictionary.dict')
        print "originally, there are {} tokens".format(len(dictionary))

        # !!! we cannot filter out 'too frequent'
        # !!! because 'POSREVIEW' and 'NEGREVIEW' are just two most frequent words
        dictionary.filter_extremes(no_below=no_below,no_above=1,keep_n=keep_n)
        print "after filtering too rare, there are {} tokens".format(len(dictionary))

        # filter out invalid tokens
        invalid_tokenids = [id for token,id in dictionary.token2id.viewitems()
                            if self.is_token_invalid(token) ]
        print "there are {} tokens are invalid".format(len(invalid_tokenids))

        dictionary.filter_tokens(bad_ids = invalid_tokenids)
        print "after filtering invalid, there are {} tokens".format(len(dictionary))

        return dictionary

def clean_dict_save():
    cleaner = DictCleaner()
    clean_dict = cleaner.clean(no_below=10,keep_n=30000)

    clean_dict.save('topic_model/codeword_dictionary.dict')
    # sort by decreasing doc-frequency
    clean_dict.save_as_text("topic_model/codeword_dictionary.txt", sort_by_word=False)

    print "dictionary is cleaned, shrinked and saved"

def build_bow_save():
    dictionary = corpora.Dictionary.load('topic_model/codeword_dictionary.dict')

    colnames = ['train','unlabeled']
    for colname in colnames:
        print "\n=========== collection[{}] to BOW, ......".format(colname)

        wstream = WordsStream(colname)
        bow_stream = (dictionary.doc2bow(words) for words in wstream.stream())

        target_file = "topic_model/codeword_{}.bow".format(colname)
        corpora.MmCorpus.serialize(target_file, bow_stream)

        print "=========== BOW[{}] saved ===========".format(colname)

    print "!!! DONE !!!"

def run_lda(n_topics):
    dictionary = corpora.Dictionary.load("topic_model/codeword_dictionary.dict")
    train_bow = corpora.MmCorpus('topic_model/codeword_train.bow')
    unlabeled_bow = corpora.MmCorpus('topic_model/codeword_unlabeled.bow')

    # model = models.LdaMulticore(train_bow, id2word=dictionary, num_topics=n_topics,passes=3)
    model = models.LdaModel(unlabeled_bow, id2word=dictionary, num_topics=n_topics,passes=2)
    print "======== LDA built on train set ========"

    # model.update(unlabeled_bow)
    # print "======== LDA updated on unlabeled set ========"

    # --------------- save result
    tag = 'codeword_popcorn'
    model_name = os.path.join("topic_model",tag+".lda_model")
    model.save(model_name)

    topic_name = os.path.join("topic_model",tag+"_topics.txt")
    print_topic_distribution(model,topic_name)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # test_words_stream()
    # extend_sentiment_words_save()
    # build_dictionary()
    # clean_dict_save()
    # uild_bow_save()
    run_lda(60)



