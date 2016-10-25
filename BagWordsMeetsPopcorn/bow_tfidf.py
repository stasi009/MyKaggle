
import re
import itertools
import logging
import text_utility
from gensim import corpora,models
from review import Review,ReviewsDAL

def words_stream(colname):
    dal = ReviewsDAL()
    review_stream = dal.load_words(colname)

    for index,r in enumerate( review_stream  ):
        yield r.sent.words
        if index % 300 == 0:
            print "{} examples loaded from mongodb[{}]".format(index+1,colname)

    dal.close()

def build_dictionary():
    train_words_stream = words_stream('train')
    unlabeled_words_stream = words_stream('unlabeled')
    wstream = itertools.chain(train_words_stream, unlabeled_words_stream)

    dictionary = corpora.Dictionary(wstream)
    dictionary.save('processed/dictionary.dict')  # store the dictionary, for future reference
    print "======== Dictionary Generated and Saved ========"

class DictCleaner(object):

    def __init__(self):

        self._invalid_pattern = re.compile(r"[^a-zA-Z_]")

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

    def clean(self,no_below=5, no_above=0.5, keep_n=100000):
        dictionary = corpora.Dictionary.load('processed/dictionary.dict')
        print "originally, there are {} tokens".format(len(dictionary))

        # filter out too often/rare
        dictionary.filter_extremes(no_below=no_below,no_above=no_above,keep_n=keep_n)
        print "after filtering too often/rare, there are {} tokens".format(len(dictionary))

        # filter out invalid tokens
        invalid_tokenids = [id for token,id in dictionary.token2id.viewitems()
                            if self.is_token_invalid(token) ]
        print "there are {} tokens are invalid".format(len(invalid_tokenids))

        dictionary.filter_tokens(bad_ids = invalid_tokenids)
        print "after filtering invalid, there are {} tokens".format(len(dictionary))

        return dictionary

def clean_dict_save():
    cleaner = DictCleaner()
    clean_dict = cleaner.clean(no_below=10,no_above=0.8,keep_n=15000)

    clean_dict.save('processed/dictionary.dict')
    # sort by decreasing doc-frequency
    clean_dict.save_as_text("processed/dictionary.txt", sort_by_word=False)

    print "dictionary is cleaned, shrinked and saved"

def build_bow_save():
    dictionary = corpora.Dictionary.load('processed/dictionary.dict')

    colnames = ['train','test','validate','unlabeled']
    for colname in colnames:
        print "\n=========== collection[{}] to BOW, ......".format(colname)

        bow_stream = (dictionary.doc2bow(words) for words in words_stream(colname))
        target_file = "processed/{}.bow".format(colname)
        corpora.MmCorpus.serialize(target_file, bow_stream)

        print "=========== BOW[{}] saved ===========".format(colname)

    print "!!! DONE !!!"

def build_tfidf_save():
    dictionary = corpora.Dictionary.load("processed/dictionary.dict")

    train_bow = corpora.MmCorpus('processed/train.bow')
    unlabeled_bow = corpora.MmCorpus('processed/unlabeled.bow')
    corpus = itertools.chain(train_bow,unlabeled_bow)

    model = models.TfidfModel(corpus=corpus, id2word=dictionary,dictionary=dictionary, normalize=True)
    model.save("processed/popcorn.tfidf_model")
    print 'TF-IDF model generated and saved.'

    colnames = ['train', 'test', 'validate', 'unlabeled']
    for colname in colnames:
        print "\n=========== BOW[{}] to TF-IDF, ......".format(colname)

        bow_stream = corpora.MmCorpus('processed/{}.bow'.format(colname))
        tfidf_stream = model[bow_stream]

        targetfile = 'processed/{}.tfidf'.format(colname)
        corpora.MmCorpus.serialize(targetfile, tfidf_stream)

        print "=========== TF-IDF[{}] saved ===========".format(colname)

    print "!!! DONE !!!"

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    raise Exception("!!! ATTENTION !!!\nthe script has run once. \nrun this script again will overwrite existing files.")

    # build_dictionary()
    # clean_dict_save()
    # build_bow_save()
    # build_tfidf_save()