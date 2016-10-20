
import re
import itertools
import logging
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
    unlabeled_words_stream = words_stream('unlabled')
    wstream = itertools.chain(train_words_stream, unlabeled_words_stream)

    dictionary = corpora.Dictionary(wstream)
    dictionary.save('processed/dictionary.dict')  # store the dictionary, for future reference
    print "======== Dictionary Generated and Saved ========"

def clean_dictionary(no_below=5, no_above=0.5, keep_n=100000):
    dictfile = 'processed/dictionary.dict'
    dictionary = corpora.Dictionary.load(dictfile)
    print "originally, there are {} tokens".format(len(dictionary))

    # filter out too often/rare
    dictionary.filter_extremes(no_below=no_below,no_above=no_above,keep_n=keep_n)
    print "after filtering too often/rare, there are {} tokens".format(len(dictionary))

    # filter out words with non-characters
    invalid_pattern = re.compile(r"[^a-zA-Z]")
    invalid_tokenids = [id for token,id in dictionary.token2id.viewitems() if invalid_pattern.search(token) is not None]
    print "there are {} tokens containing non-character".format(len(invalid_tokenids))

    dictionary.filter_tokens(bad_ids = invalid_tokenids)
    print "after filtering non-character, there are {} tokens".format(len(dictionary))

    return dictionary

def clean_dict_save():
    clean_dict = clean_dictionary(no_below=10,no_above=0.8,keep_n=15000)

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


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build_dictionary()
    # clean_dict_save()
    build_bow_save()