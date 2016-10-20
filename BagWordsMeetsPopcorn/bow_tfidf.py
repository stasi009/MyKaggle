
import re
import itertools
import logging
from gensim import corpora,models
from review import Review,ReviewsDAL

def words_stream():
    dal = ReviewsDAL()
    train_review_stream = dal.load_words("train")
    unlabled_review_stream = dal.load_words("unlabeled")

    for index,r in enumerate( itertools.chain(train_review_stream,unlabled_review_stream) ):
        yield r.sent.words
        if index % 300 == 0:
            print "{} examples loaded from mongodb".format(index+1)

    dal.close()

def build_dictionary():
    dictionary = corpora.Dictionary(words_stream())
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

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build_dictionary()
    # clean_dict_save()