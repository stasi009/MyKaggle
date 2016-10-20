
import logging
from gensim import corpora,models
from review import Review,ReviewsDAL

def words_stream():
    dal = ReviewsDAL()
    train_words_stream = dal.load_reviews_words("train")
    unlabled_words_stream

def build_dictionary():
    pass

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
