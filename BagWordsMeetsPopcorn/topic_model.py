
import csv
import nltk
from nltk.corpus import stopwords
import logging
from gensim import corpora,models

def review_to_words(r,stopwords):
    words = nltk.word_tokenize(r)
    return [w for w in words if len(w)>1 and w not in stop_words]

stop_words = set(stopwords.words("english"))



