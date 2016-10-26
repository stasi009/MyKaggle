
import itertools
import os.path
import nltk
import logging
from gensim import corpora,models
from sklearn.decomposition import PCA
import bow_tfidf

def print_topic_distribution(model,filename):
    with open(filename,"wt") as outf:
        # ---------- write each topic and words' contribution
        topics = model.show_topics(num_topics=-1, log=False, formatted=True)
        for topic in topics:
            # topic[0]: topic number
            # topic[1]: topic description
            outf.write("\n############# TOPIC {} #############\n".format(topic[0]))
            outf.write(topic[1]+"\n")

        # ---------- words statistics in all topics
        outf.write("\n\n\n****************** KEY WORDS ******************\n")
        topics = model.show_topics(num_topics=-1, log=False, formatted=False)
        keywords = (word for (_,words) in topics for (word,score) in words)

        fdist = nltk.FreqDist(keywords)
        for index,(w,c) in enumerate( fdist.most_common(100) ):
            outf.write("{}-th keyword: <{},{}>\n".format(index+1,w,c))


def run_lda_with_train_unlabeled(n_topics):
    dictionary = corpora.Dictionary.load("vsm/dictionary.dict")
    train_bow = corpora.MmCorpus('vsm/train.bow')
    unlabeled_bow = corpora.MmCorpus('vsm/unlabeled.bow')

    # model = models.LdaMulticore(train_bow, id2word=dictionary, num_topics=n_topics,passes=3)
    model = models.LdaModel(train_bow, id2word=dictionary, num_topics=n_topics,passes=3)
    print "======== LDA built on train set ========"

    model.update(unlabeled_bow)
    print "======== LDA updated on unlabeled set ========"

    # --------------- save result
    tag = 'popcorn'
    model_name = os.path.join("vsm",tag+".lda_model")
    model.save(model_name)

    topic_name = os.path.join("vsm",tag+"_topics.txt")
    print_topic_distribution(model,topic_name)

class LsiReducer(object):

    def __init__(self):
        self.dictionary = corpora.Dictionary.load('vsm/dictionary.dict')

    def fit(self, n_topics):
        train_corpus = corpora.MmCorpus('vsm/train.tfidf')
        self.model = models.LsiModel(train_corpus, id2word=self.dictionary, num_topics=n_topics)
        print "====== fitted on train corpus ======"

        unlabeled_corpus = corpora.MmCorpus('vsm/unlabeled.tfidf')
        self.model.add_documents(unlabeled_corpus)
        print '====== updated by unlabeled corpus ======'

        self.model.save('vsm/popcorn.lsi{}_model'.format(n_topics))
        print "====== LSI{} model built and saved ======".format(n_topics)

    def reduce_save(self):
        # unlabeled dataset is huge and won't be used in classification, so ignore it
        colnames = ['train','validate','test']
        for colname in colnames:
            tfidf_corpus = corpora.MmCorpus('vsm/{}.tfidf'.format(colname))
            # cannot be called by model[tfidf_corpus,True]
            lsi_corpus = self.model.__getitem__( tfidf_corpus,scaled=True)
            corpora.MmCorpus.serialize('vsm/{}.lsi{}'.format(colname,self.model.num_topics), lsi_corpus)
            print "====== TF-IDF[{}] reduced to {} dims ======".format(colname,self.model.num_topics)

def lsi_reduce(n_topics):
    r = LsiReducer()
    r.fit(n_topics)
    r.reduce_save()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # run_lda_with_train_unlabeled(50)

    lsi_reduce(1000)



