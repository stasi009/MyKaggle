
import itertools
import os.path
import nltk
import logging
from gensim import corpora,models

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
    dictionary = corpora.Dictionary.load("processed/dictionary.dict")
    train_bow = corpora.MmCorpus('processed/train.bow')
    unlabeled_bow = corpora.MmCorpus('processed/unlabeled.bow')

    # model = models.LdaMulticore(train_bow, id2word=dictionary, num_topics=n_topics,passes=3)
    model = models.LdaModel(train_bow, id2word=dictionary, num_topics=n_topics,passes=3)
    print "======== LDA built on train set ========"

    model.update(unlabeled_bow)
    print "======== LDA updated on unlabeled set ========"

    # --------------- save result
    tag = 'popcorn'
    model_name = os.path.join("processed",tag+".lda")
    model.save(model_name)

    topic_name = os.path.join("processed",tag+"_topics.txt")
    print_topic_distribution(model,topic_name)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    run_lda_with_train_unlabeled(50)



