
import pandas as pd
import text_utility

def extend_sentiment_words_save():
    poswords = pd.read_csv('datas/positive-words.txt',skiprows=35, header=None).values.flatten().tolist()
    negwords = pd.read_csv('datas/negative-words.txt',skiprows=35, header=None).values.flatten().tolist()

    neg_poswords = [w+text_utility.NEG_SUFFIX for w in poswords]
    neg_negwords = [w+text_utility.NEG_SUFFIX for w in negwords]

    poswords += neg_negwords
    negwords += neg_poswords

    pd.Series(poswords).to_csv("datas/extend_poswords.txt",index=False)
    pd.Series(negwords).to_csv("datas/extend_negwords.txt",index=False)

if __name__ == '__main__':
    extend_sentiment_words_save()
