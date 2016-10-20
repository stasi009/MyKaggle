
import re
import nltk
import text_utility

class Sentence(object):
    Lemmatizer = nltk.WordNetLemmatizer()

    def __init__(self,raw = None,words = None,aspect = None,sentiment = None):
        self.raw = raw
        self.words = words
        self.aspect = aspect
        self.sentiment = sentiment

    def to_dict(self):
        return {"raw":self.raw,"words":self.words,"aspect":self.aspect,"sentiment":self.sentiment}

    def words_no_negsuffix(self):
        return [text_utility.remove_neg_suffix(w) for w in self.words]

    @staticmethod
    def from_raw(text, stop_words,neg_mark = True):
        if not ( isinstance(stop_words,set) or isinstance(stop_words,frozenset) ):
            raise TypeError("stop_words pass in must be set or frozenset")

        sent = Sentence(text)

        ############### expand contraction and abbrevations
        text = text_utility.replace(text)

        ############### remove numbers, but need to keep punctuations, which is required in negation marking
        text = re.sub(r"[0-9\\\/\-\(\)]", " ", text)

        ############### normalize to lower case
        text = text.lower()

        ############### tokenize into words
        words = nltk.word_tokenize(text)

        ############### lemmatize
        # !!! Notice the order is important
        # !!! given ["the","parking","is","crazy"], nltk.pos_tag can recognize "parking" as Noun,
        # !!! lemmatize will return "parking"
        # !!! however, if we remove stopwords first, given ["parking","crazy"],
        # !!! nltk.pos_tag will think "parking" as Verb
        # !!! and lemmatize return "park"
        words = text_utility.lemmatize_with_pos(Sentence.Lemmatizer,words)

        ############### add negation suffix
        if neg_mark:
            words = text_utility.add_negation_suffixes(words)

        ############### remove stopwords
        # condition "len(w)>1" will remove punctuations
        words = [w for w in words if len(w)>1 and w not in stop_words]

        #
        sent.words = words
        return sent

    @staticmethod
    def from_dict(d):
        return Sentence(d.get("raw",None),d.get("words",None),d.get("aspect",None),d.get("sentiment",None))
