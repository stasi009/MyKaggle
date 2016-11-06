
import re
import nltk
from nltk.corpus import stopwords

# ***************************** Text Replacement ***************************** #
ReplacePatterns = [
                    (r'won\'t', 'will not'),
                    (r'can\'t', 'cannot'),
                    (r'i\'m', 'i am'),
                    (r'ain\'t', 'is not'),
                    (r'(\w+)\'ll', '\g<1> will'),
                    (r'(\w+)n\'t', '\g<1> not'),
                    (r'(\w+)\'ve', '\g<1> have'),
                    (r'(\w+)\'s', '\g<1> is'),
                    (r'(\w+)\'re', '\g<1> are'),
                    (r'(\w+)\'d', '\g<1> would'),
                    ]
ReplacePatterns = [(re.compile(regex, re.IGNORECASE), replacewith) for regex, replacewith in ReplacePatterns]

def replace(text):
    for (pattern, replacewith) in ReplacePatterns:
        text = re.sub(pattern, replacewith, text)
    return text

# ***************************** POS Tagging ***************************** #
Coarse2FinePosTags = {
    'n': ("NN", "NNS", "NNP", "NNPS"),
    'a': ("JJ", "JJR", "JJS"),
    'r': ("RB", "RBR", "RBS"),
    'v': ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
}
Fine2CoarsePosTags = {finetag: k for k, v in Coarse2FinePosTags.iteritems() for finetag in v}

def lemmatize_with_pos(lemmatizer, words):
    """
    to get better lemmatization result, we need to specify POS in lemmatize(w,pos) method
    however, the POS from nltk.pos_tag function is fine-grained POS,
    doesn't match what lemmatize want, which is coarse-grained POS
    so I define this function to transform fine-grained POS to coarse-grained POS
    and return 'n' for unspecified, which is the default of lemmatize() method
    """
    pos_tagged_words = nltk.pos_tag(words)
    return [lemmatizer.lemmatize(w, pos=Fine2CoarsePosTags.get(pos, 'n')) for w, pos in pos_tagged_words]

# ***************************** Negation Marking ***************************** #
# regex to match negation tokens
NEGATION_RE = re.compile("""(?x)(?:
^(?:never|no|nothing|nowhere|noone|none|not|
    havent|hasnt|hadnt|cant|couldnt|shouldnt|
    wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint|without
 )$
)
|
n't""")

# regex to match punctuation tokens
PUNCT_RE = re.compile("^[,.:;!?-]$")

NEG_SUFFIX = "_neg"  # lower-case suffix makes things easier

NegSuffixPattern = re.compile(r"{}$".format(NEG_SUFFIX))

def add_negation_suffixes(tokens):
    """
    add simple negation marking to tokenized text to aid in sentiment analysis.

    As defined in the link above, the basic approach is to
    "Append a _NEG suffix to every word appearing between a
    negation and a clause-level punctuation mark". Here, negation
    words are defined as those that match the NEGATION_RE regex, and
    clause-level punctuation marks are those that match the PUNCT_RE regex.

    Please note that this method is due to Das & Chen (2001) and Pang, Lee & Vaithyanathan (2002)

    INPUT: List of strings (tokenized sentence)
    OUTPUT: List of string with negation suffixes added

    Adds negation markings to a tokenized string.
    """

    # negation tokenization
    neg_tokens = []
    append_neg = False # stores whether to add "_NEG"

    for token in tokens:

        # if we see clause-level punctuation,
        # stop appending suffix
        if PUNCT_RE.match(token):
            append_neg = False

        # Do or do not append suffix, depending
        # on state of 'append_neg'
        if append_neg:
            neg_tokens.append(token + NEG_SUFFIX)
        else:
            neg_tokens.append(token)

        # if we see negation word,
        # start appending suffix
        if NEGATION_RE.match(token):
            append_neg = True

    return neg_tokens

def remove_neg_suffix(word):
    return NegSuffixPattern.sub('',word)

def make_stop_words():
    stop_words = stopwords.words("english")

    # below words have been processed in "negation marking" process
    # so they can be removed as stopwords
    stop_words.extend(["never", "without"])

    stop_neg_suffixed = [stopword + NEG_SUFFIX for stopword in stop_words]
    stop_words.extend(stop_neg_suffixed)

    return frozenset(stop_words)
