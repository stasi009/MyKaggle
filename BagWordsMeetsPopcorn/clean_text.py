
import os.path
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def clean_text(raw_review):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Convert to lower case
    return letters_only.lower()

def clean_save(filename):
    df = pd.read_csv(filename,sep='\t', index_col="id",error_bad_lines=False)
    df['review'] = df.review.map(clean_text)

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    basename,ext = os.path.splitext(basename)
    targetname = os.path.join(dirname,"clean_{}.csv".format(basename))

    df.to_csv(targetname,index_label='id')

    print "*** cleaned reviews saved to '{}' ***".format(targetname)

if __name__ == "__main__":
    srcfiles = ["labeledTrainData.tsv","unlabeledTrainData.tsv","testData.tsv"]
    for srcfile in srcfiles:
        clean_save(os.path.join('datas',srcfile))



