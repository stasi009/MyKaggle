
import numpy as np
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')

traindata = pd.DataFrame.from_csv("train_processed.csv")

traindata.Survived.groupby(traindata.Sex).sum()