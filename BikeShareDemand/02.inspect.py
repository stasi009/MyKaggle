import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

train_df = pd.read_csv("train_extend.csv",index_col="datetime")

train_df.boxplot(column="registered",by="holiday")
train_df.boxplot(column="casual",by="weekday")
train_df.boxplot(column="registered",by="season")

plt.scatter(train_df.temp,train_df["casual"],alpha=0.3)