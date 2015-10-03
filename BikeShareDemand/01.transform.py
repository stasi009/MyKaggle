
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from dateutil.parser import parse as dt_parse

def parse_day_time(strtime):
    dt = dt_parse(strtime)
    return pd.Series([dt.month,dt.day,dt.isoweekday(),dt.hour],index=["month","day","weekday","hour"])

def transform_save(oriFilename):
    df = pd.read_csv(oriFilename,index_col = "datetime")

    # ---------------- extend time information
    daytimes = df.index.to_series()
    time_infos = daytimes.apply(parse_day_time)
    extend_features = [time_infos,df]
    
    # ---------------- convert counts into log domain
    if "count" in df.columns:# in training dataset
        count_cols = ["casual","registered"]
        logcounts = [ np.log(df[c] + 1) for c in count_cols]
        extend_features.append(pd.concat(logcounts,axis=1,keys=[ "log_" + c for c in count_cols]))

    extend_df = pd.concat(extend_features,axis=1)

    baseFilename = os.path.splitext(oriFilename)[0]
    outFilename = baseFilename + "_extend.csv"

    extend_df.to_csv(outFilename,index_label="datetime")

if __name__ == "__main__":
    transform_save("train.csv")
    transform_save("test.csv")

