
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from dateutil.parser import parse as dt_parse

def parse_day_time(strtime):
    dt = dt_parse(strtime)
    return pd.Series( [dt.month,dt.day,dt.weekday(),dt.hour],index=["month","day","weekday","hour"])

def expandtime_save(oriFilename):
    df = pd.read_csv(oriFilename,index_col = "datetime")

    daytimes = df.index.to_series()
    time_infos = daytimes.apply(parse_day_time)

    extend_df = pd.concat([time_infos,df],axis=1)

    baseFilename = os.path.splitext(oriFilename)[0]
    outFilename = baseFilename + "_extend.csv"

    extend_df.to_csv(outFilename,index_label="datetime")

expandtime_save("train.csv")
expandtime_save("test.csv")

