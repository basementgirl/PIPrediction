import numpy as np
import pandas as pd

file_clicks= '/Users/rongrong/Documents/MyFiles/paper/purchasePredictionWithRNN/Data/yoochoose-data/yoochoose-clicks.dat'
df_clicks= pd.read_csv(file_clicks,header=None, usecols=[0,1,2])[:50]
df_clicks.columns=['sid','time','citemid']
print(df_clicks)



file_buy='/Users/rongrong/Documents/MyFiles/paper/purchasePredictionWithRNN/Data/yoochoose-data/yoochoose-buys.dat'
df_buy= pd.read_csv(file_buy,header=None, usecols=[0,1,2])
df_buy.columns=['sid','time','bitemid']
df_buy=df_buy.sort_values(['sid'],axis = 0,ascending = True)    #对每个用户按时间排序。

print(df_clicks)
print(df_buy)

