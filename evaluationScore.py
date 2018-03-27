import math
import pandas as pd
import numpy as np


def convert_com_nums(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    elif n==2:
        return 7
    elif n==3:
        return 30
    elif n==4:
        return 80

def get_com():
    comment_file = 'JData_ori/JData_Comment.csv'
    df = pd.read_csv(comment_file)
    df['comment_num'] = df['comment_num'].map(convert_com_nums)
    df['good_comment_rate']=1-df['bad_comment_rate']

    p=df['good_comment_rate']
    n=df['comment_num']
    z=1.96
    df['good_comment_rate_new']=(p+1/(2*n)*z**2-z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))) / (1+1/n*z**2)
    #return df[['sku_id','good_comment_rate']]

    m=df['sku_id']
    m.drop_duplicates(inplace=True)    #去重
    df=df[['sku_id','good_comment_rate_new']]

    df=df.groupby(['sku_id']).mean()
    return df


print(get_com())