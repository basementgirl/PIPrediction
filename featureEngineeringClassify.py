import numpy as np
import pandas as pd
import math
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from builtSession import main
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import cross_validation, metrics


def built_train(dict_n):
    lst=[]
    for i in sorted(dict_n.keys()):  # i指会话id
        print('i',i)
        fre_count = []
        n = len(dict_n[i])  # n指会话的长度

        sku_type_count = {}
        sku_lst = []  # 用来统计特征

        for j in range(1,len(dict_n[i])+1):     #对于当前会话的每一个商品
            print('hh',dict_n[i][j])
            cur_sku=dict_n[i][j][0]
            cur_typ=dict_n[i][j][1]

            fre_count.append(cur_sku)

            sku_type_count.setdefault(cur_sku,[])
            sku_type_count[cur_sku].append(cur_typ)     #记录商品的操作类型。

        for i in sku_type_count:     #对于每个商品
            sku_lst.append(i)
            sku_lst.append(fre_count.count(i))      #频次
            for j in range(1,6):
                sku_lst.append(sku_type_count[i].count(j))     #统计当前商品各操作类型的频次。

            lst.append(sku_lst)
            sku_lst=[]

    return lst




# 代码入口
if __name__ == '__main__':
    dict_n = main()
    print('dict_n',dict_n)
    print(built_train(dict_n))











