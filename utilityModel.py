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
    l=len(dict_n)
    cou=0
    cout=0

    pred_lst=[]
    posi_lst=[]
    for i in sorted(dict_n.keys()):  # i指会话id
        n = len(dict_n[i])  # n指会话的长度
        sku_uti = {}
        for j in range(1, n):  # j是指位置。之所以range后，跟的是n，而非n+1是因为session中的最后一个商品是购买，
            # 我们计算效用时不用。

            current_sku = dict_n[i][j][0]  # 当前商品
            current_type = dict_n[i][j][1]  # 当前类型

            sku_uti.setdefault(current_sku,0)  # 对每个商品建立一个x的数组.

            temp = math.exp(-(n - j) / n)*current_type/10 # 指数函数值

            sku_uti[current_sku]+= temp  # 在相应位置加上这个指数函数值

        final_sku = dict_n[i][max(dict_n[i].keys())][0]
        posi_lst.append(final_sku)

        pred_sku=max(sku_uti.items(), key=lambda x: x[1])[0]
        pred_lst.append(pred_sku)

        if pred_sku!=final_sku:
            cou+=1

    for k in posi_lst:
        if k in pred_lst:
            cout+=1
    print(len(pred_lst),len(posi_lst),cou,cout)

    prec=cou/len(pred_lst)       #代表precision
    reca=cout/len(posi_lst)       #代表recall
    return prec,reca


# 代码入口
if __name__ == '__main__':
    dict_n = main()
    res= built_train(dict_n)
    print(res)

