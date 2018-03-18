import numpy as np
import pandas as pd
import math
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from builtSession import main


def built_train(dict_n):
    y = []
    x = []
    for i in sorted(dict_n.keys()):  # i指会话id
        n=len(dict_n[i])  # n指会话的长度
        sku_uti= dict()

        for j in range(1,n):     # j是指位置。之所以range后，跟的是n，而非n+1是因为session中的最后一个商品是购买，
                                  # 我们计算效用时不用。
            ini_x=np. zeros((1, 5))        #初始化x，既指数函数值
            sku_uti.setdefault(dict_n[i][j][0],ini_x )       # 对每个商品建立一个x的数组.

            current_type= dict_n[i][j][1]     #当前类型

            temp=math.exp((n-j)/n)      #指数函数值

            sku_uti[dict_n[i][j][0]][0][current_type-1]+= temp    # 在相应位置加上这个指数函数值
                                          # sku_uti={sku_id:[[x1,x2,x3,x4,x5]]}


        final_sku=dict_n[i][max(dict_n[i].keys())][0]

        for i in sku_uti:              #建立特征x和标记y
            x.append(sku_uti[i][0])
            if i != final_sku:
                y.append(0)
            else:
                y.append(1)
    x=pd.DataFrame(x)

    print('len(x)',len(x))
    x_add=[]
    for i in range(len(y)):
        if y[i]==1:
            x_add.append(x.iloc[i])

    y_add=[1]*len(x_add)*2
    y.extend(y_add)    #注意，这里y没有返回值，直接在y的基础上做的改变。

    x_add=pd.DataFrame(x_add)
    x=pd.concat([x,x_add])
    x=pd.concat([x,x_add])
    y=pd.Series(y)

    print('y',y)
    print('x',x)
    print('y',len(y))
    print('y=1',sum(y))
    return x,y


#随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
def split_sample(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


#模型训练
def modle_train(X_train,X_test, y_train, y_test):
    xgbr = xgb.XGBClassifier()
    xgbr.fit(X_train, y_train)
    xgbr_y_predict = xgbr.predict(X_test)
    print(classification_report(y_test, xgbr_y_predict, target_names=['0', '1']))


#代码入口
if __name__=='__main__':
    dict_n = main()
    x,y=built_train(dict_n)
    X_train, X_test, y_train, y_test=split_sample(x, y)
    print(modle_train(X_train,X_test, y_train, y_test))








'''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)
'''


