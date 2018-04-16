import numpy as np
import pandas as pd
import math
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from builtSession import main
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import cross_validation,metrics


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

            temp=math.exp(-(n-j)/n)     #指数函数值

            sku_uti[dict_n[i][j][0]][0][current_type-1]+= temp    # 在相应位置加上这个指数函数值
                                          # sku_uti={sku_id:[[x1,x2,x3,x4,x5]]}


        final_sku=dict_n[i][max(dict_n[i].keys())][0]

        for i in sku_uti:              #建立特征x和标记y
            k=[i]
            k.extend(sku_uti[i][0])    #用来加上商品编号
            x.append(k)
            #x.append(sku_uti[i][0])
            if i != final_sku:
                y.append(0)
            else:
                y.append(1)        #实现了最初的x和y


    x=pd.DataFrame(x)
    x.columns = ['sku_id','clicks', 'browse', 'favor', 'addcart', 'delcart']


    #上采样
    x_add=[]
    for i in range(len(y)):
        if y[i]==1:
            x_add.append(x.iloc[i])      #复制正样本

    y_add=[1]*len(x_add)*2
    y.extend(y_add)    #注意，这里y没有返回值，直接在y的基础上做的改变。
    y=pd.Series(y)

    x_add=pd.DataFrame(x_add)
    x=pd.concat([x,x_add])
    x=pd.concat([x,x_add])


    #下采样
    for i in range(len(y)):
        if y[i]==0 and i %2==0:
            x.drop([i],inplace=True)     #删负样本
            y.drop([i],inplace=True)
    return x,y



#随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
def split_sample(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


#模型训练

def model_train(X_train,X_test, y_train, y_test):
    xgbc = xgb.XGBClassifier()
    xgbc.fit(X_train, y_train)
    xgbr_y_predict = xgbc.predict(X_test)
    
    test_auc = metrics.roc_auc_score(y_test,xgbr_y_predict)
    print('auc is :',test_auc)
    print(classification_report(y_test, xgbr_y_predict, target_names=['0', '1']))





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




#代码入口
if __name__=='__main__':
    dict_n = main()
    x,y=built_train(dict_n)

    x['sku_id']=x['sku_id'].astype(int)

    x.index=x.iloc[:,0]
    x.drop(['sku_id'],inplace=True,axis=1)

    x.fillna(0,inplace=True)

    print('hhh',x.shape,y.shape)
    print('positive sample num is: ', sum(y),'。 all sample num is ：',len(y))
    X_train, X_test, y_train, y_test=split_sample(x, y)
    print(model_train(X_train,X_test, y_train, y_test))










