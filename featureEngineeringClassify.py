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


    for i in sorted(dict_n.keys()):  # i指会话id
        fre_count = []
        n = len(dict_n[i])  # n指会话的长度

        sku_lst=[]
        sku_type_count = {}

        for j in range(len(dict_n[i])):
            fre_count.append(dict_n[i][j][0])
            sku_lst.append(dict_n[i][j][0])


            cur_sku=dict_n[i][j][0]
            cur_typ=dict_n[i][j][1]
            sku_type_count.setdefault(cur_sku,[])
            sku_type_count[cur_sku].append(cur_typ)     #记录商品的操作类型。

        sku_lst.append(sku_lst.count(cur_sku))     #统计当前商品的出现频次
        for j in range(1,6):
            sku_lst.append(sku_type_count[cur_sku].count(j))     #









    x = pd.DataFrame(x)
    x.columns = ['sku_id', 'clicks', 'browse', 'favor', 'addcart', 'delcart']

    # 上采样
    x_add = []
    for i in range(len(y)):
        if y[i] == 1:
            x_add.append(x.iloc[i])  # 复制正样本

    y_add = [1] * len(x_add) * 2
    y.extend(y_add)  # 注意，这里y没有返回值，直接在y的基础上做的改变。
    y = pd.Series(y)

    x_add = pd.DataFrame(x_add)
    x = pd.concat([x, x_add])
    x = pd.concat([x, x_add])

    # 下采样
    for i in range(len(y)):
        if y[i] == 0 and i % 2 == 0:
            x.drop([i], inplace=True)  # 删负样本
            y.drop([i], inplace=True)
    return x, y


# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
def split_sample(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


# 模型训练

def model_train(X_train, X_test, y_train, y_test):
    xgbc = xgb.XGBClassifier()
    xgbc.fit(X_train, y_train)
    xgbr_y_predict = xgbc.predict(X_test)

    test_auc = metrics.roc_auc_score(y_test, xgbr_y_predict)
    print('auc is :', test_auc)
    print(classification_report(y_test, xgbr_y_predict, target_names=['0', '1']))





def convert_com_nums(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 7
    elif n == 3:
        return 30
    elif n == 4:
        return 80


def get_com():
    comment_file = 'JData_ori/JData_Comment.csv'
    df = pd.read_csv(comment_file)
    df['comment_num'] = df['comment_num'].map(convert_com_nums)
    df['good_comment_rate'] = 1 - df['bad_comment_rate']

    p = df['good_comment_rate']
    n = df['comment_num']
    z = 1.96
    df['good_comment_rate_new'] = (p + 1 / (2 * n) * z ** 2 - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / (
    1 + 1 / n * z ** 2)
    df.drop_duplicates(inplace=True)  # 去重

    df = df[['sku_id', 'good_comment_rate_new']]

    df = df.groupby(['sku_id'], as_index=False).mean()
    return df


# 代码入口
if __name__ == '__main__':
    dict_n = main()
    x, y = built_train(dict_n)

    x['sku_id'] = x['sku_id'].astype(int)
    x_com = get_com()

    x = pd.merge(x, x_com, on=['sku_id'], how='left')
    x.index = x.iloc[:, 0]
    x.drop(['sku_id'], inplace=True, axis=1)

    x.fillna(0, inplace=True)

    print('hhh', x.shape, y.shape)
    print('positive sample num is: ', sum(y), '。 all sample num is ：', len(y))
    X_train, X_test, y_train, y_test = split_sample(x, y)
    print(model_train(X_train, X_test, y_train, y_test))









