import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from builtSession import main
from modelSet import ann_model,gbdt_model,lr_model


def built_train(dict_n):
    x=[]
    y=[]
    for i in sorted(dict_n.keys()):  # i指会话id

        fre_count = []
        ses_len = len(dict_n[i])  # n指会话的长度

        sku_type_count = {}   #统计类型
        sku_ord={}  #统计次序
        sku_fea = {} # 用来统计特征

        for j in range(1,ses_len):     #遍历一个会话
            cur_sku=dict_n[i][j][0]
            cur_typ=dict_n[i][j][1]

            fre_count.append(cur_sku)

            sku_type_count.setdefault(cur_sku,[])
            sku_type_count[cur_sku].append(cur_typ)     #记录商品的操作类型。

            sku_ord.setdefault(cur_sku,[])       #记录出现的次序
            sku_ord[cur_sku].append(j)
        for m in sku_type_count:     #对于每个商品
            sku_fea.setdefault(m,[])
            sku_fea[m].append(fre_count.count(m))      #频次
            sku_fea[m].append(ses_len-max(sku_ord[m]))    #次序

            for n in range(1,6):
                sku_fea[m].append(sku_type_count[m].count(n))     #统计当前商品各操作类型的频次。

        final_sku = dict_n[i][max(dict_n[i].keys())][0]   #当前会话的最后一个位置上的商品，即最后购买的商品

        for k in sku_fea:  # 建立特征x和标记y. sku_fea形如：{sku_id:[频次，浏览次数]}
            temp = [k]
            temp.extend(sku_fea[k])  # 用来加上商品编号
            x.append(temp)
            if k != final_sku:
                y.append(0)
            else:
                y.append(1)  # 实现了最初的x和y

    x = pd.DataFrame(x)
    x.columns = ['sku_id', 'act_num','ord' ,'click_num','browse_num', 'favor_num', 'addcart_num', 'delcart_num']

    '''
    上采样
    x_add = []
    for i in range(len(y)):
        if y[i] == 1:
            x_add.append(x.iloc[i])  # 复制正样本

    y_add = [1] * len(x_add) * 2
    y.extend(y_add)  # 注意，这里y没有返回值，直接在y的基础上做的改变。
    y = pd.Series(y)

    x_add = pd.DataFrame(x_add)
    x = pd.concat([x, x_add])
    x = pd.concat([x, x_add])'''

    y = pd.Series(y)
    # 下采样
    for i in range(len(y)):
        if y[i] == 0 and (i % 2 == 0 or i % 3 == 0  ):
            x.drop([i], inplace=True)  # 删负样本
            y.drop([i], inplace=True)
    return x,y


# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
def split_sample(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


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

#def get_comp_train_and_test():
    dict_n = main()
    x, y = built_train(dict_n)

    x['sku_id'] = x['sku_id'].astype(int)
    # x_com = get_com()

    # x = pd.merge(x, x_com, on=['sku_id'], how='left')
    x.index = x.iloc[:, 0]
    x.drop(['sku_id'], inplace=True, axis=1)

    x.fillna(0, inplace=True)

    print('hhh', x.shape, y.shape)
    print('positive sample num is: ', sum(y), '。 all sample num is ：', len(y))
    X_train, X_test, y_train, y_test = split_sample(x, y)

    #return X_train, X_test, y_train, y_test

    print(ann_model(X_train, X_test, y_train, y_test))









