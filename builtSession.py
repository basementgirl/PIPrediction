
import pandas as pd


def convert_type(type):
    if type == 1:
        return 2
    elif type == 2:
        return 4
    elif type == 3:
        return 5
    elif type == 4:
        return 6
    elif type == 5:
        return 3
    elif type == 6:
        return 1


def data_deal(df):
    #进行简单的数据预处理工作
    df=df[df['time']>='2016-02-01']
    df.drop(['model_id'],axis=1,inplace=True)   #删除列，axis默认为0代表删除行；所以要置axis=1
    df.drop(['cate'],axis=1,inplace=True)
    df.drop(['brand'],axis=1,inplace=True)
    df.drop_duplicates(inplace=True)    #去重
    df=df.sort_values(['user_id','time'],axis = 0,ascending = True)    #对每个用户按时间排序。

    df['type'] = df['type'].map(convert_type)
    df=df[:100000]
    return df


def bulit_session(df):
    dict_act=dict()
    dict_sess=dict()
    j=1
    for i in range(0, len(df)):        #依次遍历dataframe的每一行
        user_inline=df.iloc[i]['user_id']   #当前这行记录的操作的用户的编号
        sku_inline = df.iloc[i]['sku_id']   #当前这行记录的操作的商品的编号
        type_inline=df.iloc[i]['type']     #当前这行记录的操作类型

        dict_act.setdefault(user_inline, {})
        dict_act[user_inline][i] = [sku_inline, type_inline]  # 这里的i是指遍历时所在的行。后面要在转换
        if type_inline==6:         #若操作类型是购买
            dict_sess[j]=dict_act[user_inline]
            dict_act[user_inline] = {}  # 并把之前的操作序列置空，重新跟踪下个操作序列
            j += 1

    # ps"默认情况下Python的字典输出顺序是按照键的按创建顺序。因原来的位置标记不是从1开始，做下转换。
    # 转换后，形如：session_id：{1：[sku1,类型]，2：[sku2,类型]}
    dict_n = dict()
    for i in dict_sess:
        dict_n[i] = {}
        m = 1
        for j in sorted(dict_sess[i].keys()):
            dict_n[i][m] = dict_sess[i][j]
            m += 1                  #这里出的问题，因为无序输出，所以，会出现第8各session中，第二个位置为类型6的情况。用sorted限制


    dict_new=dict_n.copy()
    for k in dict_n:
        if len(dict_n[k])<5:
            dict_new.pop(k)

    return dict_new


def main():
    action_2016_03_file = 'JData_ori/JData_Action_201602.csv'
    df = pd.read_csv(action_2016_03_file)
    df=data_deal(df)
    return bulit_session(df)



'''
dict_n=main()
for i in sorted(dict_n):
    final = dict_n[i][max(dict_n[i].keys())]
    print('final_sku', final)'''


