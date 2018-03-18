import pandas as pd


action_2016_02_file='JData_ori/JData_Action_201602.csv'
df=pd.read_csv(action_2016_02_file)

#进行简单的数据预处理工作
df=df[df['time']>='2016-02-01']
df.drop(['model_id'],axis=1,inplace=True)   #删除列，axis默认为0代表删除行；所以要置axis=1
df.drop(['cate'],axis=1,inplace=True)
df.drop(['brand'],axis=1,inplace=True)
df.drop_duplicates(inplace=True)    #去重
df=df.sort_values(['user_id','time'],axis = 0,ascending = True)    #对每个用户按时间排序。
# print(df)
df=df[:500]
print('df',df)


#print(df)
#开始建立会话，形如：  session_id：{位置：[sku,类型]}   ps：这里的位置不是在遍历的第几行，而是在有购买的会话中的位置。
dict_type=dict()
dict_sku=dict()
dict_sess=dict()
j=1
for i in range(0, len(df)):        #依次遍历dataframe的每一行
    user_inline=df.iloc[i]['user_id']   #当前这行记录的操作的用户的编号
    sku_inline = df.iloc[i]['sku_id']   #当前这行记录的操作的商品的编号
    type_inline=df.iloc[i]['type']     #当前这行记录的操作类型

    dict_sku.setdefault(user_inline, {})
    dict_sku[user_inline].setdefault(i, sku_inline)  # 这里的i是指遍历时所在的行。后面要在转换
    dict_type[user_inline].setdefault(i, type_inline)

    if type_inline==4:     #若操作类型是购买
        dict_sess[j]=dict_sku[user_inline]
        dict_act[user_inline] = {}  # 并把之前的操作序列置空，重新跟踪下个操作序列
        j += 1

print(dict_sess)


#ps"默认情况下Python的字典输出顺序是按照键的按创建顺序。因原来的位置标记不是从1开始，做下转换。
dict_n=dict()
for i in dict_sess:
    dict_n[i]={}
    m=1
    for j in dict_sess[i]:
        dict_n[i][m]=dict_sess[i][j]
        m+=1
print(dict_n)


for i in dict_n:
    n=len(dict_n[i])
    print(n)
