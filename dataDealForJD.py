import pandas as pd


action_2016_02_file='JData_ori/JData_Action_201602.csv'
action_2016_03_file='jdata_sam/JData_Action_201603.csv'
action_2016_04_file='jdata_sam/JData_Action_201604.csv'

df=pd.read_csv(action_2016_02_file)

df=df[df['time']>='2016-02-01']

df.drop(['model_id'],axis=1,inplace=True)   #删除列，axis默认为0代表删除行；所以要置axis=1

df.drop_duplicates(inplace=True)    #去重

df=df.sort_values(['user_id','time'],axis = 0,ascending = True)    #对每个用户按时间排序。

#df1=df[(df['type']==2) | (df['type']==4)]    #筛选出加购和下单的记录。看是否下单中的商品是否都在购物车中出现过。有的是，有的不是。
#df2=df[df['type']==3]
#print(df[:240])


'''
#找到加入购物车的和购买的序列
dict_addcart=dict()
dict_buy=dict()

for i in range(0, len(df)):                   #一次遍历dataframe的每一行

    if df.iloc[i]['type']==2:
        dict_addcart.setdefault(df.iloc[i]['user_id'], {})

        dict_addcart[df.iloc[i]['user_id']].setdefault('addcart',[])
        dict_addcart[df.iloc[i]['user_id']]['addcart'].append(df.iloc[i]['sku_id'])

    elif df.iloc[i]['type']==4:
        dict_buy.setdefault(df.iloc[i]['user_id'], {})

        dict_buy[df.iloc[i]['user_id']].setdefault('buy', [])
        dict_buy[df.iloc[i]['user_id']]['buy'].append(df.iloc[i]['sku_id'])

count=0
for i in dict_buy:
    if i in dict_addcart:
        if dict_buy[i]['buy'][0] in dict_addcart[i]['addcart']:
            count+=1
#print(count)'''


#这段代码是以每次的购买为断点，切分成会话 ,忽略操作类型！！!不可行
dict_act=dict()
dict_sess=dict()
j=1
for i in range(0, len(df)):                   #一次遍历dataframe的每一行
    dict_act.setdefault(df.iloc[i]['user_id'], {})
    dict_act[df.iloc[i]['user_id']].setdefault('act', [])

    if df.iloc[i]['type']!=4:        #对于还没到购买断点的序列，一直积累
        dict_act[df.iloc[i]['user_id']]['act'].append(df.iloc[i]['sku_id'])
    else:
        dict_sess.setdefault(j,dict_act[df.iloc[i]['user_id']]['act'])    #遇到购买，则建立个session。
        dict_act[df.iloc[i]['user_id']]['act'].append('hahaha')       #我是标记序列和购买的分割符
        dict_act[df.iloc[i]['user_id']]['act'].append(df.iloc[i]['sku_id'])   #购买的商品放入序列的最后一个。

        dict_act[df.iloc[i]['user_id']]['act']=[]   #并把之前的操作序列置空，重新跟踪下个操作序列
        j+=1
        continue



#开始建立会话，只统计每个商品的被操作类型，不可行！！！
dict_act=dict()
dict_sess=dict()
k=1

for i in range(0, len(df)):        #依次遍历dataframe的每一行
    user_inline=df.iloc[i]['user_id']   #当前这行记录的操作的用户的编号
    sku_inline = df.iloc[i]['sku_id']   #当前这行记录的操作的商品的编号
    type_inline=df.iloc[i]['type']     #当前这行记录的操作类型

    dict_act.setdefault(user_inline, {})

    if type_inline!=4:     #若操作类型不是购买
        dict_act[user_inline].setdefault(sku_inline, [])  #记录当前用户对每个商品的操作
        dict_act[user_inline][sku_inline].append(type_inline)

    else:
        dict_sess.setdefault(k,dict_act[user_inline])    #遇到购买，则建立个session。
        dict_sess[k][sku_inline]=[type_inline]

        dict_act[user_inline]={}  #并把之前的操作序列置空，重新跟踪下个操作序列
        k+=1
        continue


print(dict_sess)
















