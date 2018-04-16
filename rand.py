import numpy as np
import pandas as pd
from builtSession import main
import random
from sklearn.metrics import classification_report



dict_n = main()
pred_all_lst = []
for i in sorted(dict_n.keys()):  #遍历所有会话
    sku_lst=[]
    ses_len = len(dict_n[i])  # n指会话的长度
    true_sku=dict_n[i][ses_len][0]  #取每个会话周期内的最后一个商品，即真实的y值

    for j in range(1, ses_len):  # 遍历一个会话
        cur_sku = dict_n[i][j][0]
        sku_lst.append(cur_sku)

        pred_sku=random.sample(sku_lst,1)  #随机抽取的值

        if cur_sku!=pred_sku:
            if cur_sku!=true_sku:
                pred_all_lst.append([cur_sku,0,0])
            else:
                pred_all_lst.append([cur_sku,0,1])
        else:
            if cur_sku!=true_sku:
                pred_all_lst.append([cur_sku,1,0])
            else:
                pred_all_lst.append([cur_sku, 1, 1])

df=pd.DataFrame(pred_all_lst)
#df.drop_duplicates(inplace=True)
df.columns=['sku_id','pred','true']

print(classification_report(df['true'], df['pred'], target_names=['0', '1']))

#print(df)




