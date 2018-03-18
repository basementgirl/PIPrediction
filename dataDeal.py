# -*- coding: utf-8 -*-

import pandas as pd

f1=open('Data/yoochoose-data/yoochoose-clicks.dat')
d_clicks=dict()
for line in f1.readlines():
    line=line.split(',')
    d_clicks.setdefault(line[0],[])
    d_clicks[line[0]].append(line[2])
print(len(d_clicks))
f1.close()


f2=open('Data/yoochoose-data/yoochoose-buys.dat')
d_buys=dict()
for line in f2.readlines():
    line=line.split(',')
    d_buys.setdefault(line[0],[])
    d_buys[line[0]].append(line[2])

print(len(d_buys.keys()))
f2.close()

for i in d_clicks:
    if i in d_buys:
        d_clicks[i].append(d_buys[i])
        print(d_clicks[i],'hhh')
    else:
        d_clicks[i].append('None')
print(d_clicks)


