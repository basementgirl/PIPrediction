import matplotlib.pyplot as plt
import math
import numpy as np

'''
y_lst=[]
x_lst=range(1,1000)
for i in x_lst:
    y=-math.log(i/1000)
    y_lst.append(y)

x_lst1=[x/1000 for x in x_lst]

plt.plot(x_lst1, y_lst,'r', label='broadcast')

plt.xlabel('x')
plt.ylabel('f(s,i)')
plt.savefig('1')

y_lst=[]
x_lst=range(1,1000)
for i in x_lst:
    y=math.exp(-i/1000)
    y_lst.append(y)

x_lst1=[-x/1000 for x in x_lst]

plt.plot(x_lst1, y_lst,'r', label='broadcast')

plt.xlabel('x')
plt.ylabel('f(s,i)')
plt.savefig('tempFile/exp')'''


def sigmoid(z):
   # h = np.zeros((len(z), 1))  # 初始化，与z的长度一置

    h = 1.0 / (1.0 + np.exp(-z))
    return h

y_lst=[]
x_lst=range(1,1000)
for i in x_lst:
    y=sigmoid(-i/1000)
    y_lst.append(y)


x_lst1=[-x/1000 for x in x_lst]

plt.plot(x_lst1, y_lst,'r', label='broadcast')

plt.xlabel('x')
plt.ylabel('f(s,i)')
plt.savefig('tempFile/sig')
