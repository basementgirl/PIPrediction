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

'''
def sigmoid(z):
   # h = np.zeros((len(z), 1))  # 初始化，与z的长度一置

    h = 1.0 / (1.0 + np.exp(-z))
    return h

y_lst=[]
x_lst=range(-50,50,1)
x_lst1=[x/10 for x in x_lst]

for i in x_lst1:
    y=sigmoid(i)
    y_lst.append(y)



plt.plot(x_lst1, y_lst,'r', label='broadcast')

plt.xlabel('z')
plt.ylabel('h(z)')
plt.title('Sigmoid Function')
plt.savefig('tempFile/sig')'''




def f(z):
   # h = np.zeros((len(z), 1))  # 初始化，与z的长度一置
    h = -np.log(1-z)
    return h

y_lst=[]
x_lst=range(1,100,1)
x_lst1=[x/100 for x in x_lst]

for i in x_lst1:
    y=f(i)
    y_lst.append(y)



plt.plot(x_lst1, y_lst,'r', label='broadcast')

plt.xlabel('h(z)')
plt.ylabel('cost(h(z),y)')
plt.title('if y=0')
plt.savefig('tempFile/log1')
