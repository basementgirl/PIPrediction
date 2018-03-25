import numpy as np
from sklearn.metrics import mean_absolute_error

import math

'''
p=0.6
n=5
t=1.96
m2=1+1/n*pow(t,2)

m1=p+1/(2*n)*pow(t,2)

m3=t*math.sqrt(p*(1-p)/n+t**2/(4*n**2))
print((m1-m3)/m2)'''


p=0.4
z=1.96

y=[z*math.sqrt(4*n*(1-p)*p+pow(z,2))/(n+pow(z,2)) for n in range(2,50,2)]
print(y)