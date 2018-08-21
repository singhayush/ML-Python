# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:38:23 2018

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('carMPG.csv',sep=',',index_col=None)

model=pd.DataFrame(df, columns = ['MPG', 'Acceleration'])

def compute_error(c,m,points):
    toterr=0
    for i in range(len(points)):
        x=points[i,2]
        y=points[i,1]
        toterr+=(y-(m*x+c))**2
        
    return toterr/float(len(points))

def step_gradient(c_cur,m_cur,points,lr):
    c_grad=0
    m_grad=0
    n=float(len(points))
    for i in range(0,len(points)):
        x=points[i,2]
        y=points[i,1]
        c_grad+= -(2/n) * (y-(m_cur*x) +c_cur)
        m_grad+= -(2/n) * x * (y-(m_cur*x) +c_cur)
        
    new_c=c_cur-(lr*c_grad)
    new_m=m_cur-(lr*m_grad)
    return(new_c,new_m)
         
def grad_desc_runner(points,start_c,start_m,lr,num_iter):
    c=start_c
    m=start_m
    for i in range(num_iter):
        c,m=step_gradient(c,m,points,lr)
    return(c,m)
    

points=model
points=points.reset_index().values

#initializing parameters
lr=0.00012
num_iter=80
ini_c=0
ini_m=0

print("starting grad descent at c = {0},m = {1}, error = {2}".
      format(ini_c,ini_m,compute_error(ini_c,ini_m,points)))

[c,m]=grad_desc_runner(points,ini_c,ini_m,lr,num_iter)

print("after {0} iterations c = {1},m = {2},error = {3}".
      format(num_iter,c,m,compute_error(c,m,points)))

#line form
print("line form: y={0}x+{1}".format(m,c))

for i in range(0,len(points)):
    x=points[i,2]
    y=points[i,1]
    plt.scatter(x,y,color='red')
plt.plot(model.Acceleration,(m*model.Acceleration + c),color='blue')