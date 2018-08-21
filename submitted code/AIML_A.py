# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

df=pd.read_excel('statsis.xlsx')

#mean
print("mean=",df["Salary in Rs."].mean())
#median
print("median=",df["Salary in Rs."].median())
#mode
print("mode=",df["Salary in Rs."].mode()[0])
#variance
print("variance=",df["Salary in Rs."].var())
#std dev
print("standard deviation=",df["Salary in Rs."].std())
