# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import cross_validation,linear_model

df = pd.read_csv('carMPG.csv',sep=',',index_col=None)

print(df.head())
#im=Imputer(missing_values='NaN',strategy='mean',axis=0)
#df=im.fit_transform(df)
#print(df)

#print(df.dtypes)
#print(df.describe())
 
#Load the data
model=pd.DataFrame(df, columns = ['MPG', 'Acceleration'])

#model[:10]

#Split the data into training/testing sets

X=np.array(model.drop(["MPG"],1))
y=np.array(model["MPG"])

# Split the targets into training/testing sets
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.20)

# Create linear regression object
regr=linear_model.LinearRegression()

print(X_train.shape, y_train.shape)

#reshaping training and test sets

X_train=X_train.reshape(len(X_train),1)
y_train=y_train.reshape(len(y_train),1)

print(X_train.shape, y_train.shape)

regr.fit(X_train,y_train)

X_test=X_test.reshape(len(X_test),1)
y_test=y_test.reshape(len(y_test),1)

print(X_test.shape,y_test.shape)

#variance score:R^2 
print("accuracy using R^2:",regr.score(X_test,y_test))
#coef, intercept
print("Coefficient is:",regr.coef_,"\nIntercept is:",regr.intercept_)

pred=regr.predict(X_test)

#mean squared error
print("Mean squared error: %.2f"% np.mean((pred-y_test)**2))

#Plot outputs
plt.scatter(X_test,y_test,color='black')
plt.plot(X_test, pred, color='blue',linewidth=1)

    


