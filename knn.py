# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:46:28 2018

@author: AY370389
"""

import numpy as np
import pandas as pd
#import sklean
from sklearn import neighbors, preprocessing, cross_validation
cols=['age','YEO','Positive','status']
df=pd.read_csv("haberman.data.txt",sep=',',header=None,index_col=None)
df.columns=cols

print(df.head(10))
x=np.array(df['age']).reshape(-1,1)
y=np.array(df['status'])
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)

knn=neighbors.KNeighborsClassifier()

knn.fit(x_train,y_train)

pred=knn.predict(x_test)
print("prediction=",pred)
print("y_test=",y_test)

#accuracy check

print("accuracy score={0}".format(knn.score(x_test,y_test)))
print(knn.get_params(deep=True))