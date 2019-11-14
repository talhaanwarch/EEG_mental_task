# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:25:06 2019

@author: Talha
"""

 
#%%load dataset
import numpy as np
X=np.load('feature.npy')
y=np.load('label.npy')
#%% split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)
#%% scale dataset
#Note scaling does not work well, particulary Standard scaler in our data, becuase i think we normalized raw data already
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#%%  check accuracy without feature selection
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,y_pred))
print('accuracy is ',accuracy_score(y_test, y_pred))

#%% print accuracy by feeding single feature
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier

feature_list=['mean','std','ptp','var','skewness','kurtosis','app_epy','perm_epy',
             'svd_epy','spectral_epy','sample_epy','katz','higuchi','petrosian']
for i,f in zip(range(0,X_train.shape[1],18),feature_list):
    clf=KNeighborsClassifier(3)
    clf.fit(X_train[:,i:i+18],y_train)
    y_pred = clf.predict(X_test[:,i:i+18])
    acc=accuracy_score(y_test, y_pred)
    print(f,acc)

#%% remove feature with accuracy less than a threshold
threshold=0.41
f=0   
start=0
fe=feature_list    
for i in range(14):
    clf=KNeighborsClassifier(3)
    clf.fit(X_train[:,start:start+18],y_train)
    y_pred = clf.predict(X_test[:,start:start+18])
    acc=accuracy_score(y_test, y_pred)
    print(fe[f],acc)        
    #ind=fe.index(f)
    
    if acc<0.45:
        X_train=np.delete(X_train,np.s_[start:start+18],axis=1)
        X_test=np.delete(X_test,np.s_[start:start+18],axis=1)
        fe.pop(f)
    else:
        start+=18
        f+=1
        
print('feaature left',fe)        
#%%check accuracy again
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,y_pred))
print('accuracy is ',accuracy_score(y_test, y_pred))

#%%cross validation with standard scaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

scalar = StandardScaler()
clf = KNeighborsClassifier(2)

pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])

cv = KFold(n_splits=5)
scores = cross_val_score(pipeline, X, y, cv = cv)
print('average accuracy : ',np.array(scores).mean(),np.std(np.array(scores)))

#%%cross validation without standard scaler
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(3)
scores = cross_val_score(clf, X, y, cv=6)
print('average accuracy : ',np.array(scores).mean(),np.std(np.array(scores)))

