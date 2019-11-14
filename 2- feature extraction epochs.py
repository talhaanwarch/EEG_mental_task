# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:59:37 2019

@author: Talha
"""


import numpy as np
features1=[]
for i in range(36):
    feature=[]
    a=np.load('D:/Datasets/EEG dataset/mental artithematic/preprocessed_data/subject_1/subject_1_{}.npy'.format(i))
    b=a.reshape(-1,3000,18)
    for i in b:
        feature.append(concatenate(i))
    features1.append(np.mean(np.array(feature),axis=0))


import numpy as np
features2=[]
for i in range(36):
    feature=[]
    a=np.load('D:/Datasets/EEG dataset/mental artithematic/preprocessed_data/subject_2/subject_2_{}.npy'.format(i))
    b=a.reshape(-1,3000,18)
    for i in b:
        feature.append(concatenate(i))
    features2.append(np.mean(np.array(feature),axis=0))



x1=np.array(features1)        
x2=np.array(features2)      

X=np.concatenate((x1,x2),axis=0)

y=np.concatenate(((np.zeros(x1.shape[0])),(np.ones(x2.shape[0]))))
np.save('features/nonorm_epochs_mean_feature.npy',X)
np.save('features/label.npy',y)