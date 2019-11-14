# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:06:59 2019

@author: Talha
"""

import numpy as np
from scipy import stats
import pyeeg
from entropy import *


def mean(data):
    return np.mean(data,axis=0)
    
def std(data):
    return np.std(data,axis=0)

def ptp(data):
    return np.ptp(data,axis=0)

def var(data):
        return np.var(data,axis=0)

def skewness(data):
    return stats.skew(data,axis=0)

def kurtosis(data):
    return stats.kurtosis(data,axis=0)

def app_epy(data):
    result=[]
    for i in data.T:
        result.append(app_entropy(i, order=2, metric='chebyshev'))
    return np.array(result)

def perm_epy(data):
    result=[]
    for i in data.T:
        result.append(perm_entropy(i, order=3, normalize=True))
    return np.array(result)

def svd_epy(data):
    result=[]
    for i in data.T:
        result.append(svd_entropy(i, order=3, delay=1, normalize=True))
    return np.array(result)

def spectral_epy(data):
    result=[]
    for i in data.T:
        result.append(spectral_entropy(i, 100, method='welch', normalize=True))
    return np.array(result)

def sample_epy(data):
    result=[]
    for i in data.T:
        result.append(sample_entropy(i, order=2, metric='chebyshev'))
    return np.array(result)


def katz(data):
    result=[]
    for i in data.T:
        result.append(katz_fd(i))
    return np.array(result)

def higuchi(data):
    result=[]
    for i in data.T:
        result.append(higuchi_fd(i))
    return np.array(result)


def petrosian(data):
    result=[]
    for i in data.T:
        result.append(petrosian_fd(i))
    return np.array(result)

def concatenate(x):
    return np.concatenate((mean(data),std(data),ptp(data),var(data),skewness(data),kurtosis(data),
                      app_epy(data),perm_epy(data),svd_epy(data),spectral_epy(data),sample_epy(data),
                      katz(data),higuchi(data),petrosian(data)),axis=0)
features2=[]
for i in range(36):
    data=np.load('D:/Datasets/EEG dataset/mental artithematic/preprocessed_data/subject_2/subject_2_{}.npy'.format(i))
    features2.append(concatenate(data))



features1=[]
for i in range(36):
    data=np.load('D:/Datasets/EEG dataset/mental artithematic/preprocessed_data/subject_1/subject_1_{}.npy'.format(i))
    features1.append(concatenate(data))


x1=np.array(features1)        
x2=np.array(features2)      

X=np.concatenate((x1,x2),axis=0)

y=np.concatenate(((np.zeros(x1.shape[0])),(np.ones(x2.shape[0]))))

np.save('feature.npy',X)
np.save('label.npy',y)