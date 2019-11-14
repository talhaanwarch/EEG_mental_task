# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:30:40 2019

@author: Talha
"""

import glob
import os
import mne
import re
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

path = r'data/' # use your path
all_files = glob.glob(os.path.join(path, "*.edf")) 

i=0
j=0
for filename in (all_files):
    if int(re.findall(r'\d+',filename)[1])==1:
        data=mne.io.read_raw_edf(filename,preload=True).get_data()[0:-3,10000:40000].T
        scaler=MaxAbsScaler()        
        data=scaler.fit_transform(data)
        np.save('D:/Datasets/EEG dataset/mental artithematic/preprocessed_data/subject_1/subject_1_{}'.format(i),data)
        i+=1
    else:
        data=mne.io.read_raw_edf(filename,preload=True).get_data()[0:-3,0:30000].T
        scaler=MaxAbsScaler()        
        data=scaler.fit_transform(data)
        np.save('D:/Datasets/EEG dataset/mental artithematic/preprocessed_data/subject_2/subject_2_{}'.format(j),data)
        j+=1


