# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:05:50 2019

@author: Talha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
import re
import mne
from scipy import stats


from keras.models import Sequential
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D,MaxPooling1D,Dropout,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from keras.layers import LeakyReLU
import keras_metrics as km


import glob
import os

path = r'data/' # use your path
all_files = glob.glob(os.path.join(path, "*.edf")) 

data1=[]
data2=[]
for filename in (all_files):
    if int(re.findall(r'\d+',filename)[1])==1:
        data1.append(mne.io.read_raw_edf(filename,preload=True).get_data()[0:-3,::]);
    else:
        data2.append(mne.io.read_raw_edf(filename,preload=True).get_data()[0:-3,::]);

len(data1)

data1x=[]
data2x=[]
for i in range(36):
    data1x.append(stats.zscore(data1[i] ))
for i in range(36):
    data2x.append(stats.zscore(data2[i]))


from sklearn.decomposition import FastICA, PCA
from scipy import signal as ss

#ica = FastICA(n_components=1)
pca=PCA(1)

data1pca=[]
data2pca=[]
for i in range(36):
    data1pca.append(pca.fit_transform(data1x[i].T).ravel())
for i in range(36):
    data2pca.append(pca.fit_transform(data2x[i].T).ravel())

sample=500
seed=0
data2pca[0].shape

data_2=[]
for i in range(36):
    data_2.append(np.reshape(data2pca[i],(-1,sample)))
data_1=[]
for i in range(36):
    data_1.append(np.reshape(data1pca[i],(-1,sample)))    

data1=np.concatenate(data_1)
data2=np.concatenate(data_2)
data1.shape,data2.shape

y1=np.zeros(data1.shape[0])
y2=np.ones(data2.shape[0])
# y=np.concatenate((y1,y2))
# y=keras.utils.to_categorical(y,2)
# y.shape

data1.shape

def np_pop(my_array,i,j):
    pop = my_array[i:j]
    new_array = np.delete(my_array,np.s_[i:j],0)
    return [pop,new_array]
#[B,poparow] = poprow(A,0,2)
accuracy=[]
precision=[]
recall=[]
loss=[]
val_loss=[]
val_accuracy=[]

for p in range(1,6):
    X_train_1,X_test_1=np_pop(data1,int(p*data1.shape[0]/6),int(data1.shape[0]))
    X_train_2,X_test_2=np_pop(data2,int(p*data2.shape[0]/6),int(data2.shape[0]))
    y_train_1,y_test_1=np_pop(y1,int(p*y1.shape[0]/6),int(y1.shape[0]))
    y_train_2,y_test_2=np_pop(y2,int(p*y2.shape[0]/6),int(y2.shape[0]))
    print(X_train_1.shape,X_train_2.shape,y_train_1.shape,y_train_2.shape)
    X_train=np.concatenate((X_train_1,X_train_2),axis=0)
    y_train=np.concatenate((y_train_1,y_train_2))
    print(X_train.shape,y_train.shape)    
    
    
    X_test=np.concatenate((X_test_1,X_test_2),axis=0)
    y_test=np.concatenate((y_test_1,y_test_2))
    print(X_test.shape,y_test.shape)
    
    idx=np.random.RandomState(seed=42).permutation(len(y_train))
    X_train=X_train[idx]
    y_train=y_train[idx]
    
    idx=np.random.RandomState(seed=42).permutation(len(y_test))
    X_test=X_test[idx]
    y_test=y_test[idx]
    
    X_train=np.reshape(X_train,(X_train.shape[0], sample,1))
    X_test=np.reshape(X_test,(X_test.shape[0], sample,1))
    
    print(X_train.shape,X_test.shape)
    
    y_train=keras.utils.to_categorical(y_train,2)
    y_test=keras.utils.to_categorical(y_test,2)
    
    y_test.shape,y_train.shape
    
    model = Sequential()
    model.add(Conv1D(20, (8), input_shape=(sample,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
       
    model.add(Conv1D(20, (6)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    

        
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
       
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy',km.binary_precision(), km.binary_recall()])
    
    model.summary()
    
    # count number of parameters in the model
    numParams    = model.count_params()    
    
    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='checkpoint.h5', verbose=1,
                                   save_best_only=True)
    
    class_weights = {0:1, 1:2}
    
    fittedModel = model.fit(X_train, y_train, batch_size =100, epochs = 30, 
                            verbose = 2, validation_data=(X_test, y_test),
                            callbacks=[checkpointer], class_weight = class_weights)
    
    model.history.history.keys()
    
    accuracy.append(np.array(model.history.history['acc']))
    val_accuracy.append(np.array(model.history.history['val_acc']))
    precision.append(np.array(model.history.history['val_precision']))
    recall.append(np.array(model.history.history['val_recall']))
    loss.append(np.array(model.history.history['loss']))
    val_loss.append(np.array(model.history.history['val_loss']))

    K.clear_session()
result=np.concatenate((accuracy,val_accuracy,precision,recall,loss,val_loss))  

print('training accuracy: ',np.array(accuracy).mean())
print('training loss: ',np.array(loss).mean())
print('precision :',np.array(precision).mean())
print('recall :', np.array(recall).mean() )
print('validation accuracy: ',np.array(val_accuracy).mean())
print('validation loss :', np.array(val_loss).mean() )
