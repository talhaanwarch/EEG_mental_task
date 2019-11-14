# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:02:31 2019

@author: Talha
"""
import numpy as np
X=np.load('features/nonorm_epochs_mean_feature.npy')
y=np.load('features/label.npy')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA,KernelPCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

pca=KernelPCA()

clf = SVC(kernel="rbf")
n_components = [10,15,20,25,30,35,40]
Cs = [ 1,2,3,5,7,10,15,30,50,70,100]
gammas = [0.001,0.01,0.02,0.03,0.04,0.05]


scalar = Normalizer()

#pipe = Pipeline(steps=[('transformer', scalar),('pca', pca), ('estimator', clf)])
pipe = Pipeline(steps=[('pca', pca), ('estimator', clf)])

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              estimator__C=Cs,estimator__gamma=gammas),cv=5,n_jobs=-1)

results = estimator.fit(X,y)
results.best_params_
scores = cross_val_score(estimator, X, y, cv = 5)
print('average accuracy : ',np.array(scores).mean(),np.std(np.array(scores)))


