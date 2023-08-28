#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:36:36 2023

@author: dina
"""
import os
import pandas as pd
import numpy as np
import pickle5 as pickle
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

#### KNN dependancies
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef
import faiss 

#input_path =""
label_path = "dataset/" ### according to the file structure
### open embedded vectors
with open( "X_trainp2_2.pkl", "rb") as fh:
  X_train = pickle.load(fh)
  
with open( "X_valtestp2_2.pkl", "rb") as fh:
  X_valtest = pickle.load(fh)
  
with open( "X_testp2_2.pkl", "rb") as fh:
  X_test = pickle.load(fh) 

y_train = pd.read_table(os.path.join(label_path, "y_train.csv"), sep = ',',index_col = 0)
y_valtest = pd.read_table(os.path.join(label_path, "y_valtest.csv"), sep = ',',index_col = 0)
y_test = pd.read_table(os.path.join(label_path, "y_test.csv"), sep = ',',index_col = 0)


class_size = np.array(y_train.value_counts()) 
print(class_size)
#### select some percentage of samples

train = pd.concat([X_train,y_train], axis=1)
percentage = 0.2
print("percentage:",percentage)
n= int(len(y_train) * percentage)
subset = train.sample(n=n,random_state = 10)
print(subset["Label"].value_counts())
y_train = subset["Label"]
X_train = subset.drop("Label",axis=1)
print(X_train)

# use a single GPU  
res = faiss.StandardGpuResources()
### converting dataframe to numpy
X_train = X_train.to_numpy()


X_valtest = X_valtest.to_numpy()


X_test = X_test.to_numpy()
print(y_train.shape,X_train.shape)
y_train = y_train.to_numpy().flatten()
y_test = y_test.to_numpy().flatten()
y_valtest = y_valtest.to_numpy().flatten()

neigh = 0
print("neigh", neigh)
 

d = X_train.shape[1]

index = faiss.IndexFlatL2(d)   # build the index
## Using an IVF index
gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index)

# assert not gpu_index_ivf.is_trained
gpu_index_ivf.train(X_train)        # add vectors to the index
# assert gpu_index_ivf.is_trained
# print(gpu_index_flat.is_trained)
gpu_index_ivf.add(X_train)                  # add vectors to the index
print(gpu_index_ivf.ntotal)

k = 5 ### number of neighbors

D, I = gpu_index_ivf.search(X_valtest, k)
D_test, I_test = gpu_index_ivf.search(X_test, k)



print(class_size)
print("K",k)
votes = y_train[I]
votes_test = y_train[I_test]


weight = np.array([0.9893, 0.9718]) ## precompute the weights
print("weight", weight)
predictions = [np.argmax((np.bincount(x, minlength=2) * weight)) for x in votes]
predictions_test = [np.argmax((np.bincount(x, minlength=2) * weight)) for x in votes_test]

# print(votes[:,2])
# report = classification_report(y_valtest, predictions, digits=4, output_dict=True)
# report = pd.DataFrame(report).transpose()
# print("weight-10",report)

report = classification_report(y_test, predictions_test, digits=4, output_dict=True)
report = pd.DataFrame(report).transpose()
# report = report.append(k)
print("wustl-Tabmlp5_1",report)
report.to_csv("model2_20_classesize.csv")



