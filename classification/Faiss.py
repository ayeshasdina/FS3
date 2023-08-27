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

input_path ="/project/dmani2_uksr/dina_workplace/wideDeep/Wustl_EHMS/contrastive_learning/random_5/Embedding_Data/Datatabmlp5/"
label_path = "/project/dmani2_uksr/dina_workplace/wideDeep/Wustl_EHMS/contrastive_learning/random_5/Embedding_Data/Datatabmlp5/"
### stored embedding
with open(os.path.join(input_path, "X_trainp2_2.pkl"), "rb") as fh:
  X_train = pickle.load(fh)
  
with open(os.path.join(input_path, "X_valtestp2_2.pkl"), "rb") as fh:
  X_valtest = pickle.load(fh)
  
with open(os.path.join(input_path, "X_testp2_2.pkl"), "rb") as fh:
  X_test = pickle.load(fh) 

y_train = pd.read_table(os.path.join(label_path, "y_train.csv"), sep = ',',index_col = 0)
y_valtest = pd.read_table(os.path.join(label_path, "y_valtest.csv"), sep = ',',index_col = 0)
y_test = pd.read_table(os.path.join(label_path, "y_test.csv"), sep = ',',index_col = 0)


class_size = np.array(y_train.value_counts()) 
print(class_size)
#### select some percentage of samples
print("model:X_trainp2_2.pkl")
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
def transform_pred(y_index, y_truth):
    # y_truth = y_truth.to_numpy()
    y_pred = []
    for i , j in enumerate(y_index[:,neigh]):
        y_pred.append(y_truth[j])
    
    return y_pred  

d = X_train.shape[1]

index = faiss.IndexFlatL2(d)   # build the index
## Using an IVF index
# nlist = 200
# quantizer = faiss.IndexFlatL2(d)  # the other index
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# index.nprobe = 150
# make it into a gpu index
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index)

# assert not gpu_index_ivf.is_trained
gpu_index_ivf.train(X_train)        # add vectors to the index
# assert gpu_index_ivf.is_trained
# print(gpu_index_flat.is_trained)
gpu_index_ivf.add(X_train)                  # add vectors to the index
print(gpu_index_ivf.ntotal)

k = 10

D, I = gpu_index_ivf.search(X_valtest, k)
D_test, I_test = gpu_index_ivf.search(X_test, k)

y_pred = transform_pred(I_test, y_train)
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
report = pd.DataFrame(report).transpose()
print("top-5",report)
# report.to_csv("reportminerTab5_top.csv")


print(class_size)
print("K",k)
votes = y_train[I]
votes_test = y_train[I_test]
predictions = [np.argmax((np.bincount(x, minlength=2) / class_size.astype(np.float32))) for x in votes]
predictions_test = [np.argmax((np.bincount(x, minlength=2) / class_size.astype(np.float32))) for x in votes_test]
# weight = np.array([0.92, 0.93,0.99, 0.98,  0.99])
# weight = np.array([0.9893, 0.9718])
# print("weight", weight)
# predictions = [np.argmax((np.bincount(x, minlength=2) * weight)) for x in votes]
# predictions_test = [np.argmax((np.bincount(x, minlength=2) * weight)) for x in votes_test]

# print(votes[:,2])
report = classification_report(y_valtest, predictions, digits=4, output_dict=True)
report = pd.DataFrame(report).transpose()
print("weight-10",report)

report = classification_report(y_test, predictions_test, digits=4, output_dict=True)
report = pd.DataFrame(report).transpose()
# report = report.append(k)
print("wustl-Tabmlp5_1",report)
report.to_csv("model2_20_classesize.csv")
MCC = matthews_corrcoef(y_test, predictions_test)
print("MCC", MCC)


