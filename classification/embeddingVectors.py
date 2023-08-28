#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:35:00 2023

@author: dina
"""

import os
import pandas as pd
import pickle5 as pickle
from sklearn.preprocessing import LabelEncoder

### WideDeep dependency
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep, FTTransformer
from pytorch_widedeep.metrics import Accuracy
from sklearn.preprocessing import LabelEncoder,normalize
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep import Tab2Vec
#### KNN dependancies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

### loading model
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features1 = pretrained_model
        # self.features1 = Tab2Vec(self.features1, tab_preprocessor)
        # self.features2 = pretrained_model
    def forward(self, x1, x2):
        x1 = self.features1(x1)
        # x1 = nn.functional.normalize(x1)
        # x2 = self.features1(x2)
        # x2 = nn.functional.normalize(x2)
        # x2 = torch.transpose(x2, 0, 1)
        # x = torch.matmul(x1,x2)
        # print(x)
        # x = -x
        return x

# input_path = "/Users/dina/Documents/Cyber_security/classification/BoT-IoT/Code_2/pickle_Data/input"
input_path = "/dataset/"
#model_path = "/project/dmani2_uksr/dina_workplace/wideDeep/Wustl_EHMS/contrastive_learning/random_5/Datatabmlp5_models/"
model_name = "tabmlp_ran5_5.pt"
print("tabmlp_ran5_5.pt")

train_file = "trainRawData.csv"
valtest_file = "valtestRawData.csv"
test_file ="testRawData.csv"
target_col = "Label"
le = LabelEncoder()

train = pd.read_csv(os.path.join(input_path,train_file ), index_col=0)
y_train = le.fit_transform(train[target_col])
valtest = pd.read_csv(os.path.join(input_path, valtest_file), index_col=0)
y_valtest = le.fit_transform(valtest[target_col])
test = pd.read_csv(os.path.join(input_path, test_file), index_col=0)
y_test = le.fit_transform(test[target_col])



train["Sport"] = le.fit_transform(train["Sport"])
train["Dport"] = le.fit_transform(train["Dport"])

valtest["Sport"] = le.fit_transform(valtest["Sport"])
valtest["Dport"] = le.fit_transform(valtest["Dport"])

test["Sport"] = le.fit_transform(test["Sport"])
test["Dport"] = le.fit_transform(test["Dport"])


cat_embed_cols = [
    # "Proto",
    "SrcAddr",
    "DstAddr",
    # "Sport",
    # "Dport",
    "SrcMac",	
    "DstMac",

]
# 'sIpId','dIpId', ### later use.
continuous_cols = ['Sport','Dport','SrcBytes',	'DstBytes',	'SrcLoad',	'DstLoad',	'SrcGap',	'DstGap',	'SIntPkt',	'DIntPkt',	'SIntPktAct',	'DIntPktAct',	'SrcJitter',	'DstJitter',	
                   'sMaxPktSz',	'dMaxPktSz',	'sMinPktSz',	'dMinPktSz',	'Dur',	'Trans',	'TotPkts',	'TotBytes',	'Load',	'Loss',	'pLoss',	'pSrcLoss',	
                   'pDstLoss',	'Rate',	'Packet_num',	'Temp',	'SpO2',	'Pulse_Rate',	'SYS',	'DIA',	'Heart_rate',	'Resp_Rate',	'ST'	

                   ]
  
 

def load_model():

    ### Wide Preprocess
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        # with_attention=True,
        # with_cls_token=True,  # this is optional
    )

    

    

    ##### phase2 pretrained encoder
    
    model = torch.load(os.path.join(model_name),map_location= torch.device('cpu'))
    pretrained_model = model.features1
    print(pretrained_model)
    #Train
    X_tab = tab_preprocessor.fit_transform(train)
    x1 = torch.from_numpy(X_tab)
    X_vec = pretrained_model(x1)
    
    # valtest
    X_tab = tab_preprocessor.fit_transform(valtest)
    x1 = torch.from_numpy(X_tab)
    X_vec_val = pretrained_model(x1)
    
    # test
    X_tab = tab_preprocessor.fit_transform(test)
    x1 = torch.from_numpy(X_tab)
    X_vec_test = pretrained_model(x1)
    
    X_vec = X_vec.detach().numpy()
    X_vec_val = X_vec_val.detach().numpy()
    X_vec_test = X_vec_test.detach().numpy()
    
    # Convert array to DataFrame
    X_vec = pd.DataFrame.from_records(X_vec)
    X_vec_val = pd.DataFrame.from_records(X_vec_val)
    X_vec_test = pd.DataFrame.from_records(X_vec_test)
    
    X_vec.to_pickle("X_trainp2_5.pkl")
    X_vec_val.to_pickle("X_valtestp2_5.pkl")
    X_vec_test.to_pickle("X_testp2_5.pkl")

    return(X_vec,y_train,X_vec_test,y_test)



if __name__ == "__main__":
    X_train,y_train,X_test,y_test = load_model()
    print("loading KNN...")
    
    
    
    
    
    
    
