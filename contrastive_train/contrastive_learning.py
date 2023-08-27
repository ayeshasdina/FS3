#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:03:28 2023

@author: dina
"""
import pandas as pd
import numpy as np
import os
import pdb
# import pickle as pickle
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
### WideDeep dependency
import torch
import torch.nn as nn


from pytorchtools import EarlyStopping
from pytorch_widedeep.preprocessing import TabPreprocessor
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
from pytorch_metric_learning import miners,losses
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.models import WideDeep
import torch.nn.functional as F
from pytorch_metric_learning.distances import DotProductSimilarity

# model_path = "pretrained_models"
model_path = "/pretrain/" ## change the directory in case you have folder stracture 
input_path = "/dataset/" ## change the directory in case you have folder stracture 


model_name = "Tabmlp_EM70.pt"

trained_model_name = "tabmlp_ran5.pt" ## change it accordingly if you want
num_sample = 5
batch_size = 5
epoch_num = 100
lr = 0.001
wd = 1e-6
#debug = False
patience = 30


train_file = "trainRawData.csv"
valtest_file = "valtestRawData.csv"
target_col = "Label"
le = LabelEncoder()

train = pd.read_csv(os.path.join(input_path,train_file ), index_col=0)
train = train.groupby(target_col).sample(num_sample)
print(train.index)
label = le.fit_transform(train[target_col])
valtest = pd.read_csv(os.path.join(input_path, valtest_file), index_col=0)
val_label = le.fit_transform(valtest[target_col])


train["Sport"] = le.fit_transform(train["Sport"])
train["Dport"] = le.fit_transform(train["Dport"])


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



### Wide Preprocess
tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols,
    continuous_cols=continuous_cols,
    # with_attention=True,
    # with_cls_token=True,  # this is optional
)


#### activating GPU and CPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
#### preparing pytorch Dataset
class PreparingDataset(Dataset):
    def __init__(self, X1,Y):
        
        self.X1 = X1
        # self.X2 = X2
        self.y = Y
        print("Done")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # print(idx)
        return self.X1[idx], self.y[idx]
##### functions to train the model    
def get_optimizer(model, lr=lr, wd=wd):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
    return optim, scheduler

def train_model(model, optim, train_dl, i,debug=False):
    model.train()
    total = 0
    sum_loss = 0
    count = 0
    for x1, y in train_dl:
        # print("X1",x1)
        # print("X2",x2)

        count += 1
        # if debug and count == 65:
        #     pdb.set_trace()
        batch = y.shape[0]
        output = model(x1)
        # #### Triplet
        loss_func = losses.TripletMarginLoss(margin=0.1)
        miner = miners.MultiSimilarityMiner(epsilon=0.1)
        hard_pairs = miner(output, y)
        loss = loss_func(output, y,hard_pairs)
        ####contrastive loss
        # loss_func = losses.ContrastiveLoss()
        # loss = loss_func(output,y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
        # spot_check.to_csv("Dot_product_csv/Epoch"+str(i)+"_batch"+ str(count)+".txt")

    return sum_loss/total
def val_loss(model, val_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1,  y in val_dl:
        current_batch_size = y.shape[0]
        output = model(x1)

        #### Triplet
        loss_func = losses.TripletMarginLoss(margin=0.1)
        miner = miners.MultiSimilarityMiner(epsilon=0.1)
        hard_pairs = miner(output, y)
        loss = loss_func(output, y,hard_pairs)
        ####contrastive loss
        # loss_func = losses.ContrastiveLoss()
        # loss = loss_func(output,y)
        
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        # pred = torch.max(out, 1)[1]
        # correct += (pred == y).float().sum().item()
    # print("valid loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))
    print("valid loss %.3f " % (sum_loss/total))
    print(output.size())


    # return (sum_loss/total, correct/total)
    return (sum_loss/total)
def train_loop(model, epochs, lr=lr, wd=wd,patience = 5):
    optim,scheduler = get_optimizer(model, lr = lr, wd = wd)
    train_loss = []
    valid_loss = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, path = trained_model_name,verbose=True)
    for i in range(epochs):
        print("Epoch:",i)
        loss = train_model(model, optim, train_dl, i, debug)
        train_loss.append(loss) 
        print("training loss: ", loss)
        v_loss = val_loss(model, val_dl)
        # scheduler.step(v_loss)
        valid_loss.append(v_loss)
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model

        early_stopping(v_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return (train_loss, valid_loss)
#### declaring the Model with pretrained model.        


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features1 = pretrained_model
        # self.features2 = pretrained_model
    def forward(self, x1):
        x1 = self.features1(x1)

        return x1



# contrastive_denoising_model = torch.load("encoder_decoder_model.pt",map_location= torch.device('cpu'))
# contrastive_denoising_model = torch.load("pretrained_weights/encoder_decoder_model.pt")
contrastive_denoising_model = torch.load(os.path.join(model_path,model_name))
pretrained_model = contrastive_denoising_model.encoder
# pretrained_model = contrastive_denoising_model.model
print(pretrained_model)



### preparing data for pretrain model
train = tab_preprocessor.fit_transform(train)
valtest = tab_preprocessor.fit_transform(valtest)

### Label
label = np.array(label)
label = label.ravel() ## csv
val_label = np.array(val_label)
val_label = val_label.ravel() ## csv
#### geting default device from the system
device = get_default_device()
#creating train and valid datasets
train_ds = PreparingDataset(train, label)
val_ds = PreparingDataset(valtest,  val_label)
### using torch dataloader

train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle= True)
val_dl = DataLoader(val_ds, batch_size=batch_size,shuffle= True)
### Dataloader using CPU/GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
#### training the model
model = MyModel()
to_device(model,device)
train_loss, valid_loss = train_loop(model, epochs= epoch_num, lr= lr, wd=wd, patience = patience)

#### plot
# epochs = range(1,epoch_num + 1)
# plt.plot(epochs, train_loss, 'g', label='Training loss')
# plt.plot(epochs, valid_loss, 'b', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("train_valloss5.pdf")
# plt.show()
