from datasets import ecgDataset
import numpy as np
from torch.utils.data import DataLoader
import random
import process_utils
import matplotlib.pyplot as plt

import torch
from model7500 import dp
# d = ecgDataset("./train-new3-balanced.hdf5", isTrain=False)
# train_loader = DataLoader(dataset=d, batch_size=64, shuffle=True, drop_last=True)
# for i, datas in enumerate(train_loader):
#     data,label = datas
#     data = data[:,:,0,0:2]
#     for idx,ecg in enumerate(data):
#         print(label[idx])
#         if label[idx][0] == 1 and label[idx][1] == 0:
#             marker = '>'
#             color = 'g'
#         elif label[idx][0] == 0 and label[idx][1] == 1:
#             marker = '<'
#             color='b'
#         elif label[idx][0] == 1 and label[idx][1] == 1:
#             marker = '*'
#             color='r'
#         else:
#             marker = '.'
#             color='k'
#         for c_n,c in enumerate(ecg): 
#             # plt.figure()
#             if c_n != 0:
#                 continue
#             plt.scatter(c[0],c[1], marker=marker,c=color)
#     plt.show()


print(torch.__version__)
d = ecgDataset("train-7500-ori.hdf5", isTrain=False)
train_loader = DataLoader(dataset=d, batch_size=1, shuffle=True, drop_last=True)
network = dp().train()
count = 0
for i, datas in enumerate(train_loader):
    data,label = datas
    out = network(data)
    # data = data[:,:,0,0:2]
    for idx,ecg in enumerate(data):
        # print(label[idx])
        # print(ecg.shape)
        ecg = ecg.numpy()*5
        # for c in ecg:
        #     plt.plot(np.arange(0,len(c)),c)
        plt.plot(np.arange(0,7500),ecg[0],color="g")
        
        out = out.numpy()
        plt.plot(np.arange(0,7500), out[0][0],color="b")
        
        print(ecg[0][:10],out[0][0][:10])
        
        plt.legend(['origin', 'after dropout'])
        plt.title("dropout"),plt.show()
        
        # r = process_utils.process_ecg(ecg, label[idx])
    # plt.show()