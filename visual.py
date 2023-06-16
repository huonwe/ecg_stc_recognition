from datasets import ecgDataset
import numpy as np
from torch.utils.data import DataLoader
import random
import process_utils
import matplotlib.pyplot as plt

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


d = ecgDataset("./train-balanced.hdf5", isTrain=False)
train_loader = DataLoader(dataset=d, batch_size=64, shuffle=True, drop_last=True)
for i, datas in enumerate(train_loader):
    data,label = datas
    # data = data[:,:,0,0:2]
    for idx,ecg in enumerate(data):
        print(label[idx])
        print(ecg.shape)
        ecg = ecg.numpy()
        _ = process_utils.process_ecg(ecg, label[idx])

    plt.show()