# encoding=utf8
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader
import random
import process_utils
import matplotlib.pyplot as plt

class ecgTestset(Dataset):
    def __init__(self, hdf5Path):
        self.data = h5py.File(hdf5Path, "r")
        self.len = len(self.data["labels"])
        
    def __getitem__(self, item):
        prepared_data = np.array(self.data["data"][item])
        data = torch.from_numpy(prepared_data)
        label = torch.from_numpy(self.data["labels"][item])

        return data, label
    
    def __len__(self):
        return self.len

    def __del__(self):
        self.data.close()
        
class ecgDataset(Dataset):
    def __init__(self, hdf5Path, isTrain=True, Sub=False, shift=False):
        self.isTrain = isTrain
        self.data = h5py.File(hdf5Path, "r")
        self.len = len(self.data["labels"])
        self.shift = shift
        self.isSub = Sub
        if self.isSub:
            self.avg = torch.from_numpy(ecg_avg(self.data))
        
    def __getitem__(self, item):
        if self.isTrain:
            prepared_data = np.array(self.data["data"][item])
            data = torch.from_numpy(prepared_data)
            label = torch.from_numpy(self.data["labels"][item])
            if self.shift:
                shift = random.randint(0,1000)
                data_shift = torch.zeros((12,7500))
                data_shift[:,shift:] = data[:,0:(7500-shift)]
                data_shift[:,0:shift] = data[:,(7500-shift):]
                data = data_shift
        else:
            prepared_data = np.array(self.data["data"][item])
            data = torch.from_numpy(prepared_data)
            label = torch.from_numpy(self.data["labels"][item])
        if self.isSub:
            # plt.title(label)
            # plt.plot(np.arange(0,250),data[0],c='g')
            scale = torch.max(data,dim=1).values / torch.max(self.avg,dim=1).values
            template = torch.mul(self.avg , scale.view(1,12).transpose(1,0))
            data_max_t = torch.argmax(data,dim=1)
            template_max_t = torch.argmax(template,dim=1)
            offset = torch.sub(data_max_t, template_max_t)
            template_offset = torch.zeros_like(template)
            for c_idx, c_offset in enumerate(offset):
                if c_offset == 0:
                    template_offset[c_idx] = template[c_idx]
                elif c_offset > 0:
                    template_offset[c_idx][c_offset:] = template[c_idx][:250-c_offset]
                    template_offset[c_idx][:c_offset] = template[c_idx][0]
                elif c_offset < 0:
                    c_offset = - c_offset
                    template_offset[c_idx][:250-c_offset] = template[c_idx][c_offset:]
                    template_offset[c_idx][:c_offset] = template[c_idx][-1]
            # print(offset)
            # plt.plot(np.arange(0,250),template_offset[0],c='b')
            data = torch.sub(data, template_offset).float()
            # plt.plot(np.arange(0,250),data[0],c='r')
            # plt.show()
        return data, label
    
    def __len__(self):
        return self.len

    def __del__(self):
        self.data.close()


def ecg_avg(data: h5py.File) -> np.array:
    sum = np.zeros((12,250))
    count = np.zeros((12))
    
    for idx, ecg_label in enumerate(data["labels"]):
        if ecg_label[2] == 1:
            for c_idx,c in enumerate(data["data"][idx]):
                if np.max(c) == 0:
                    continue
                sum[c_idx] += c
                count[c_idx] += 1
    avg = np.zeros((12,250))
    for c_idx in range(0,12):
        avg[c_idx] = sum[c_idx] / count[c_idx]
        # plt.plot(np.arange(0,250),avg[c_idx])
    return avg

if __name__ == "__main__":
    d = ecgDataset("./train-250-balanced.hdf5", isTrain=False, isAvg=True)
    train_loader = DataLoader(dataset=d, batch_size=1, shuffle=True, drop_last=True)
    for i, data in enumerate(train_loader):
        ecg, label = data
        ecg = ecg[0]
        # 12 7500
        time = np.arange(0,250)
        
        # plt.title(label[0])
        # for c in ecg:
        #     plt.plot(time,c)
        # plt.show()
