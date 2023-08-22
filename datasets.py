# encoding=utf8
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader
import random
import process_utils
import matplotlib.pyplot as plt

class ecgCheckset(Dataset):
    def __init__(self):
        self.data = h5py.File("check.hdf5", "r")
        self.len = len(self.data["name"])
        
    def __getitem__(self, item):
        prepared_data = np.array(self.data["data"][item])
        data = torch.from_numpy(prepared_data)
        # print(self.data["name"][item])
        name = self.data["name"][item]

        return data, name
    
    def __len__(self):
        return self.len

    def __del__(self):
        self.data.close()

class ecgTestset(Dataset):
    def __init__(self,path):
        self.data = h5py.File(path, "r")
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
        if isTrain and "7500" in hdf5Path:
            self.shift = True
        else:
            self.shift = False 
        self.isSub = Sub
        if self.isSub:
            self.avg = torch.from_numpy(ecg_avg(self.data))
        
    def __getitem__(self, item):
        prepared_data = np.array(self.data["data"][item])
        data = torch.from_numpy(prepared_data)
        label = torch.from_numpy(self.data["labels"][item])
        if self.isTrain:
            scale = random.randint(80,120) / 100
            data = data * scale
            if self.shift:
                shift = random.randint(0,1000)
                data_shift = torch.zeros((12,7500))
                data_shift[:,shift:] = data[:,0:(7500-shift)]
                data_shift[:,0:shift] = data[:,(7500-shift):]
                data = data_shift * scale
        # else:
            # scale = torch.randint(70,130) / 100
            # prepared_data = np.array(self.data["data"][item]) * scale
            # data = torch.from_numpy(prepared_data)
            # label = torch.from_numpy(self.data["labels"][item])
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
    d = ecgDataset("./train-7500-ori.hdf5", isTrain=False)
    d2 = ecgDataset("./train-7500-ori.hdf5", isTrain=True)
    train_loader = DataLoader(dataset=d, batch_size=1, shuffle=False, drop_last=True)
    train_loader2 = DataLoader(dataset=d2, batch_size=1, shuffle=False, drop_last=True)
    for i, data in enumerate(train_loader):
        ecg, label = data
        ecg = ecg[0]
        print(ecg.shape)
        # 12 7500
        time = np.arange(0,7500)
        
        # plt.title(label[0])
        for c in ecg:
            print(c.shape)
            plt.plot(time,c)
            break
        for i, data in enumerate(train_loader2):
            ecg, label = data
            ecg = ecg[0]
            for c in ecg:
                plt.plot(time,c)
                break
            break
        plt.show()
