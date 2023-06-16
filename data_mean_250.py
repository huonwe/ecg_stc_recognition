#encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio
import pandas as pd

from process_utils import process_ecgv2, process_ecg
from process_utils_250 import process_ecg_writer

class HDF5DatasetWriter:
    def __init__(self, dims=None, outputPath=None, bufSize=100):
        assert dims != None and outputPath != None
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("data", dims, dtype="float32")
        self.labels = self.db.create_dataset("labels", [dims[0],3], dtype="int")
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0  # 用来进行计数

    def add(self, rows, labels):
        self.buffer["data"].extend([rows])
        self.buffer["labels"].extend([labels])
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()

if __name__ =='__main__':
    index = 0
    trainWriter = HDF5DatasetWriter(dims=[4000,12,250],outputPath='./train-250-mean.hdf5')
    # ValWriter = HDF5DatasetWriter(dims=[100,12,250],outputPath='./val-250-mean.hdf5')
    count_val_001 = 0
    count_val_110 = 0
    count_val_100 = 0
    count_val_010 = 0
    
    count_train_001 = 0
    count_train_110 = 0
    count_train_100 = 0
    count_train_010 = 0
    
    df = pd.read_excel("./data/Train.xlsx","Sheet1")
    
    try:
        for i in range(0,len(df)):
            # print(i)
            name = df.iloc[i]['name']
            ste = df.iloc[i]['STE']
            std = df.iloc[i]['STD']
            Others = df.iloc[i]['Others']
            path = os.path.join("./data/Train",name)
            data = scio.loadmat(path)
            index += 1
            if index % 50 == 0:
                print("到第",i+1,"个mat")
            label = [int(ste), int(std), int(Others)]
            label_a = np.array(label)
            
            ecg = data['ecg']
            new_data = process_ecg(ecg, label_a)
            if type(new_data) == type(None):
                print("ERR TO PROCESS ",i)
                continue
            else:
                trainWriter.add(new_data,label_a)

    finally:
        trainWriter.close()
        print("fin")
    