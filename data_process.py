#encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio
import pandas as pd

from process_utils import process_ecgv2, process_ecg


class HDF5DatasetWriter:
    def __init__(self, dims=None, outputPath=None, bufSize=100):
        # 构建两种数据，一种用来存储图像特征一种用来存储标签
        # if dims is None:
        #     dims = [57851, 64, 1000]
        assert dims != None and outputPath != None
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("data", dims, dtype="float32")
        self.labels = self.db.create_dataset("labels", [dims[0],3], dtype="int")

        # 设置buffer大小，并初始化buffer
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0  # 用来进行计数

    def add(self, rows, labels):
        self.buffer["data"].extend([rows])
        self.buffer["labels"].extend([labels])
        # print("buffer: ",self.buffer['data'])
        # self.data[self.idx] = rows
        # self.labels[self.idx] = labels
        # 查看是否需要将缓冲区的数据添加到磁盘中
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将buffer中的内容写入磁盘之后重置buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        
        # print(self.buffer["labels"])
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        if len(self.buffer["data"]) > 0:  # 查看是否缓冲区中还有数据
            self.flush()

        self.db.close()

if __name__ =='__main__':
    index = 0
    trainWriter = HDF5DatasetWriter(dims=[3200,12,250],outputPath='./train-250.hdf5')
    ValWriter = HDF5DatasetWriter(dims=[800,12,250],outputPath='./val-250.hdf5')
    count001 = 0
    count110 = 0
    count100 = 0
    count010 = 0
    countn001 = 0
    count_train = 0
    count_val = 0
    df = pd.read_excel("./data/Train.xlsx","Sheet1")
    
    count0 = 0
    count1 = 0
    try:
        for i in range(0,len(df)):
            print(i)
            name = df.iloc[i]['name']
            ste = df.iloc[i]['STE']
            std = df.iloc[i]['STD']
            Others = df.iloc[i]['Others']
            path = os.path.join("./data/Train",name)
            data = scio.loadmat(path)
            index += 1
            if index % 50 == 0:
                print("index: %s" % index)
                print("count001 = %d" % count001)
                print("count100 = %d" % count100)
                print("count010 = %d" % count010)
                print("count110 = %d" % count110)
                print("count_train = %d" % count_train)
                print("count_val = %d" % count_val)
                print("count0: ",count0)
                print("count1: ",count1)
            label = [int(ste), int(std), int(Others)]
            label_a = np.array(label)
            # print("label: %s" % label_a)
            
            ecg = data['ecg']
            
            new_data = process_ecg(ecg, label_a)
            if type(new_data) == type(None):
                print("ERR TO PROCESS ",i)
                continue
            
            if label == [0, 0, 1]:
                count0 += 1
                if count001 < 692:
                    ValWriter.add(new_data,label_a)
                    count001 += 1
                    count_val += 1
                else:
                    trainWriter.add(new_data,label_a)
                    count_train += 1
            else:
                count1 += 1
                if label == [1, 0, 0] and count100 < 36:
                    ValWriter.add(new_data,label_a)
                    count100 += 1
                    count_val += 1
                elif label == [0, 1, 0] and count010 < 36:
                    ValWriter.add(new_data,label_a)
                    count010 += 1
                    count_val += 1
                elif label == [1, 1, 0] and count110 < 36:
                    ValWriter.add(new_data,label_a)
                    count110 += 1
                    count_val += 1
                else:
                    trainWriter.add(new_data,label_a)
                    count_train += 1
            # if i == 1:
            #    break
    finally:
        trainWriter.close()
        ValWriter.close()
        print("index: %s" % index)
        print("count001 = %d" % count001)
        print("count100 = %d" % count100)
        print("count010 = %d" % count010)
        print("count110 = %d" % count110)
        print("count0: ", count0)
        print("count1: ", count1)
        print("count_train = %d" % count_train)
        print("count_val = %d" % count_val)
