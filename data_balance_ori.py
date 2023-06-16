#encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio
import pandas as pd

from process_utils import process_ecgv2, process_ecg
import random

import matplotlib.pyplot as plt

class HDF5DatasetWriter:
    def __init__(self, dims=None, outputPath=None, bufSize=100, startIndex = 0):
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
        self.idx = startIndex  # 用来进行计数

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
    trainWriter = HDF5DatasetWriter(dims=[3200 + 1200 * 3,12,150],outputPath='./train-balanced-sub.hdf5')
    valWriter = HDF5DatasetWriter(dims=[800,12,150],outputPath='./val-balanced-sub.hdf5')
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
            # print(i)
            name = df.iloc[i]['name']
            ste = df.iloc[i]['STE']
            std = df.iloc[i]['STD']
            Others = df.iloc[i]['Others']
            path = os.path.join("./data/Train",name)
            data = scio.loadmat(path)
            index += 1
            if index % 10 == 0:
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
            # new_data = data['ecg']
            if type(new_data) == type(None):
                print("ERR TO PROCESS ",i)
                continue
            
            if label == [0, 0, 1]:
                count0 += 1
                if count001 < 692:
                    valWriter.add(new_data,label_a)
                    count001 += 1
                    count_val += 1
                else:
                    trainWriter.add(new_data,label_a)
                    count_train += 1
            else:
                count1 += 1
                if label == [1, 0, 0] and count100 < 36:
                    valWriter.add(new_data,label_a)
                    count100 += 1
                    count_val += 1
                elif label == [0, 1, 0] and count010 < 36:
                    valWriter.add(new_data,label_a)
                    count010 += 1
                    count_val += 1
                elif label == [1, 1, 0] and count110 < 36:
                    valWriter.add(new_data,label_a)
                    count110 += 1
                    count_val += 1
                else:
                    trainWriter.add(new_data,label_a)
                    count_train += 1
            # if i == 1:
            #    break
    finally:
        trainWriter.flush()
        valWriter.flush()
        valWriter.close()
        print("index: %s" % index)
        print("count train = %d", count_train)
        print("count val = %d", count_val)
        print("count100 = %d" % count100)
        print("count010 = %d" % count010)
        print("count110 = %d" % count110)
        print("origin data write over")

    try:
        print("start data transform")
        count001 = 0
        count110 = 0
        count100 = 0
        count010 = 0
        train_ecgs = h5py.File("train-balanced-150.hdf5", "r")
        stop = False
        i = 0
        index = 0
            
        while(not stop):
            ecg = train_ecgs['data'][i]
            label = train_ecgs['labels'][i]
            
            i = (i + 1) % 3200 - 5
            
    
            if index % 10 == 0 and index != 0:
                print("i: %s" % i)
                print("count100 = %d" % count100)
                print("count010 = %d" % count010)
                print("count110 = %d" % count110)
            # new_data = process_ecgv2(ecg, label_a)
            new_data = ecg
            if type(new_data) == type(None):
                print("ERR TO PROCESS ",i)
                continue
            # print(label)
            # os._exit(0)
            if label[2] == 1:
                continue
            else:
                index += 1
                # shift = random.randint(-10,10)
                # data_shift = np.zeros((12,7500))
                # data_shift[:,shift:] = new_data[:,0:(7500-shift)]
                # data_shift[:,0:shift] = new_data[:,(7500-shift):]
                # new_data = data_shift
                #
                # print(shift)
                # time = np.arange(0, 7500)
                # plt.plot(time,data_shift[0], color="b")
                # plt.plot(time,new_data[0], color="r")
                # plt.show()
                #
                new_data = ecg
                count1 += 1
                if label[0] == 1 and label[1] == 0 and count100 < 1200:
                    trainWriter.add(new_data,label)
                    count100 += 1
                elif label[0] == 0 and label[1] == 1 and count010 < 1200:
                    trainWriter.add(new_data,label)
                    count010 += 1
                elif label[0] == 1 and label[1] == 1 and count110 < 1200:
                    trainWriter.add(new_data,label)
                    count110 += 1
                print("count100 = %d" % count100)
                print("count010 = %d" % count010)
                print("count110 = %d" % count110)
            if count100 >= 1200 and count010 >= 1200 and count110 >= 1200:
                break
    finally:
        trainWriter.close()
        print("index: %s" % index)
        print("count100 = %d" % count100)
        print("count010 = %d" % count010)
        print("count110 = %d" % count110)