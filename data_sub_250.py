# encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio
import pandas as pd

from process_utils import process_ecgv2
import random

import matplotlib.pyplot as plt


class HDF5DatasetWriter:
    def __init__(self, dims=None, outputPath=None, bufSize=100, startIndex=0):
        # 构建两种数据，一种用来存储图像特征一种用来存储标签
        # if dims is None:
        #     dims = [57851, 64, 1000]
        assert dims != None and outputPath != None
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("data", dims, dtype="float32")
        self.labels = self.db.create_dataset("labels", [dims[0], 3], dtype="int")

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
        self.data[self.idx : i] = self.buffer["data"]

        # print(self.buffer["labels"])
        self.labels[self.idx : i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        if len(self.buffer["data"]) > 0:  # 查看是否缓冲区中还有数据
            self.flush()

        self.db.close()


if __name__ == "__main__":
    index = 0
    trainWriter = HDF5DatasetWriter(
        dims=[3173 + 1200*3, 12, 250], outputPath="./train-250-balanced-sub.hdf5"
    )
    valWriter = HDF5DatasetWriter(
        dims=[800, 12, 250], outputPath="./val-250-balanced-sub.hdf5"
    )

    count001 = 0
    count110 = 0
    count100 = 0
    count010 = 0
    countn001 = 0
    count_train = 0
    count_val = 0
    # df = pd.read_excel("./data/Train.xlsx","Sheet1")

    count0 = 0
    count1 = 0
    
    sum = np.zeros((12,250))
    count = np.zeros((12))
    
    ecgs = h5py.File("./train-250.hdf5", "r")

    for idx, ecg_label in enumerate(ecgs["labels"]):
        if ecg_label[2] == 1:
            for c_idx,c in enumerate(ecgs["data"][idx]):
                if np.max(c) == 0:
                    continue
                sum[c_idx] += c
                count[c_idx] += 1
    avg = np.zeros((12,250))
    for c_idx in range(0,12):
        avg[c_idx] = sum[c_idx] / count[c_idx]
    #     plt.plot(np.arange(0,250),avg[c_idx])
    # plt.show()
    train_datas = h5py.File("./train-250-balanced.hdf5", "r")
    train_ecgs = train_datas['data']
    train_labels = train_datas['labels']
    