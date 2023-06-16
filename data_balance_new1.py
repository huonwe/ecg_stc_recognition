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
    # trainWriter = HDF5DatasetWriter(dims=[3200 + 1500 * 3,12,7500],outputPath='./train-balanced.hdf5')
    trainWriter = HDF5DatasetWriter(
        dims=[3200 - 4 + 1200 * 3, 12, 150], outputPath="./train-new1-balanced.hdf5"
    )
    valWriter = HDF5DatasetWriter(
        dims=[800 - 1 + 200 * 3, 12, 150], outputPath="./val-new1-balanced.hdf5"
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
    try:
        train_ecgs = h5py.File("./train-new1.hdf5", "r")

        for i in range(0, 3200 - 4):
            data = train_ecgs["data"][i]
            label = train_ecgs["labels"][i]
            trainWriter.add(data, label)
        print("origin data write over")

        stop = False
        i = 0
        index = 0
        print("expand data write start")
        while not stop:
            ecg = train_ecgs["data"][i]
            label = train_ecgs["labels"][i]
            i = (i + 1) % (3200 - 4)
            if index % 10 == 0 and index != 0:
                print("i: %s" % i)
                print("count100 = %d" % count100)
                print("count010 = %d" % count010)
                print("count110 = %d" % count110)
            new_data = ecg
            if label[2] == 1:
                continue
            else:
                index += 1
                count1 += 1
                if label[0] == 1 and label[1] == 0 and count100 < 1200:
                    trainWriter.add(new_data, label)
                    count100 += 1
                elif label[0] == 0 and label[1] == 1 and count010 < 1200:
                    trainWriter.add(new_data, label)
                    count010 += 1
                elif label[0] == 1 and label[1] == 1 and count110 < 1200:
                    trainWriter.add(new_data, label)
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

    try:
        count010 = 0
        count110 = 0
        count100 = 0
        val_ecgs = h5py.File("./val-new1.hdf5", "r")
        stop = False
        i = 0
        index = 0
        for i in range(0, 800 - 1):
            ecg = val_ecgs["data"][i]
            label = val_ecgs["labels"][i]
            valWriter.add(ecg, label)
        print("origin val data write over")
        print("expand data write start")
        while not stop:
            ecg = val_ecgs["data"][i]
            label = val_ecgs["labels"][i]

            i = (i + 1) % (800 - 1)

            if index % 10 == 0 and index != 0:
                print("i: %s" % i)
                print("count100 = %d" % count100)
                print("count010 = %d" % count010)
                print("count110 = %d" % count110)
            # new_data = process_ecgv2(ecg, label_a)
            new_data = ecg
            if type(new_data) == type(None):
                print("ERR TO PROCESS ", i)
                continue
            # print(label)
            # os._exit(0)
            if label[2] == 1:
                continue
            else:
                index += 1
                count1 += 1
                if label[0] == 1 and label[1] == 0 and count100 < 200:
                    valWriter.add(new_data, label)
                    count100 += 1
                elif label[0] == 0 and label[1] == 1 and count010 < 200:
                    valWriter.add(new_data, label)
                    count010 += 1
                elif label[0] == 1 and label[1] == 1 and count110 < 200:
                    valWriter.add(new_data, label)
                    count110 += 1
                print("count100 = %d" % count100)
                print("count010 = %d" % count010)
                print("count110 = %d" % count110)
            if count100 >= 200 and count010 >= 200 and count110 >= 200:
                break
    finally:
        valWriter.close()
        print("index: %s" % index)
        print("count100 = %d" % count100)
        print("count010 = %d" % count010)
        print("count110 = %d" % count110)
