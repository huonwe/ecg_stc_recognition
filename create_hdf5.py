#encoding=utf8
import h5py
import os
import numpy as np
import scipy.io as scio


class HDF5DatasetWriter:
    def __init__(self, dims=None, outputPath=None, bufSize=100):
        # 构建两种数据，一种用来存储图像特征一种用来存储标签
        # if dims is None:
        #     dims = [57851, 64, 1000]
        assert dims != None and outputPath != None
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("data", dims)
        self.labels = self.db.create_dataset("labels", [dims[0],3])

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
    # trainWriter = HDF5DatasetWriter(dims=[3200,12,15*500],outputPath='./train.hdf5')
    # ValWriter = HDF5DatasetWriter(dims=[800,12,15*500],outputPath='./val.hdf5')
    TestWriter = HDF5DatasetWriter(dims=[1000,12,15*500],outputPath='./test.hdf5')
    count001 = 0
    count110 = 0
    count100 = 0
    count010 = 0
    
    count_train = 0
    try:
        for root,dirs,files in os.walk('./data/Test'):
            for mat in files:
                index += 1
                if index % 50 == 0:
                    print("index: %s" % index)
                path = os.path.join(root, mat)
                data = scio.loadmat(path)
                # print(data['ecg'])
                # os._exit(0)
                # data = h5py.File(path)
                # print(data['label'][0])
                label = [data['label'][0][0][0][0], data['label'][0][1][0][0], data['label'][0][2][0][0]]
                label_a = np.array(label)
                # print("label: %s" % label_a)
                TestWriter.add(data['ecg'],label_a)
    finally:
        TestWriter.close()
