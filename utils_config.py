import os
import sys
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from future_model import *
from combine_model import *
from model_special import *
from model_paper import *
from transformer import *
from nextModel import *
from randomModel import *
from type3Model import *
from model7500 import *
from model250 import *
from resnet import resnet50, resnet101

from datasets import ecgDataset, ecgTestset
from torchvision import datasets, transforms

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

lstm_seri = ["lstm", "blstm"]
gru_seri = ["gru", "bgru"]

def get_config(network):
    cfg = edict()
    # COMMON
    cfg.lr = 0.001
    cfg.dropout = 0.3
    cfg.batch_size = 64
    cfg.network = network
    cfg.num_lstm = 2

    return cfg


def get_model(cfg, **kwargs):
    if cfg.network == "split":
        return SplitNet(batch_size=cfg.batch_size)
    elif cfg.network == "splitv2":
        return SplitNetV2(batch_size=cfg.batch_size)
    elif cfg.network == "splitv3":
        return SplitNetV3(batch_size=cfg.batch_size)
    elif cfg.network == "ati":
        return ATI_CNN()
    elif cfg.network == "ati2":
        return ATI_CNN2()
    elif cfg.network == "tf":
        return TransformerTS()
    elif cfg.network == "type2":
        return Type2()
    elif cfg.network == "type3":
        return Type3()
    elif cfg.network == "type4":
        return Type4()
    elif cfg.network == "type5":
        return Type5()
    elif cfg.network == "type6":
        return Type6()
    elif cfg.network == "type7":
        return Type7()
    elif cfg.network == "STC":
        return STC()
    elif cfg.network == "splite":
        return Splite()
    elif cfg.network == "resnet":
        model = resnet50()
        return model
    elif cfg.network == "resnet101":
        model = resnet101()
        return model
    elif cfg.network == "random":
        return RANDOM()
    elif cfg.network == "model7500":
        return model7500()
    elif cfg.network == "model250":
        return model250()
    else:
        raise ValueError()


def prepare_data(batch_size, set:str):
    # path_train = f"train-balanced.hdf5"
    # path_val = f"val-balanced.hdf5"
    path_train = f"train-{set}.hdf5"
    # path_val = f"val-{set}.hdf5"

    print("preparing train data: " + path_train)
    # print("preparing val data: " + path_val)
    if "7500" in set:
        dataset_train = ecgDataset(path_train,shift=True, isTrain=True)
        dataset_test = ecgTestset("test.hdf5")
        print("test.hdf5")
    else:
        dataset_train = ecgDataset(path_train, isTrain=True)
        dataset_test = ecgTestset("test-250-mean.hdf5")
        print("test-250-mean.hdf5")
    
    
    # length = len(dataset_val)
    # val_size = int(0.5 * length)

    # val_set, test_set = torch.utils.data.random_split(
    #     dataset_val, [val_size, length - val_size]
    # )
    num_channel = 12

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    # val_loader = DataLoader(
    #     dataset=dataset_val, batch_size=batch_size, shuffle=True, drop_last=True
    # )
    test_loader = DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader, num_channel



def prepare_test(batch_size):
    dataset_test = ecgTestset("test-250-mean.hdf5")
    print("test-250-mean.hdf5")

    test_loader = DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return test_loader
