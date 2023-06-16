import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from utils_config import get_config, get_model, prepare_test

# import torchvision.ops.focal_loss as FocalLoss
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
from loss import *

check_time = 10


def train(args):
    device = torch.device("cuda:" + args.ctx if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.network)
    print(device)
    # print(cfg)
    test_loader = prepare_test(
        batch_size=cfg.batch_size
    )
    model_path = (
        "./models/model250/200/model_1.pth"
    )
    
    score = test(model_path,test_loader,device,cfg)
    print(score)
    

@torch.no_grad()
def test(model_path, test_loader, device, cfg, marker="test"):
    network = get_model(cfg, num_channel=12).to(device)
    network.load_state_dict(torch.load(model_path))
    network = network.eval()
    predict_mat = []
    label_mat = []
    for _, v_data in enumerate(test_loader):
        inputs, labels = v_data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = network(inputs)

        predict_mat.extend(F.sigmoid(outputs))
        label_mat.extend(labels)


    score1 = auprc(predict_mat, label_mat, 0, cfg,marker)
    score2 = auprc(predict_mat, label_mat, 1, cfg,marker)
    score3 = auprc(predict_mat, label_mat, 2, cfg,marker)

    for idx_label, label in enumerate(label_mat):
        if label[0] == 1 or label[1] == 1:
            print(f"label:\t {[label.item() for label in label_mat[idx_label]]}")
            print(f"pred:\t {[pred.item() for pred in predict_mat[idx_label]]}")
            break
    try:
        print(f"weight: {network.weight}")
    except AttributeError:
        pass
    result = {}
    result['score'] = [score1,score2,score3]
    return result


def pred2Lable(pred, threshold=0.5) -> list:
    # pred = F.sigmoid(pred)
    pred_truth = [0, 0, 0]
    if pred[0] > threshold:
        pred_truth[0] = 1
    if pred[1] > threshold:
        pred_truth[1] = 1
    if pred[2] > threshold:
        pred_truth[2] = 1
    if pred_truth[0] or pred_truth[1]:
        pred_truth[2] = 0
    if pred_truth[2] == 1:
        pred_truth[0] = 0
        pred_truth[1] = 0
    return pred_truth


def pacc(ouputs, labels) -> float:
    correct = 0
    length = len(labels)
    for idx, pred in enumerate(ouputs):
        pred_truth = pred2Lable(pred)
        if pred_truth == [x.item() for x in labels[idx]]:
            correct += 1

    return correct / length


def auprc(predict_mat, label_mat, dim, cfg, marker=""):
    truth = np.array([x[dim].item() for x in label_mat])
    try:
        pred = np.array([x[dim].item() for x in predict_mat])
    except Exception:
        pred = np.array([x[dim] for x in predict_mat])

    precision, recall, threholds = precision_recall_curve(truth, pred, pos_label=1)
    score = average_precision_score(truth, pred, pos_label=1)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.scatter(recall, precision, c="r")

    plt.savefig(f"./auprc/{cfg.network}-{marker}-{dim}.jpg")
    plt.cla()
    return score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train option")
    parser.add_argument("network", type=str, help="Pytorch ECG Traing")
    parser.add_argument("--ctx", default="0", type=str, help="cuda")
    parser.add_argument(
        "--set", default="250-mean", type=str, help="hdf5 suffix"
    )

    train(parser.parse_args())
