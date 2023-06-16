import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from utils_config import get_config, get_model, prepare_data

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
    print("maxEpoch: ", args.maxEpoch)
    train_loader, test_loader, num_channel = prepare_data(
        batch_size=cfg.batch_size, set=args.set
    )
    model_path = (
        "./models/" + cfg.network + "/" + str(args.maxEpoch) + "/model" + ".pth"
    )
    if not os.path.exists("./models/" + cfg.network):
        os.mkdir("./models/" + cfg.network)
    if not os.path.exists("./models/" + cfg.network + "/" + str(args.maxEpoch)):
        os.mkdir("./models/" + cfg.network + "/" + str(args.maxEpoch))

    # train_acc = []
    train_loss = []
    # val_loss = []
    test_loss = []
    # val_acc = []

    network = get_model(cfg, num_channel=num_channel).to(device)
    # network = torch.compile(network)
    network = network.train()
    torch.compile(network)
    if args.resume:
        network.load_state_dict(torch.load(model_path))
        print("resumed")
    else:
        try:
            network.initialize()
        except AttributeError:
            pass
    print(cfg.network)
    print(network)

    # weight=torch.from_numpy(np.array([20,20,1])).float()
    # weight = torch.from_numpy(np.array([0.5, 0.5, 1])).float()
    weight = torch.from_numpy(np.array([1, 1, 0.1])).float()
    criterion = nn.MultiLabelSoftMarginLoss(weight,reduction="mean")
    # criterion = nn.BCEWithLogitsLoss(weight,reduction="mean")

    # criterion = nn.MSELoss()
    # criterion = multilabel_categorical_crossentropy
    criterion.to(device)
    # ,lr=0.0001, momentum=0.9
    optimizer = optim.NAdam(network.parameters())
    # optimizer = optim.SGD(network.parameters(),lr=0.0005,momentum=0.9)

    epoch = 0
    num_print = 0
    stop = False
    while not stop:
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # print(labels[0])
            # time = np.arange(0, 40)
            # plt.plot(time, inputs[0][0][0][3:])
            # plt.show()
            # inputs = inputs[:,:,2,3:73].squeeze()
            # labels = labels[:,:2]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            # print(outputs[0])
            # print(outputs.shape)
            # print(labels.shape)
            optimizer.zero_grad()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            count1 = 0
            count2 = 0
            count3 = 0
            for l in labels:
                if l[0] == 1:
                    count1 += 1
                if l[1] == 1:
                    count2 += 1
                if l[2] == 1:
                    count3 += 1

            if i == 0:
                continue
            if i % check_time == 0:
                num_print += 1
                loss_avg = running_loss / check_time

                print("[%d, %5d] loss: %.6f" % (epoch, i, loss_avg))
                running_loss = 0
        

        if epoch % 2 == 0:
            print("saving model...")
            torch.save(network.state_dict(), model_path)
            
            train_loss.append(loss_avg)
            # vali(network,criterion,val_loader,device,cfg,args,train_loss,val_loss,val_acc,loss_avg,"val set")
            # result_val = valiv2(network,criterion,val_loader,device,cfg,"val")
            result_test = valiv2(model_path,criterion,test_loader,device,cfg,"test")
            
            # val_loss.append(result_val['loss'])
            test_loss.append(result_test['loss'])

            plt.cla()
            plt.plot(np.arange(0, len(train_loss)), train_loss, color="r", label="train loss")
            # plt.plot(np.arange(0, len(val_loss)), val_loss, color="b", label="val loss")
            plt.plot(np.arange(0, len(test_loss)), test_loss, color="g", label="test loss")
            # plt.plot(x, [a for a in val_acc], color="g", label="val acc")
            plt.legend(loc="upper left", bbox_to_anchor=(0, 1.0))
            plt.savefig(f"./results/{cfg.network}-loss.jpg")
            plt.cla()
            
            # print(f"val loss: {result_val['loss']}\t val score: {result_val['score']}")
            print(f"test loss: {result_test['loss']}\t test score: {result_test['score']}")

        epoch += 1
    print("fin")

@torch.no_grad()
def vali(network, criterion, val_loader, device, cfg, args, train_loss, val_loss, val_acc, loss_avg, mark=""):
    total_loss = []
    print("validating...",mark)
    predict_mat = []
    label_mat = []
    for _, v_data in enumerate(val_loader):
        inputs, labels = v_data
        # inputs = inputs[:,:,2,3:73].squeeze()
        inputs, labels = inputs.to(device), labels.to(device)
        # labels = labels[:,:2]
        outputs = network(inputs)
        # testy = network(testx)
        v_loss = criterion(outputs, labels.float())

        v_loss = v_loss.item()
        # outputs = F.sigmoid(outputs)
        total_loss.append(v_loss)

        predict_mat.extend(F.sigmoid(outputs))
        label_mat.extend(labels)

    # testx = Variable(torch.ones((32,12,150)).cuda(),requires_grad=True)
    # testy = network(testx)
    this_loss = np.mean(total_loss)
    val_loss.append(this_loss)
    train_loss.append(loss_avg)

    # for idx,pred in enumerate(predict_mat):
    #     predict_mat[idx] = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))

    # ac = pacc(predict_mat, label_mat)
    # print(f"acc: {ac}")
    # val_acc.append(ac)
    x = np.arange(0, len(val_loss))
    # plt.figure()
    # plt.xlim(0,5)
    # plt.ylim(0,1.5)
    plt.plot(x, train_loss, color="r", label="train loss")
    plt.plot(x, val_loss, color="b", label="val loss")
    # plt.plot(x, [a for a in val_acc], color="g", label="val acc")
    plt.legend(loc="upper left", bbox_to_anchor=(0, 1.0))
    plt.savefig(f"./results/{cfg.network}-loss.jpg")
    plt.cla()
    # predict_mat

    score1 = auprc(predict_mat, label_mat, 0, cfg)
    score2 = auprc(predict_mat, label_mat, 1, cfg)
    score3 = auprc(predict_mat, label_mat, 2, cfg)

    # pred = np.array([x.cpu().numpy() for x in predict_mat])
    # truth = np.array([x.cpu().numpy() for x in label_mat])
    # f1 = f1_score(y_true=pred, y_pred=truth,average="weighted")

    # acc = pacc(predict_mat, label_mat)

    # count11 = 0
    # count22 = 0
    # count33 = 0
    # for l in label_mat:
    #     if l[0] == 1:
    #         count11 += 1
    #     if l[1] == 1:
    #         count22 += 1
    #     if l[2] == 1:
    #         count33 += 1
    print(f"loss: {this_loss}")

    print(f"score1: {score1}")
    print(f"score2: {score2}")
    print(f"score3: {score3}")
    # print(f"count11:{count11} count22:{count22} count33:{count33}")
    for idx_label, label in enumerate(label_mat):
        if label[0] == 1 or label[1] == 1:
            print(f"label:\t {[label.item() for label in label_mat[idx_label]]}")
            print(f"pred:\t {[pred.item() for pred in predict_mat[idx_label]]}")
            break
    try:
        print(f"weight: {F.softmax(network.weight, dim=-1)}")
    except AttributeError:
        pass

    # if args.set in ["balanced-150", "250-balanced"]:
    #     try:
    #         at = (
    #             network.attentioned
    #             .cpu()
    #             .numpy()
    #         )
    #         ecg = inputs[0].cpu().numpy()
    #         # ecg[30:80] = 0
    #         plt.gray()
    #         plt.imshow(at[0])
    #         plt.savefig(f"./results/{cfg.network}-attention.jpg")
    #         plt.cla()
    #         plt.title(labels[0])
    #         for c in ecg:
    #             plt.plot(np.arange(0, 250), c)
    #         # (at[0][0]-np.min(at[0][0]))/(np.max(at[0][0]-np.min(at[0][0])))*100
    #         for a in at[0]:
    #             plt.plot(np.arange(0, 250), a*100, color="r")
    #         # print(at[0] * 100)
    #         plt.savefig(f"./results/{cfg.network}-attention2.jpg")
    #         plt.cla()
    #     except AttributeError:
    #         pass

# @torch.no_grad()
# def test(cfg, ctx, maxEpoch, test_loader=None):
#     assert test_loader is not None
#     device = torch.device("cuda:" + ctx if torch.cuda.is_available() else "cpu")
#     print("testing ", cfg.network)
#     maxEpoch = str(maxEpoch)
#     model_path = (
#         "./models/" + cfg.network + "/" + maxEpoch + "/model" + ".pth"
#     )
#     network = get_model(cfg)

#     # network.load_state_dict(torch.load(model_path))
#     network.load_state_dict(
#         {k.replace("_orig_mod.", ""): v for k, v in torch.load(model_path).items()}
#     )

#     network.eval().to(device)

#     # net_combine = cfg.network.split("_")

#     criterion = nn.BCEWithLogitsLoss()

#     with torch.no_grad():
#         test_loss = 0.0

#         predict_mat = []
#         label_mat = []
#         for _, t_data in enumerate(test_loader):
#             t_input, t_label = t_data
#             # t_input = t_input[:,:,2,3:73].squeeze()
#             t_input, t_label = t_input.to(device), t_label.to(device)

#             t_outputs = network(t_input)
#             t_loss = criterion(t_outputs, t_label.float()).item()
#             test_loss += t_loss

#             predict_mat.extend(F.sigmoid(t_outputs))
#             label_mat.extend(t_label)

#         score1 = auprc(predict_mat, label_mat, 0, cfg)
#         score2 = auprc(predict_mat, label_mat, 1, cfg)
#         score3 = auprc(predict_mat, label_mat, 2, cfg)

#         # pred = np.array([x.cpu().numpy() for x in predict_mat])
#         # truth = np.array([x.cpu().numpy() for x in label_mat])
#         # f1 = f1_score(y_true=pred, y_pred=truth,average="weighted")
#         print(f"test loss: {test_loss / len(test_loader)}")
#         print(f"score1: {score1}")
#         print(f"score2: {score2}")
#         print(f"score3: {score3}")
#         # print(f"f1: {f1}")
#         # print("test_acc: ", test_acc / len(test_loader))



@torch.no_grad()
def valiv2(model_path,criterion, val_loader, device, cfg, marker=""):
    network = get_model(cfg, num_channel=12).to(device)
    network.load_state_dict(torch.load(model_path))
    network = network.eval()
    total_loss = []
    # print("validating...",mark)
    predict_mat = []
    label_mat = []
    for _, v_data in enumerate(val_loader):
        inputs, labels = v_data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = network(inputs)
        v_loss = criterion(outputs, labels.float())
        v_loss = v_loss.item()
        total_loss.append(v_loss)

        predict_mat.extend(F.sigmoid(outputs))
        label_mat.extend(labels)

    this_loss = np.mean(total_loss)

    # for idx,pred in enumerate(predict_mat):
    #     predict_mat[idx] = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))

    score1 = auprc(predict_mat, label_mat, 0, cfg,marker)
    score2 = auprc(predict_mat, label_mat, 1, cfg,marker)
    score3 = auprc(predict_mat, label_mat, 2, cfg,marker)

    # count11 = 0
    # count22 = 0
    # count33 = 0
    # for l in label_mat:
    #     if l[0] == 1:
    #         count11 += 1
    #     if l[1] == 1:
    #         count22 += 1
    #     if l[2] == 1:
    #         count33 += 1
    # print(count11,count22,count33)
    # print(f"loss: {this_loss}")
    # print(f"score1: {score1}")
    # print(f"score2: {score2}")
    # print(f"score3: {score3}")
    for idx_label, label in enumerate(label_mat):
        if label[0] == 1 or label[1] == 1:
            print(f"label:\t {[label.item() for label in label_mat[idx_label]]}")
            print(f"pred:\t {[pred.item() for pred in predict_mat[idx_label]]}")
            break
    try:
        # print(f"weight: {F.softmax(network.weight, dim=-1)}")
        print(f"weight: {network.weight}")
    except AttributeError:
        pass
    result = {}
    result['loss'] = this_loss
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


def cross_entropy(logits, y):
    s = torch.exp(logits)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train option")
    parser.add_argument("network", type=str, help="Pytorch ECG Traing")
    parser.add_argument("--path", default=None, type=str, help="dataset path")
    parser.add_argument("--ctx", default="0", type=str, help="cuda")
    parser.add_argument("--maxEpoch", default=200, type=int, help="max epoch num")
    parser.add_argument("--resume", default=False, type=bool, help="continue")
    parser.add_argument(
        "--set", default="ori-STC-balanced", type=str, help="hdf5 suffix"
    )

    train(parser.parse_args())
