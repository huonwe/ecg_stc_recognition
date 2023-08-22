# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50, resnet18, resnet34
from transformer import TransformerTS

class model250(nn.Module):
    def __init__(self):
        super(model250, self).__init__()
        self.resnet = resnet18()
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=6, num_layers=3,
            dropout=0.3
        )
        self.fc1 = nn.Linear(6*250,3)
        self.at = nn.MultiheadAttention(6,2,dropout=0.5)
        
        self.weight = nn.Parameter(torch.Tensor([0.5]))
        self.dp = nn.Dropout(p=0.8)
    def forward(self, x):
        x = self.dp(x)
        
        x1 = x.permute(2,0,1)
        x1,_ = self.lstm(x1)
        x1,_ = self.at(x1,x1,x1)
        self.attentioned = x1.permute(1,2,0)
        x1 = torch.flatten(x1.permute(1,2,0),start_dim=1)
        x1 = self.fc1(x1)
        x3 = self.resnet(x)
        
        mix = self.weight[0]*x1 + (1-self.weight[0])*x3
        # batch_size, 3
        # print(mix_min.shape)
        # min_v, _ = torch.max(mix[:,:2],dim=-1)
        # mix[:,2] -= min_v
        # mix[:,:2] = mix[:,:2] - mix[:,2].reshape(-1,1).repeat(1,2)
        return mix

class model250v2(nn.Module):
    def __init__(self):
        super(model250v2, self).__init__()
        self.resnet = resnet18(num_classes=2)
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=6, num_layers=3,
            dropout=0.3
        )
        self.fc1 = nn.Linear(6*250,2)
        self.at = nn.MultiheadAttention(6,2,dropout=0.5)
        
        self.weight = nn.Parameter(torch.Tensor([0.5]))
        self.dp = nn.Dropout(p=0.8)
        
        # self.tmp_result = torch.zeros((64,3))
    def forward(self, x):
        x = self.dp(x)
        
        x1 = x.permute(2,0,1)
        x1,_ = self.lstm(x1)
        x1,_ = self.at(x1,x1,x1)
        self.attentioned = x1.permute(1,2,0)
        x1 = torch.flatten(x1.permute(1,2,0),start_dim=1)
        x1 = self.fc1(x1)
        x3 = self.resnet(x)
        
        mix = self.weight[0]*x1 + (1-self.weight[0])*x3
        # batch_size, 3
        # print(mix_min.shape)
        # min_v, _ = torch.max(mix[:,:2],dim=-1)
        # mix[:,2] -= min_v
        # mix[:,:2] = mix[:,:2] - mix[:,2].reshape(-1,1).repeat(1,2)
        
        # self.tmp_result[:,:-1] = mix
        # self.tmp_result[:,-1] = torch.multiply(1-mix[:,0],1-)
        return mix
        