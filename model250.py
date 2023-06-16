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
        self.resnet_g = resnet18(in_channels=250)
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=6, num_layers=3,
            dropout=0.3
        )
        self.fc1 = nn.Linear(6*250,3)
        # self.fc2 = nn.Linear(250, 3)
        # self.at0 = nn.MultiheadAttention(12,4,dropout=0.5)
        self.at = nn.MultiheadAttention(6,2,dropout=0.5)
        
        self.weight = nn.Parameter(torch.ones(3))
    def forward(self, x):
        # x1,_ = self.at0(x.permute(2,0,1),x.permute(2,0,1),x.permute(2,0,1))
        
        x1 = x.permute(2,0,1)
        x1,_ = self.lstm(x1)

        x1,_ = self.at(x1,x1,x1)
        self.attentioned = x1.permute(1,2,0)
        x1 = torch.flatten(x1.permute(1,2,0),start_dim=1)
        x1 = self.fc1(x1)
        
        x2 = self.resnet_g(x.permute(0,2,1))
        x3 = self.resnet(x)
        
        mix = self.weight[0]*x1 + self.weight[1]*x2 + self.weight[2]*x3
        # batch_size, 3
        # print(mix_min.shape)
        # min_v, _ = torch.max(mix[:,:2],dim=-1)
        # mix[:,2] -= min_v
        
        # mix[:,:2] = mix[:,:2] - mix[:,2].reshape(-1,1).repeat(1,2)
        
        
        return mix
        