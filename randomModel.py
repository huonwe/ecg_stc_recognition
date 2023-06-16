# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class RANDOM(nn.Module):
    def __init__(self):
        super(RANDOM, self).__init__()
        self.p = nn.Parameter(torch.randn(32,3))
    def forward(self, x):
        return F.sigmoid(self.p)
