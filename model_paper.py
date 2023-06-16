# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class ATI_CNN(nn.Module):
    def __init__(self):
        super(ATI_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=12,out_channels=64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        
        self.conv3 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv4 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        
        self.conv5 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.conv6 = nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.conv7 = nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        
        self.conv8 = nn.Conv1d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        self.conv9 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.conv10 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        
        self.conv11 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.conv12 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.conv13 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=3,padding=1)

        self.maxpool = nn.MaxPool1d(3,3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(512)
        self.bn10 = nn.BatchNorm1d(512)
        
        self.bn11 = nn.BatchNorm1d(512)
        self.bn12 = nn.BatchNorm1d(512)
        self.bn13 = nn.BatchNorm1d(512)
        
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=32, num_layers=2, batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(32*15,3)
        self.attention = SelfAttention(2,32,32,0.1)
        # self.attention0 = SelfAttention(2,7500,7500,0.2)
    def forward(self, x):
        # x = self.attention0(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.maxpool(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.maxpool(x)
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.conv9(x)
        x = F.relu(self.bn9(x))
        x = self.conv10(x)
        x = F.relu(self.bn10(x))
        x = self.maxpool(x)
        x = self.conv11(x)
        x = F.relu(self.bn11(x))
        x = self.conv12(x)
        x = F.relu(self.bn12(x))
        x = self.conv13(x)
        x = F.relu(self.bn13(x))
        x = self.maxpool(x)
        
        v, _ = self.lstm(x.permute(0,2,1))
        attention_out = self.attention(v)
        # print(attention_out.shape)
        x = self.fc1(attention_out.reshape(-1,32*15))
        x = F.sigmoid(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.zeros_(param)




class ATI_CNN2(nn.Module):
    def __init__(self):
        super(ATI_CNN2, self).__init__()
        self.ati = ATI_CNN()
        self.attention0 = SelfAttention(2,7500,7500,0.2)
    def forward(self, x):
        x = self.attention0(x)
        x = self.ati(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.zeros_(param)

class BertSelfAttention(nn.Module):
    def __init__(self):
        super(BertSelfAttention, self).__init__()
        self.query = nn.Linear(30*32, 30*32) # 输入768， 输出768
        self.key = nn.Linear(30*32, 30*32) # 输入768， 输出768
    
    def forward(self,hidden_states): # hidden_states 维度是（L, 768）
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        out = torch.matmul(attention_probs, hidden_states)
        return out
    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # print("hidden states shape: ",hidden_states.shape)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

if __name__ == "__main__":
    x = torch.randn(32,32,30)
    omiga = nn.Parameter(torch.randn(32,32))