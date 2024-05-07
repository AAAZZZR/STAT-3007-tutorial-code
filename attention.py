import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        _,_,H,W = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weight = torch.matmul(q, k.transpose(1, 2)) #batch,H,W
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        attn_value = torch.matmul(attn_weight, v)
        return attn_value
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads,         
            self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, 
            self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, 
            self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attended_values = torch.matmul(attn_weights, v).transpose(1, 
        2).contiguous().view(batch_size, seq_len, embed_dim)

        x = self.fc(attended_values) + x

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,_,_ = x.size()
        avg_val = self.avgpool(x).view(b, c)
        max_val = self.maxpool(x).view(b, c)
        avg_out = self.fc(avg_val).view(b, c,1,1)
        max_out = self.fc(max_val).view(b, c,1,1)
        output=self.sigmoid(avg_out+max_out)
        return output
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding= 7 //2 ) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_val = torch.max(x, dim=1, keepdim=True)[0] #only value
        avg_val = torch.mean(x, dim=1, keepdim=True)
        output = torch.cat([max_val, avg_val], dim=1)
        output=self.conv(output)
        output=self.sigmoid(output)
        return output