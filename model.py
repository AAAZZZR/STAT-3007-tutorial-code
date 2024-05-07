import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from attention import *

class basicCNN(nn.Module):
    def __init__(self):
        super(basicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1) #28>24
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)#24>20
        self.fc = nn.Sequential(
            nn.Linear(20*20*16,120,bias = True),
            nn.ReLU(),
            nn.Linear(120,84,bias = True),
            nn.ReLU(),
            nn.Linear(84,10,bias = True)
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x
    def predict(self, x):

        o = self.forward(x) # the same as self.forward(x)
        l = torch.max(o, 1)[1] 
        return l
    def score(self, x, y):

        pred_cls = self.predict(x)
        true_cls = torch.max(y, 1)[1]
        return (pred_cls == true_cls).sum().float().item() / len(y)


class attentionCNN(basicCNN):
    def __init__(self,attention_type):
        super(attentionCNN, self).__init__()
        self.attention_type = attention_type
        if attention_type == 'se':
            self.attention = SELayer(16)
        elif attention_type == 'channel':
            self.attention = ChannelAttention(16)
        elif attention_type == "spatial":
            self.attention = SpatialAttention()
    
    def forward(self, x, attention_type = None):
        if attention_type is None:
            attention_type = self.attention_type 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Apply attention mechanism
        if hasattr(self, 'attention'):  # Check if attention mechanism is defined
            attention_map = self.attention(x)
            x = x * attention_map  # Apply attention

        # Flatten the output before passing it to fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x