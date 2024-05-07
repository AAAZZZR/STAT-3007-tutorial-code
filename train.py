from zipfile import ZipFile
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
torch.manual_seed(1)
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.nn.functional import one_hot

from util import *
fashion_tr = torchvision.datasets.FashionMNIST('~/.torchvision', train=True, download=True)
fashion_ts = torchvision.datasets.FashionMNIST('~/.torchvision', train=False, download=True)

# input values are normalized to [0, 1]
x_tr, y_tr = fashion_tr.data.float()/255, fashion_tr.targets
x_ts, y_ts = fashion_ts.data.float()/255, fashion_ts.targets

x_tr_cnn, y_tr_cnn = x_tr.unsqueeze(1), one_hot(y_tr)
x_ts_cnn, y_ts_cnn = x_ts.unsqueeze(1), one_hot(y_ts)
classes = fashion_tr.classes

def train(net,x, y, lossfunc, lr=0.1, momentum=0, batch_size=600, nepochs=20):
    net.train()
    device = next(net.parameters()).device # check what device the net parameters are on
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    # training loop
    dataloader = DataLoader(DatasetWrapper(x, y), batch_size=batch_size, shuffle=True)
    
    loop = tqdm(range(nepochs), ncols=110)
    lossfunc = lossfunc
    train_accuracies = []
    for i in loop: # for each epoch
        t0 = time()
        
        # Task: fill in your training code below and compute epoch_loss (the average loss on all batches in a epoch)
        epoch_loss = 0
        n_batches = 0 
        for (x_batch, y_batch) in dataloader:
            
            pred = net(x_batch.to(device))
            loss = lossfunc(pred, y_batch.float())
            # loss = lossfunc(pred, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
       
        epoch_loss /= n_batches
        
        # evaluate network performance
        acc = test(net, x, y, batch_size=batch_size)
        train_accuracies.append(acc)
        # show training progress
        loop.set_postfix(loss="%5.5f" % (epoch_loss),
                         train_acc="%.2f%%" % (100*acc))
    plt.plot(range(nepochs), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.show()
# try running test(FashionMLP(), x_ts_mlp, y_ts_mlp, showerrors=True) to see what the code does
def test(net, x, y, batch_size=600, showerrors=False):
    net.eval()
    with torch.no_grad(): # disable automatic gradient computation for efficiency
        device = next(net.parameters()).device

        pred_cls = []
        # make predictions on mini-batches  
        dataloader = DataLoader(DatasetWrapper(x), batch_size=batch_size, shuffle=False)
        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            pred_cls.append(torch.max(net(x_batch), 1)[1].cpu())

        # compute accuracy
        pred_cls = torch.cat(pred_cls) # concat predictions on the mini-batches
        true_cls = torch.max(y, 1)[1].cpu()
        acc = (pred_cls == true_cls).sum().float() / len(y)

        # show errors if required
        if showerrors:
            idx_errors = (pred_cls != true_cls)

            x_errors = x[idx_errors][:10].cpu()
            y_pred = pred_cls[idx_errors][:10].cpu().numpy()
            y_true = true_cls[idx_errors][:10].cpu().numpy()

            plot_gallery(x_errors.squeeze(),
                         titles=[classes[y_true[i]] + '\n->' + classes[y_pred[i]] for i in range(10)],
                         xscale=1.5, yscale=1.5, nrow=2)

        return acc        
    
    
    