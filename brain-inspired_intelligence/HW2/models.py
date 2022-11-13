"""

    filename : models.py
    author : Yichuan Ma
    Date : 2022/11/12
    Description : 实现了相同结构的SNN , CNN模型
    Reference :  Spikingjelly Document
    
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing


class SNN_Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=False,  dtype=torch.float32),
            nn.BatchNorm3d(8, dtype=torch.float32),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 10, bias=False, dtype=torch.float32),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
            )

    def forward(self, x):

        x = self.static_conv(x)
        # print(x.shape)
        # print('0 col shape', x[:,0,:,:,:].shape)
        
        out_spikes_counter = self.net(self.conv(x[:,0,:,:,:]))
        for t in range(1, self.T):
            out_spikes_counter += self.net(self.conv(x[:,t,:,:,:]))

        return out_spikes_counter / self.T
    
    
class CNN_Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 10, bias=False, dtype=torch.double),
            )

    def forward(self, x):

        out_spikes_counter = self.net(x[...,0,...])
        for t in range(1, self.T):
            out_spikes_counter += self.net(x[...,t,...])

        return out_spikes_counter / self.T