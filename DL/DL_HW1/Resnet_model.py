#coding=utf8
"""
    filename: Resnet_model.py
    Author: Ma  Yichuan
    data: 10/13
    description: 基于jittor框架实现并训练Resnet
    Reference: Dive Into Deep Learning, jittor document
"""

import jittor as jt
from jittor import nn
import pickle
import numpy as np


class Residual(nn.Module):

    def __init__(self, input_channels, num_channels,  # 当前后通道不一致时需要将输出先通过1x1卷积层变形
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        # 归一化模块
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def execute(self, X):
        Y = jt.nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
        Y += X

        return jt.nn.relu(Y)


def Resnet_Block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []

    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))

        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 定义Resnet架构
class Resnet(nn.Module):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.b_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
                                 nn.BatchNorm2d(64), nn.Relu(),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b_2 = nn.Sequential(*Resnet_Block(64, 64, 2, first_block=True))
        self.b_3 = nn.Sequential(*Resnet_Block(64, 128, 2))
        self.b_4 = nn.Sequential(*Resnet_Block(128, 256, 2))
        self.b_5 = nn.Sequential(*Resnet_Block(256, 512, 2))
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        self.Softmax = nn.Softmax()

    def Show_net(self):
        X = jt.rand((1, 1, 224, 224))
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    def execute(self, x):
        x = self.b_1(x)
        x = self.b_2(x)
        x = self.b_3(x)
        x = self.b_4(x)
        x = self.b_5(x)
        x = self.Avgpool(x)
        x = jt.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        x = self.Softmax(x)
        return x


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """评价net在data上的表现"""
    if isinstance(net, nn.Module):  #如果是nn函数
        net.eval()  # 将模型设置为评估模式

    num_pred, num_sum = 0, 0
    with jt.no_grad():
        for X, y in data_iter:
            num_pred = num_pred + accuracy(net(X), y)
            num_sum = num_sum + y.numel()
    return num_pred / num_sum  #返回正确率
