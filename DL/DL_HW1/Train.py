#coding=utf8
"""

    filename: HW1_1.py
    data: 10/13
    description: 基于jittor框架实现并训练Resnet
    Reference: jittor官方文档

    目标:
    1. 下载并学习使用CIFAR-10数据集
    2. 搭建深度学习训练框架 Jittor，Mindspore 或 Paddle （三选一，推荐Jittor），并设计深度学习模型
    3. 在给定的要求下改进模型

"""


import jittor as jt
from jittor import nn
from jittor import models
import jittor_Resnet
import pickle
import numpy as np
import Resnet_model
import os, sys, logging
from jittor import nn
import numpy as np
import jittor.transform as trans


def train(model, train_loader, optimizer, epoch, writer=None):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print('tmp_batch=', batch_idx)
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step(loss)
        if batch_idx % 10 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))
            if writer:
                writer.add_scalar('Train/Loss', loss.data[0], global_step=batch_idx + epoch * len(train_loader))


if __name__ == "__main__":

    batch_size = 64
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 10

    train_transform = trans.Compose([
        trans.Resize(32),
        trans.RandomHorizontalFlip(),
        trans.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = trans.Compose([
        trans.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = jt.dataset.CIFAR10(root='./data/cifar/', train=True, transform=train_transform, target_transform=None, download=True).set_attrs(batch_size=batch_size, shuffle=True)
    val_loader = jt.dataset.CIFAR10(root='./data/cifar/', train=False, transform=test_transform, target_transform=None, download=True).set_attrs(batch_size=batch_size, shuffle=True)

    # model = jittor_Resnet.ResNet18()
    model = Resnet_model.Resnet()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)

    for epoch in range(0, epochs):
        print('epoch=', epoch)
        train(model, train_loader, optimizer, epoch)

    model.save('./model/Resnet_params.pkl')
