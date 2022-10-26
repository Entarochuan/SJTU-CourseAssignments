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
import pickle
import numpy as np
import Resnet_model
import os, sys, logging
from jittor import nn
import numpy as np
import jittor.transform as trans


def train(model, train_loader, optimizer, epochs, writer=None):

    model.train()

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)
            loss = nn.cross_entropy_loss(outputs, targets)
            optimizer.step(loss)
            if batch_idx % 100 == 0:
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx, int(len(train_loader)/64),
                        100. * batch_idx / (len(train_loader)/64.), loss.data[0]))


def train_edit(model, train_loader, optimizer, epochs, writer=None):

    model.train()

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # New dataset
            if batch_idx % 10 != 0:
                mask = (targets >= 5)
            else:
                mask = (targets >= 0)

            targets = targets[mask]
            inputs = inputs[mask]

            outputs = model(inputs)
            loss = nn.cross_entropy_loss(outputs, targets)
            optimizer.step(loss)
            if batch_idx % 100 == 0:
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx, int(len(train_loader)/64),
                        100. * batch_idx / (len(train_loader)/64.), loss.data[0]))


def test(model, val_loader):

    model.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        outputs = model(inputs)
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        total_loss += nn.cross_entropy_loss(outputs, targets).data[0]

        if batch_idx % 10 == 0:
            print('test acc = {}, test loss = {}'.format(total_acc / total_num, total_loss / batch_idx))


if __name__ == "__main__":

    batch_size = 64
    learning_rate = 0.01
    momentum = 0.95
    weight_decay = 1e-4
    epochs = 20

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

    model = Resnet_model.Resnet()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)

    # if not os.path.exists('./model/Resnet_params.pkl'):
    #     train(model, train_loader, optimizer, epochs)
    #     model.save('./model/Resnet_params.pkl')

    # train(model, train_loader, optimizer, epochs)
    # model.save('./model/Resnet_params.pkl')
    # train_edit(model, train_loader, optimizer, epochs)
    # model.save('./model/Resnet_edit_params.pkl')

    if os.path.exists('./model/Resnet_params.pkl'):
        print('Origin data:')
        model.load('./model/Resnet_params.pkl')
        test(model, val_loader)

    if os.path.exists('./model/Resnet_edit_params.pkl'):
        print('Edited data:')
        model.load('./model/Resnet_edit_params.pkl')
        test(model, val_loader)
