# coding=utf8
"""

    filename: HW1_1.py
    data: 10/13
    description: jittor Resnet
    Reference: jittor document

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
import random

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

random.seed(999)


def train(model, train_loader, optimizer, epochs, writer=None):
    model.train()

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)

            # print(outputs.shape)
            # print(targets.shape)
            
            loss = nn.cross_entropy_loss(outputs, targets)
            optimizer.step(loss)
            # if batch_idx == 1:
            #     break
            
            if batch_idx % 100 == 0:
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx, int(len(train_loader) / 64),
                    100. * batch_idx / (len(train_loader) / 64.), loss.data[0]))


def train_edit_B(model, train_loader, optimizer, epochs, writer=None):
    model.train()

    for epoch in range(epochs):
        Flag_of_label = False
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            #
            i = random.randint(1, 10)

            if i % 10 != 0:
                flag = False
                mask = (targets >= 5)
            else:
                flag = True
                mask = (targets >= 0)

            targets = targets[mask]
            inputs = inputs[mask]

            if not Flag_of_label:
                labels = targets
                images = inputs
                Flag_of_Label = True
            else:
                labels = jt.concat((labels, targets))
                images = jt.concat((inputs, images))

            if not flag:
                # 不充分data训练
                outputs = model(inputs)
                weight = jt.array([0.001, 0.001, 0.001, 0.001, 0.001, 1., 1., 1., 1., 1.])
                loss = nn.cross_entropy_loss(outputs, targets, weight) * 0.001

                optimizer.step(loss)

                if batch_idx % 100 == 0:
                    print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx, int(len(train_loader) / 64),
                        100. * batch_idx / (len(train_loader) / 64.), loss.data[0] * 1000 ))

            else:
                # Resample
                nsamples, nx, ny, nz = images.shape
                reshaped_images = images.reshape((nsamples, nx * ny * nz))
                # undersampling
                Resample_type = RandomOverSampler(random_state=0)
                reshaped_images, labels = Resample_type.fit_resample(reshaped_images, labels)

                # reshaped again
                nsamples, nsum = reshaped_images.shape
                images = reshaped_images.reshape((nsamples, nx, ny, nz))

                images = jt.array(images)
                labels = jt.array(labels)
                # print(images.shape)
                # print(labels.shape)

                outputs = model(inputs)
                # loss re-weighted
                weight = jt.array([1.1, 1.1, 1.1, 1.1, 1.1, 1., 1., 1., 1., 1.])
                loss = nn.cross_entropy_loss(outputs, targets, weight)

                optimizer.step(loss)

                if batch_idx % 100 == 0:
                    print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx, int(len(train_loader) / 64),
                        100. * batch_idx / (len(train_loader) / 64.), loss.data[0]))


def train_edit_A(model, train_loader, optimizer, epochs, writer=None):
    model.train()

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # New dataset
            i = random.randint(1, 10)
            if i % 10 != 0:
                flag = False
                mask = (targets >= 5)
            else:
                flag = True
                mask = (targets >= 0)

            targets = targets[mask]
            inputs = inputs[mask]

            outputs = model(inputs)
            # loss re-weighted
            if flag:
                loss = nn.cross_entropy_loss(outputs, targets)
            else:
                loss = nn.cross_entropy_loss(outputs, targets) * 0.001

            optimizer.step(loss)

            if batch_idx % 100 == 0:
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx, int(len(train_loader) / 64),
                    100. * batch_idx / (len(train_loader) / 64.), loss.data[0] * 1000 ))


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
        acc = np.sum(targets.data == pred)
        total_acc += acc
        total_num += batch_size
        total_loss += nn.cross_entropy_loss(outputs, targets).data[0]

        if batch_idx % 10 == 0:
            print('test acc = {}, test loss = {}'.format(total_acc / total_num, total_loss / batch_idx))


if __name__ == "__main__":

    jt.flags.use_cuda = 1
    batch_size = 64
    learning_rate = 0.005
    momentum = 0.91
    weight_decay = 1e-4
    epochs = 30

    train_transform = trans.Compose([
        trans.Resize(32),
        trans.RandomHorizontalFlip(),
        trans.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = trans.Compose([
        trans.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = jt.dataset.CIFAR10(root='./data/cifar/', train=True, transform=train_transform,
                                      target_transform=None, download=True).set_attrs(batch_size=batch_size,
                                                                                      shuffle=True)
    val_loader = jt.dataset.CIFAR10(root='./data/cifar/', train=False, transform=test_transform, target_transform=None,
                                    download=True).set_attrs(batch_size=batch_size, shuffle=True)

    model = Resnet_model.Resnet()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)

    # if not os.path.exists('./model/Resnet_params.pkl'):
    #     train(model, train_loader, optimizer, epochs)
    #     model.save('./model/Resnet_params.pkl')
        
    # if not os.path.exists('./model/Resnet_edit_A_params.pkl'):
    #     train_edit_A(model, train_loader, optimizer, epochs)
    #     model.save('./model/Resnet_edit_A_params.pkl')
    
    # if not os.path.exists('./model/Resnet_edit_B_params.pkl'):
    #     train_edit_B(model, train_loader, optimizer, epochs)
    #     model.save('./model/Resnet_edit_B_params.pkl')
    


    # if os.path.exists('./model/Resnet_params.pkl'):
    #     print('Training with the original data:')
    #     model.load('./model/Resnet_params.pkl')
    #     test(model, val_loader)

    # if os.path.exists('./model/Resnet_edit_A_params.pkl'):
    #     print('Edited data, train_method_1:')
    #     model.load('./model/Resnet_edit_A_params.pkl')
    #     test(model, val_loader)
    
    if os.path.exists('./model/Resnet_edit_B_params.pkl'):
        print('Edited data, train_method_2:')
        model.load('./model/Resnet_edit_B_params.pkl')
        test(model, val_loader)

