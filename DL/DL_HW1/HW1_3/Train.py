"""
    filename : Model.py
    Author : Ma Yichuan
    Description : 实现图片切分模型
    Referrence : paddle document 
    
"""

from bdb import Breakpoint
import paddle 
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.metric import Accuracy
import paddle.vision.transforms as transforms
import itertools
import os, sys, logging
from paddle.vision.datasets import Cifar10
import Model
import numpy as np

# paddle.utils.run_check()

def Load_Data():
    
    normalize = transforms.Normalize(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375], data_format='CHW')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(), transforms.Transpose(),
        normalize,
    ])

    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform)

    batch_size = 64 
    train_loader = paddle.io.DataLoader(train_dataset, return_list=True, shuffle=True, 
                                        batch_size=batch_size, drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset, return_list=True, shuffle=True, 
                                    batch_size=batch_size, drop_last=True)

    return train_loader, test_loader, train_dataset, test_dataset

    
def train(model, train_loader, test_loader, loss, epochs, optimizer):
    
    print('start training...')
    model.train()
    best_acc = 0 
    for epoch in range(epochs):
        
        for batch_idx, (img, _) in enumerate(train_loader()) :  
            
            X_split, label = Model.random_cut(img)
            output = model(X_split)
            
            label  = paddle.to_tensor(label)
            output = paddle.to_tensor(output)
            
            ls = loss(output, label)
            ls.mean().backward()
            optimizer.step()
            optimizer.clear_grad()
            
            if batch_idx % 100 == 0:
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx, int(len(train_loader)),
                    100. * batch_idx / float(len(train_loader)), ls.item()))

        # evaluate
        if best_acc == 0:
            best_acc = Model.Evaluate(model, test_loader, 0)
        else:
            best_acc = Model.Evaluate(model, test_loader, best_acc)

        print('All time best Accuracy: {:.6f}'.format(best_acc))
    
    paddle.save(model.Blk_1.state_dict(), './Pretrained_Resnet/pretrain.pdparams')


def Train_Pretrained_Model(train_dataset, test_dataset):
    """
        把Resnet部分作为预训练模型导入。
    """
    
    layer_state_dict = paddle.load('./Pretrained_Resnet/pretrain.pdparams')
    # Pretrained_Model = Model.Resnet_Block_img(512)
    
    model = Model.Resnet_Classification(512)
    model.set_state_dict(layer_state_dict)
    
    model = paddle.Model(model) 
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), Accuracy())
    model.fit(train_dataset, epochs=2, batch_size=64, verbose=1)
    model.evaluate(test_dataset, batch_size=64, verbose=1)

    
if __name__ == "__main__":
    
    paddle.device.set_device("gpu:1")
    train_loader, test_loader , train_dataset, test_dataset = Load_Data()
    
    batch_size = 64
    learning_rate = 0.05
    momentum = 0.91
    weight_decay = 1e-4
    epochs = 10  
    
    model = Model.PremNet(pic_num=4, cnn_hidden_size=512, batch_size=batch_size)
    loss = nn.CrossEntropyLoss(use_softmax=False)
    
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,parameters=model.parameters(), weight_decay=weight_decay)


    """
    
        train : 训练模型
        
        Train_Pretrained_Model : 将预训练参数加载到Resenet模型中。
        
    """
    
    train(model, train_loader,test_loader, loss, epochs, optimizer)
    Train_Pretrained_Model(train_dataset, test_dataset)
    
        
    
