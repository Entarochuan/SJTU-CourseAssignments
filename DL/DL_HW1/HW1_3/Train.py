"""
    filename : Model.py
    Author : Ma Yichuan
    Description : 实现图片切分模型
    Referrence : paddle document 
    conda env: DL on server
"""

# Requiring Paddle Tools
import paddle 
import paddle.nn.functional as F
import paddle.nn as nn
import paddle.vision.transforms as transforms
from paddle.metric import Accuracy

import random
import sys

# Functions implemented 
import Model
import numpy as np
from dataset import * 
import Args

# paddle.utils.run_check()

def Load_Data(batch_size):
    
    """ Load Cifar10 dataset

    Args:
        batch_size (int): Batch_szie

    Returns:
        dataloader, dataset 
    """
    normalize = transforms.Normalize(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375], data_format='CHW')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(), transforms.Transpose(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(), transforms.Transpose(),
        normalize,
    ])


    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
    test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform_test)

    return train_dataset, test_dataset

    
def train(model, train_loader, test_loader, loss, epochs, optimizer, visual=False):
            
    print('start training...')
    
    model.train()
    best_acc = 0 
    
    for epoch in range(epochs):
        
        for batch_idx, (img, _) in enumerate(train_loader()) :  
            
            # print(img.shape) shape = (256, 4, 3, 16, 16)
            
            if img.shape[0] < 256 :
                break
            output = model(img)
            ls = loss(output, _)
            
            ls.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            if (batch_idx+1) % 10 == 0:
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx+1, int(len(train_loader)),
                    100. * batch_idx / float(len(train_loader)), ls.item()))
                
        # evaluate            
        if best_acc == 0:
            best_acc = Evaluate(model, test_loader, 0)
        else:
            best_acc = Evaluate(model, test_loader, best_acc)

        # print('All time best Accuracy: {:.6f}'.format(best_acc))
    
    paddle.save(model.Blk_1.state_dict(), './Pretrained_Resnet/pretrain.pdparams')


def Train_Pretrained_Model(train_dataset, test_dataset, use_pretrained=True):
    
    """
        把Resnet部分作为预训练模型导入。
    """
    
    model = Model.Resnet_Classification(512)
    
    if use_pretrained:
        layer_state_dict = paddle.load('./Pretrained_Resnet/pretrain.pdparams')
        # Pretrained_Model = Model.Resnet_Block_img(512)
        
        model.set_state_dict(layer_state_dict)
        
        model = paddle.Model(model) 
        optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        model.prepare(optim, paddle.nn.CrossEntropyLoss(), Accuracy())
        model.fit(train_dataset, epochs=30, batch_size=64, verbose=1)
        model.evaluate(test_dataset, batch_size=64, verbose=1)
    
    else:
        model = paddle.Model(model) 
        optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        model.prepare(optim, paddle.nn.CrossEntropyLoss(), Accuracy())
        model.fit(train_dataset, epochs=30, batch_size=64, verbose=1)
        model.evaluate(test_dataset, batch_size=64, verbose=1)

    
if __name__ == "__main__":
    
    
    # Initialize Args
    args = Args.init_args(sys.argv[1:])
    # print(args)
    
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    seed =  args.seed
    device = args.device
    lr = args.lr
    data_path = args.data_path
    hidden_size = args.hidden_size
    use_attention = args.use_attention
    
    # Pre-Settings
    random.seed(999)
    device_use = 'gpu:'+str(device)
    paddle.device.set_device(device_use)
    
    # Load data
    cifar10_train_new = Cifar_Cut('Train', data_path = data_path)
    cifar10_test_new = Cifar_Cut('Test', data_path = data_path)
    train_loader = paddle.io.DataLoader(cifar10_train_new, shuffle=True, batch_size=batch_size)
    test_loader = paddle.io.DataLoader(cifar10_test_new, shuffle=True, batch_size=batch_size)

    # model initialization
    model = Model.Stitch_Net(pic_num=4, cnn_hidden_size=hidden_size, batch_size=batch_size, use_attention=use_attention)
    loss = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(learning_rate=lr,parameters=model.parameters())
    
    # visualize
    # model = paddle.Model(model)
    # model.summary((256, 4, 3, 16, 16))
    
    """
    
        train : 训练模型
        
        Train_Pretrained_Model : 将预训练参数加载到Resenet模型中训练。
        
    """
    
    train(model, train_loader,test_loader, loss, max_epoch, optimizer)
    
    train_dataset, test_dataset = Load_Data(64)
    Train_Pretrained_Model(train_dataset, test_dataset, False)
    
        
    
