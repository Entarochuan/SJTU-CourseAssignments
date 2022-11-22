"""

    filename : Train_and_eval.py
    author : Yichuan Ma
    Date : 2022/11/12
    Description : Train and evaluate the modeels
    Reference :  Spikingjelly Document
    
"""

import torch
import numpy as np
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torchvision import transforms
from torch.utils.data import DataLoader
import models
from spikingjelly.clock_driven import functional, surrogate, neuron
from torchvision.datasets import CIFAR10
import torchvision
import torch.nn.functional as F


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  #每行中最大的数作为预测类别
    cmp = y_hat.type(y.dtype) == y #转换数据类型后作比较
    return float(cmp.type(y.dtype).sum())


def evaluation(model, test_loader, SNN=False):
    
    print('evaluating...')
    model.eval()
    metric = np.array([0., 0.])
    accuracies = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            # print(x)
            # print(frames.shape)
            # if frames.shape[1] < 32:
            #     break
            x = torch.tensor(x, dtype = torch.float32)
            
            if SNN:
                x = x.transpose(0, 1)

            functional.reset_net(model)
            y_pred = model(x)
            metric[0] = metric[0] + accuracy(y_pred, y)
            metric[1] = metric[1] + y.numel()
            accuracies.append(metric[0]/metric[1])

            if batch_idx % 50 == 0:
                print('batch:[{}/{} ({:.0f}%)]\tAccuracy: {:.6f}'.format(
                    batch_idx, int(len(test_loader)),
                    100. * batch_idx / float(len(test_loader)), accuracies[batch_idx]))
            functional.reset_net(model)
    print('end evaluation.')


# print(torch.cuda.is_available())
# root_dir = './data/cifar_10'
# transfrom = transforms.ToTensor()
# train_set = CIFAR10DVS(root_dir, data_type='frame', frames_number=8, split_by='number')
# test_set = CIFAR10DVS(root_dir, data_type='frame', frames_number=8, split_by='number')




# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((32, 32)),
#     torchvision.transforms.ToTensor()]
# )

# # print(torch.cuda.is_available())
# # print(torch.__version__)

# train_set = CIFAR10(root='./data/origin_data/', train=True, transform=transform, download=True)
# test_set  = CIFAR10(root='./data/origin_data/', train=False, transform=transform, download=True)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=True)

# model = models.DVSGestureNet(channels=3, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True).to(device)
# model = models.SNN_Net(tau=tau, T=T).to(device)

# model = models.CNN_Net()
    
 # trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, shuffle=False)



def train_SNN():
    
    device = 'cuda:0'
    batch_size = 32
    learning_rate = 5e-2
    T = int(8)
    tau = float(2.0)
    train_epoch = int(20)
    log_dir = './log/'

    root_dir = './data/cifar_10'
    transfrom = transforms.ToTensor()
    train_set = CIFAR10DVS(root_dir, data_type='frame', frames_number=8, split_by='number')
    test_set = CIFAR10DVS(root_dir, data_type='frame', frames_number=8, split_by='number')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=True)
    model = models.SNN_Net(tau=tau, T=T)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()
    
    for epoch in range(train_epoch):
        model.train()
        for batch_idx, (frames, label) in enumerate(train_loader):
            # print(x)
            print(frames.shape)
            if frames.shape[0] < 32:
                break
            optimizer.zero_grad()
            frames = torch.tensor(frames, dtype = torch.float32)
            # frames = frames.to(device)
            frames = frames.transpose(0,1)
            # label = label.to(device)
            
            pred = model(frames)
            l = loss(pred, label)
            l.backward(retain_graph=True)
            optimizer.step()
            
            functional.reset_net(model)
            
            if batch_idx % 50 == 0:
                # print(torch.cuda.get_device_capability(device))
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx, int(len(train_loader)),
                    100. * batch_idx / float(len(train_loader)), l.item()))
        
        evaluation(model, test_loader, SNN=True)   
        model.train()

    
def train_CNN():
    
    device = 'cuda:0'
    batch_size = 32
    learning_rate = 5e-2
    T = int(8)
    tau = float(2.0)
    train_epoch = int(20)
    log_dir = './log/'
    
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()]
    )
    train_set = CIFAR10(root='./data/origin_data/', train=True, transform=transform, download=True)
    test_set  = CIFAR10(root='./data/origin_data/', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=True)
    model = models.CNN_Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()
    
    for epoch in range(train_epoch):
        model.train()
        for batch_idx, (frames, label) in enumerate(train_loader):
            
            optimizer.zero_grad()
            frames = torch.tensor(frames, dtype = torch.float32)
            
            pred = model(frames)
            l = loss(pred, label)
            l.backward(retain_graph=True)
            optimizer.step()
            
            # functional.reset_net(model)
            
            if batch_idx % 50 == 0:
                # print(torch.cuda.get_device_capability(device))
                print('Epoch: {} batch:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx, int(len(train_loader)),
                    100. * batch_idx / float(len(train_loader)), l.item()))
        
        evaluation(model, test_loader)   
        model.train()
        

if __name__ == '__main__':
    train_CNN()