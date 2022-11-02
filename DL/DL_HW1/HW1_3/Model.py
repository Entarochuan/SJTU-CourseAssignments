"""
    filename : Model.py
    Author : Ma Yichuan
    Description : 实现图片切分模型
    Reference : paddle document
"""

# 期望输入: 4个切分后的数组

import paddle.nn as nn
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor
import numpy as np
import math
import paddle.fluid as fluid
import pygmtools


class ResidualBlock(nn.Layer):
    def __init__(self, inchannel, outchannel, stride):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2D(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias_attr=False),
            nn.BatchNorm2D(outchannel),
            nn.ReLU(),
            nn.Conv2D(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(outchannel)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            # 使x的变化一致 使用与F相同的outchannel与stride
            self.shortcut = nn.Sequential(
                nn.Conv2D(inchannel, outchannel, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(outchannel)
            )


    def forward(self, x):
        out = self.left(x)
        # 这里是直接加上去了，DenseNet是累起来的，相当于扩充了通道数
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
# Resnet_Block get features
class Resnet_Block_img(paddle.nn.Layer):

    def __init__(self, hidden_size=512):
        super(Resnet_Block_img, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )
                
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fcout = nn.Linear(in_features=2048, out_features=hidden_size)


    def make_layer(self, block, channels, num_blocks, stride):
        # 堆叠的层数与num_blocks有关
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            # 储存上次输出的通道数，以便一次输入回去
            self.inchannel = channels
        return nn.Sequential(*layers)
    
      
    def forward(self, x):
        # 3*32*32
        out = self.conv1(x)
        # 64*32*32
        out = self.layer1(out)
        # 256*8*8
        out = self.layer2(out)
        # 512*4*4
        out = F.avg_pool2d(out,4)
        out = paddle.reshape(out,[out.shape[0], -1])
        out = self.fcout(out)
        return out


class Multilayer_FC7(paddle.nn.Layer):

    def __init__(self, input_size, output_size):
        super(Multilayer_FC7, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=512)
        self.linear_mid = nn.Linear(in_features=512, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=4096)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        
        x = self.out(x)
        
        return x


class Multilayer_FC4(paddle.nn.Layer):

    def __init__(self, input_size=4096, output_size=16):
        super(Multilayer_FC4, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=256)
        self.linear_mid = nn.Linear(in_features=256, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=output_size)
        # self.BatchNorm = nn.BatchNorm1D(output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.out(x)
        
        return x
    

class PremNet(paddle.nn.Layer):

    def __init__(self, pic_num, cnn_hidden_size=512, batch_size=64):
        super(PremNet, self).__init__()
        self.pic_num = pic_num
        self.cnn_hidden_Size = cnn_hidden_size * 4 
        self.batch_size = batch_size
        self.Blk_1 = Resnet_Block_img(512)
        self.feature_size = pic_num * pic_num
        self.FC7 = Multilayer_FC7(self.cnn_hidden_Size, self.feature_size)
        self.FC4 = Multilayer_FC4(4096, 16)
        
    def forward(self, x):
        
        """
        
            思路：变成(n*n, 1)维向量后reshape, loss相加
            
        """

        features = []
        for xi in x:
            feature_i = self.Blk_1(xi)
            features.append(feature_i)

        feature = paddle.concat(x=features, axis=1)
        # print(feature.shape)
        feature = self.FC7(feature)
        feature = self.FC4(feature)
        length = self.pic_num
        feature = paddle.reshape(feature, [self.batch_size, length, length])
        output = pygmtools.linear_solvers.sinkhorn(np.array(feature))
        return output


def random_cut(img, len=4):
    
    """
        左上:0, 右上:1, 左下:2, 右下:3
        打乱的顺序表示图片放置的位置.
        Example : order[0] = 1 表示原来放在左上的图像现在放在了右下
    """
    
    order = np.arange(len)
    order = np.random.permutation(order)  # 打乱的顺序表示图片放置的位置
    
    X_split = paddle.split(img, 2, axis=2)
    X1_y_split = paddle.split(X_split[0], 2, axis=3)
    X2_y_split = paddle.split(X_split[1], 2, axis=3)
    X_split_tmp = X1_y_split + X2_y_split
    X_split = X_split_tmp

    """ 重排序 """
    
    for i in range(len):
        X_split[order[i]] = X_split_tmp[i]
    
    # print('output shape=', output.shape)
    # print(output)
    # print(order)
    
    """ 标签 """

    label_tmp = np.empty((1, 4, 1))

    # label_tmp[0][0][order[0]] = 1
    # label_tmp[0][1][order[1]] = 1
    # label_tmp[0][2][order[2]] = 1
    # label_tmp[0][3][order[3]] = 1
    
    label_tmp[0][0] = [order[0]]
    label_tmp[0][1] = [order[1]]
    label_tmp[0][2] = [order[2]]
    label_tmp[0][3] = [order[3]]
    
    label_tmp = paddle.to_tensor(label_tmp, dtype='int64')
    # print(label_tmp.shape)
    label = label_tmp
    for i in range(63):
        label = paddle.concat((label, label_tmp))
    
    # print('label shape=', label.shape)
    
    return X_split, label


def Evaluate(model, test_loader, best_acc):

    print('evaluating...')
    model.eval()
    accuracies = []
    result = [0, 0]
    for batch_idx, (img, _) in enumerate(test_loader()):
        # print(img.shape)
        X_split, label = random_cut(img)
        output = model(X_split)
        
        for i in range(len(label)): # 每一组切分图片
            
            label_i  = label[i]    # 4*1
            output_i = output[i]   # 4*4
            
            for j in range(4):
                
                if np.argmax(output_i[j]) != label_i[j]:
                    flag = False
                    result[1] = result[1] + 1
                    break
                
            result[0] = result[0] + 1
            result[1] = result[1] + 1
        
        accuracies.append( result[0] / result[1] )
        
        # if accuracies[batch_idx] > best_acc :
        #     best_acc = accuracies[batch_idx]
        
        if batch_idx % 30 == 0 and batch_idx >=60 :
            print('batch:[{}/{} ({:.0f}%)]\tAccuracy: {:.6f}'.format(
                    batch_idx, int(len(test_loader)),
                    100. * batch_idx / float(len(test_loader)), accuracies[batch_idx]))
    
    if accuracies[len(test_loader)-1] > best_acc : 
        best_acc = accuracies[batch_idx]       
    # print('Best Accuracy: {:.6f}'.format(best_acc))        
    print('End evaluation, returning to training function.')      
    model.train()
    
    return best_acc


class Resnet_Classification(nn.Layer):

    def __init__(self, hidden_size=512):
        super(Resnet_Classification, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )
                
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.Class = nn.Linear(in_features=8192, out_features=10)


    def make_layer(self, block, channels, num_blocks, stride):
        # 堆叠的层数与num_blocks有关
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            # 储存上次输出的通道数，以便一次输入回去
            self.inchannel = channels
        return nn.Sequential(*layers)
    
      
    def forward(self, x):
        # 3*32*32
        out = self.conv1(x)
        # 64*32*32
        out = self.layer1(out)
        # 256*8*8
        out = self.layer2(out)
        # 512*4*4
        out = F.avg_pool2d(out,4)
        out = paddle.reshape(out,[out.shape[0], -1])
        out = self.Class(out)
        return out