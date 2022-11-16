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
pygmtools.BACKEND = 'paddle'

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

    def __init__(self, hidden_size=256):
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
        self.linear_mid = nn.Sequential(nn.Linear(in_features=512, out_features=512),
                                        nn.ReLU(), 
                                        nn.Dropout(0.25))
        self.out = nn.Linear(in_features=512, out_features=output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        
        x = self.linear_mid(x)
        x = self.linear_mid(x)
        x = self.linear_mid(x)
        x = self.linear_mid(x)
        x = self.linear_mid(x)
        
        x = self.out(x)
        
        return x


class Multilayer_FC4(paddle.nn.Layer):

    def __init__(self, input_size=512, output_size=16):
        super(Multilayer_FC4, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=512)
        self.linear_mid = nn.Linear(in_features=512, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=output_size)
        # self.BatchNorm = nn.BatchNorm1D(output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear_mid(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.out(x)
        
        return x
    
    
class Stitch_Net(paddle.nn.Layer):

    def __init__(self, pic_num, cnn_hidden_size=256, batch_size=64, use_attention=False):
        super(Stitch_Net, self).__init__()
        self.pic_num = pic_num
        self.use_attention = use_attention
        self.cnn_hidden_Size = cnn_hidden_size
        
        if use_attention:
            self.FC7_hidden_size = cnn_hidden_size * 8  # 引入注意力机制 
        else:
            self.FC7_hidden_size = cnn_hidden_size * 4
            
        self.batch_size = batch_size
        self.Blk_1 = Resnet_Block_img(cnn_hidden_size)
        self.Attention = paddle.incubate.nn.FusedMultiHeadAttention(256, 2)  # Multi-head Attention
        self.feature_size = pic_num * pic_num
        self.FC7 = Multilayer_FC7(self.FC7_hidden_size, self.feature_size)
        # self.FC4 = Multilayer_FC4(4096, 16)
        
    def forward(self, x):
        
        """
        
            思路：变成(n*n, 1)维向量后reshape, loss相加
            
        """

        features = []
        query = paddle.to_tensor(np.empty((self.pic_num, self.batch_size, self.cnn_hidden_Size)), dtype='float32')
        
        for i in range(self.pic_num):
            xi = x[:,i,:,:]
            # print(xi.shape) 
            xi = paddle.to_tensor(xi, dtype='float32')
            feature_i = self.Blk_1(xi)          
            query[i] = feature_i
            features.append(feature_i)
            
        # print(query.shape): (4, 256, 256)     
        query = paddle.transpose(query, perm=[1, 0, 2])   
        Attend = self.Attention(query, query, query)  # Self Attention
        Attend = paddle.transpose(Attend, perm=[1, 0, 2])
        # print(Attend.shape)
        if self.use_attention:
            for i in range(4):
                features.append(Attend[i])
            
        feature = paddle.concat(x=features, axis=1) # print(feature.shape)
        feature = self.FC7(feature)
        length = self.pic_num
        feature = paddle.reshape(feature, [256, length, length]) + (1e-4)  # print(feature.shape)：(64, 16)
        
        output = pygmtools.linear_solvers.sinkhorn(feature)
        return output


# 加载预训练参数，训练Resnet分类网络。
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