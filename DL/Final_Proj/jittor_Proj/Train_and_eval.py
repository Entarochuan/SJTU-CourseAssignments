"""
    filename : Train_and_eval.py
    Author : Atanis
    Description : 训练主函数
    Reference : jittor document 
    conda env: ycm 
"""

import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image
import json
from jittor.dataset import Dataset
import jittor as jt
import jittor.nn as nn

# 实现
import args, dataset
import models
import Train_funcs


if __name__ =='__main__':
    
    # load arguments
    Args          = args.init_args()
    max_epoch     = Args.max_epoch
    batch_size    = Args.batch_size
    seed          = Args.seed
    learning_rate = Args.lr
    opt_choice    = Args.optimizer
    cnn_choice    = Args.cnn_type
    
    # set arguments
    jt.flags.use_cuda = jt.has_cuda
    _ = jt.seed(seed)
    
    # set training settings
    train_loader = dataset.GMdataset(mode='train', batch_size=batch_size)  
    test_loader  = dataset.GMdataset(mode='test' , batch_size=batch_size)  
    
    model = models.GMNet(cnn_choice=cnn_choice)
    if opt_choice == 'Adam':
        optimizer = nn.Adam(model.parameters(), lr=learning_rate)      
    elif opt_choice == 'SGD':
        momentum     = Args.SGD_settings[0]
        weight_decay = Args.SGD_settings[0]
        optimizer = nn.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
    print('Args and settings loaded, start training.')
    
    # Start train and eval
    Train_funcs.train(Args, train_loader, test_loader, model, optimizer)
    