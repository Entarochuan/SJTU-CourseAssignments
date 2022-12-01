"""
    filename : Train_funcs.py
    Author : Atanis
    Description : 实现并封装基于jittor框架的训练函数
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

# print(jt.has_cuda)
jt.flags.use_cuda = jt.has_cuda


def train(Args, train_loader, test_loader, model, optimizer):
    """_summary_

    Args:
        Args : 
        train_loader (_type_): _description_
        test_loader (_type_): _description_
        use_data (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
    """
    print('Start training...')
    model.train()
    max_epoch  = Args.max_epoch
    batch_size = Args.batch_size
    
    for epoch in range(max_epoch):
        
        model.train()
        print('Training the {}th epoch'.format(epoch+1))
        
        for batch_idx, (X, X_gt) in enumerate(train_loader):
            # if batch_idx == 10:
            #     break
            imgs1 = jt.array(X['imgs1'], dtype=jt.float32)  # 问题出在要转换数据类型, solved 11/28 20:02
            imgs2 = jt.array(X['imgs2'], dtype=jt.float32)
            kpts1 = jt.array(X['kpts1'], dtype=jt.float32)
            kpts2 = jt.array(X['kpts2'], dtype=jt.float32)
            A1    = jt.array(X['A1'], dtype=jt.float32)
            A2    = jt.array(X['A2'], dtype=jt.float32)
            X_out = model(imgs1, imgs2, kpts1, kpts2, A1, A2)
            
            loss = pygm.utils.permutation_loss(X_out, X_gt)
            optimizer.step(loss)
            
            if batch_idx+1 % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_size * batch_idx, len(train_loader) ,
                    batch_size * 100. * batch_idx / len(train_loader), loss.numpy()[0]))
                
        use_data = [True, 1000]
        eval(Args, test_loader, model, epoch, use_data=use_data)    
    test_loader.remove_cache(True)    


def eval(Args, test_loader, model, epoch, use_data=[False, 1000], 
         classes = ['Face', 'Motorbike', 'Car', 'Duck', 'Winebottle'] ):
    
    """_summary_

    Args:
        test_loader 
        model 
        epoch (int): max peoch
        use_data (list, optional): 是否使用全部数据， 如否，使用多少数据
    """
    
    print('Start evaluating...')
    model.eval()

    batch_size = Args.batch_size
    
    test_loader.remove_cache()  # 每次测试前需要先清除cache!
    pred_dict_list = []
    loss_list = []
    

    for batch_idx, (X, X_gt) in enumerate(test_loader):
        
        if use_data[0] == False:
            if batch_idx >= use_data[1]:
                break

        imgs1 = jt.array(X['imgs1'], dtype=jt.float32)  # 问题出在要转换数据类型, solved 11/28 20:02
        imgs2 = jt.array(X['imgs2'], dtype=jt.float32)
        kpts1 = jt.array(X['kpts1'], dtype=jt.float32)
        kpts2 = jt.array(X['kpts2'], dtype=jt.float32)
        A1    = jt.array(X['A1'], dtype=jt.float32)
        A2    = jt.array(X['A2'], dtype=jt.float32)
        cls   = X['cls']
        ids   = X['ids']
        
        X_out = model(imgs1, imgs2, kpts1, kpts2, A1, A2)  # (B, 10, 10)
        loss = pygm.utils.permutation_loss(X_out, X_gt)
        loss_list.append(loss.numpy()[0])
        Avg_loss = np.array(loss_list).sum() / len(loss_list)
        
        # print(X_out) jt.var
        # loss = 0.
        # for i in range(batch_size):
        #     X_out_i = X_out[i]
        #     X_gt_i = X_gt[i]
        #     loss += pygm.utils.permutation_loss(X_out_i, X_gt_i)
            
        pred_dict = {
            'ids'     : None,
            'cls'     : None,
            'perm_mat' : None
        }
        
        if len(ids[0]) < batch_size:
            break
        for i in range(batch_size):
            pred_dict['ids'] = tuple([ids[0][i], ids[1][i]])
            pred_dict['cls'] = cls[i]
            pred_dict['perm_mat'] = X_out.numpy()[i]
            # print(pred_dict['perm_mat'].shape)
            pred_dict_list.append(pred_dict)
            
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t loss={} Avg_loss={}'.format(
                epoch+1, batch_size * batch_idx, len(test_loader) ,
                batch_size * 100. * batch_idx / len(test_loader), loss, Avg_loss))

       
    eval_val = test_loader.Evaluate(pred_dict_list, classes)

    print(eval_val)
    print('End evaluating.')            
    
