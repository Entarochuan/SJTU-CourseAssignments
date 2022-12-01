# """
#     filename : model.py
#     Author : Atanis
#     Description : 实现基于jittor框架的训练函数
#     Reference : jittor document 
#     conda env: ycm 
# """

# # python train_funcs.py

# # 框架
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
from jittor.models import Resnet18, densenet121, vgg16_bn, vgg16

# 实现
import args, dataset
import models
_ = jt.seed(999)
jt.flags.use_cuda = 1

# cnn1 = vgg16(True)
# print('1', cnn1.features)

# cnn2 = vgg16_bn(True)
# print(cnn2.features)

model = models.GMNet()
model.show_cnn()
# cnt=0
# obj_resize = (256, 256)

# # benchmark = pygm.benchmark.Benchmark(name='WillowObject', sets='test', obj_resize=(256, 256), problem='2GM')

# class CNNNet(nn.Module):
#     def __init__(self, vgg16_module):
#         super(CNNNet, self).__init__()
#         # The naming of the layers follow ThinkMatch convention to load pretrained models.
#         self.node_layers = nn.Sequential(*[_ for _ in list(vgg16_module.features.values())[:31]])
#         self.edge_layers = nn.Sequential(*[_ for _ in list(vgg16_module.features.values())[31:38]])

#     def execute(self, inp_img):
#         feat_local = self.node_layers(inp_img)
#         feat_global = self.edge_layers(feat_local)
#         return feat_local, feat_global
    

# # benchmark = pygm.benchmark.Benchmark(name='WillowObject', sets='train', obj_resize=(256, 256), problem='2GM')

# # print('Start loading cache.')
# # for batch_idx, (X, X_gt) in enumerate(test_loader):
# #     if batch_idx % 100 == 0:
# #         print(' [{}/{} ({:.0f}%)]\t'.format(
# #             batch_size * batch_idx, len(test_loader) ,
# #             batch_size * 100. * batch_idx / len(test_loader)))
# # print('Cache loaded.')


# def train(train_loader, test_loader, model, optimizer):
    
#     print('Start training...')
#     model.train()
#     # max_epoch = args.max_epoch
#     # seed = args.seed
#     # _ = jt.seed(seed)
    
#     max_epoch = 20
#     for epoch in range(max_epoch):
        
#         model.train()
#         print('Training the {}th epoch'.format(epoch+1))
        
#         for batch_idx, (X, X_gt) in enumerate(train_loader):
#             # if batch_idx==10:
#             #     break
#             imgs1 = jt.array(X['imgs1'], dtype=jt.float32)  # 问题出在要转换数据类型, solved 11/28 20:02
#             imgs2 = jt.array(X['imgs2'], dtype=jt.float32)
#             kpts1 = jt.array(X['kpts1'], dtype=jt.float32)
#             kpts2 = jt.array(X['kpts2'], dtype=jt.float32)
#             A1    = jt.array(X['A1'], dtype=jt.float32)
#             A2    = jt.array(X['A2'], dtype=jt.float32)
#             X_out = model(imgs1, imgs2, kpts1, kpts2, A1, A2)
            
#             loss = pygm.utils.permutation_loss(X_out, X_gt)
#             optimizer.step(loss)
            
#             if batch_idx % 10 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx, len(train_loader) ,
#                     100. * batch_idx / len(train_loader), loss.numpy()[0]))
                
#         eval(test_loader, model, epoch)    
#     test_loader.remove_cache(True)    
    
# def eval(test_loader, model, epoch):
    
#     print('Start evaluating...')
#     model.eval()

#     test_loader.remove_cache()  # 每次测试前需要先清除cache!
#     pred_dict_list = []

#     for batch_idx, (X, X_gt) in enumerate(test_loader):
        
#         if batch_idx == 50:
#             break
#         # vgg16_cnn = jt.models.vgg16_bn(True)
#         # cnn = CNNNet(vgg16_cnn)
#         # print(cnn.node_layers)
#         # print(list(vgg16_cnn.features.values())[30])
        
#         # print(X['kpts1'].shape) (B x 2 x 10)
#         # print(X['imgs1'].shape) (B x C x W x H) (16, 3, 256, 256)
#         # print(X['A2'])
#         # print(cnt)
#         # if cnt==59:
#         #     print(imgs1.shape)
#         # load the data
#         imgs1 = jt.array(X['imgs1'], dtype=jt.float32)  # 问题出在要转换数据类型, solved 11/28 20:02
#         imgs2 = jt.array(X['imgs2'], dtype=jt.float32)
#         kpts1 = jt.array(X['kpts1'], dtype=jt.float32)
#         kpts2 = jt.array(X['kpts2'], dtype=jt.float32)
#         A1    = jt.array(X['A1'], dtype=jt.float32)
#         A2    = jt.array(X['A2'], dtype=jt.float32)
#         cls   = X['cls']
#         ids   = X['ids']
#         # print(ids)
#         # print(ids)
#         # print(cls)
#         # print(ids)
#         # print(tuple([ids[0][0].data[0], ids[1][0].data[0]]))
#         # print(kpts1)
#         # rounded_kpts1 = jt.round_int(kpts1) # shape=(B, 2, 10)
#         # rounded_kpts2 = jt.round_int(kpts2) # shape=(B, 2, 10)
#         # print(rounded_kpts1)
        
#         X_out = model(imgs1, imgs2, kpts1, kpts2, A1, A2)  # (B, 10, 10)
#         # print(X_out) jt.var
#         loss = 0.
#         for i in range(batch_size):
#             X_out_i = X_out[i]
#             X_gt_i = X_gt[i]
#             loss += pygm.utils.permutation_loss(X_out_i, X_gt_i)
            
#         pred_dict = {
#             'ids'     : None,
#             'cls'     : None,
#             'perm_mat' : None
#         }
        
#         for i in range(batch_size):
#             pred_dict['ids'] = tuple([ids[0][i], ids[1][i]])
#             pred_dict['cls'] = cls[i]
#             pred_dict['perm_mat'] = X_out.numpy()[i]
#             # print(pred_dict['perm_mat'].shape)
#             pred_dict_list.append(pred_dict)
            
#         # print(len(pred_dict_list))

#         # # print(X_gt)
#         # loss = pygm.utils.permutation_loss(X_out, X_gt)
#         # print(f'loss={loss:.4f}')
        
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlenth: {} loss={}'.format(
#                 epoch+1, batch_idx, len(test_loader) ,
#                 100. * batch_idx / len(test_loader), len(pred_dict_list), loss))

#     classes = ['Face', 'Motorbike', 'Car', 'Duck', 'Winebottle']    
#     eval_val = test_loader.Evaluate(pred_dict_list, classes)
#     # eval_val = benchmark.eval(pred_dict_list, classes)
       
#     print(eval_val)
#     print('End evaluating.')            


# if __name__ =='__main__':
    
#     batch_size = 2

#     train_loader = dataset.GMdataset(mode='train', batch_size=batch_size, shuffle=True)
#     test_loader  = dataset.GMdataset(mode='test' , batch_size=batch_size)
#     learning_rate = 0.0000075
#     momentum = 0.91
#     weight_decay = 1e-5
#     epochs = 10

#     # nn.SGD()
#     model = models.GMNet()
#     # optimizer = nn.SGD(model.parameters(), momentum=momentum, lr=learning_rate, weight_decay=weight_decay)
#     optimizer = nn.Adam(model.parameters(), lr=learning_rate)

#     train(train_loader, test_loader, model, optimizer)