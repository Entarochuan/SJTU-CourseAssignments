"""
    filename : model.py
    Author : Atanis
    Description : 基于jittor框架搭建模型
    Reference : jittor document 
    conda env: ycm 
"""

import pygmtools as pygm
import jittor as jt
from jittor.models import Resnet18, densenet121, vgg16_bn
import jittor.nn as nn
import numpy as np


def l2norm(node_feat):
    return nn.Functional.local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)
    
    
class CNNNet(nn.Module):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = nn.Sequential(*[_ for _ in list(vgg16_module.features.values())[:31]])
        self.edge_layers = nn.Sequential(*[_ for _ in list(vgg16_module.features.values())[31:38]])

    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global
    
    
class GMNet(nn.Module):
    
    def __init__(self, obj_resize=(256, 256), pretrained_gm=False):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=pretrained_gm) # fetch the network object
        self.vgg16_cnn = vgg16_bn(True)
        self.cnn = CNNNet(self.vgg16_cnn)
        self.obj_resize = obj_resize

    def execute(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        # feat1_local = l2norm(feat1_local)
        # feat1_global = l2norm(feat1_global)
        # feat2_local = l2norm(feat2_local)
        # feat2_global = l2norm(feat2_global)
        
        # print(feat1_global.shape) (1, 512, 16, 16)
        
        # upsample feature map
        feat1_local_upsample = nn.interpolate(feat1_local, self.obj_resize, mode='bilinear')
        feat1_global_upsample = nn.interpolate(feat1_global, self.obj_resize, mode='bilinear')
        feat2_local_upsample = nn.interpolate(feat2_local, self.obj_resize, mode='bilinear')
        feat2_global_upsample = nn.interpolate(feat2_global, self.obj_resize, mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)
        # print(feat1_upsample.shape) (B, 1024, 256, 256)
        
        # assign node features
        rounded_kpts1 = jt.round_int(kpts1) # shape=(B, 2, 10)
        rounded_kpts2 = jt.round_int(kpts2) # shape=(B, 2, 10)
        
        batch_size = rounded_kpts1.shape[0]
        node1 = jt.zeros((batch_size, 10, 1024))
        node2 = jt.zeros((batch_size, 10, 1024))
        
        for i in range(batch_size):
            rounded_kpts1_tmp = rounded_kpts1.data[i]
            rounded_kpts2_tmp = rounded_kpts2.data[i]
            print(rounded_kpts1)
            tmp = feat1_upsample[i, :, rounded_kpts1_tmp[0], rounded_kpts1_tmp[1]].t()
            print(tmp.data)
            node1[i] = feat1_upsample[i, :, rounded_kpts1_tmp[0], rounded_kpts1_tmp[1]].t().data
            node2[i] = feat2_upsample[i, :, rounded_kpts2_tmp[0], rounded_kpts2_tmp[1]].t()
            
            # print(feat1_upsample[i, :, rounded_kpts1_tmp[0], rounded_kpts1_tmp[1]].t().shape)
        
        # node1 = feat1_upsample[:, :, rounded_kpts1[:, 0], rounded_kpts1[:, 1]].t()  #  
        # node2 = feat2_upsample[:, :, rounded_kpts2[:, 1], rounded_kpts2[:, 1]].t()  # (1024, 10, B)
        # node1 = node1.transpose(2, 1, 0)  # (Batch_size, 10, 1024)  A1: (batch, 10, 10)
        # node2 = node2.transpose(2, 1, 0)  # (Batch_size, 10, 1024)  A2: (batch, 10, 10)

        # 
        # print(node1.shape)
        # print(node2.shape)
        # print(A1.shape)
        # print(A2.shape)
        # print(type(node1), type(node2), type(A1), type(A2)) 都是jittor.var
        # PCA-GM matching layers
        # A1 = jt.array(A1, dtype=jt.float32)
        # A2 = jt.array(A2, dtype=jt.float32)
        
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        return X


vgg16_cnn = vgg16_bn(True)
cnn = CNNNet(vgg16_cnn)







for epoch in range(10):
    
    # # 预热

    
    # # if cnt==10:
    # #     break
    # # cnt=0
    model.train()
    for batch_idx, (X, X_gt) in enumerate(train_loader):
        
        imgs1 = jt.array(X['imgs1'], dtype=jt.float32)  # 问题出在要转换数据类型, solved 11/28 20:02
        imgs2 = jt.array(X['imgs2'], dtype=jt.float32)
        kpts1 = jt.array(X['kpts1'], dtype=jt.float32)
        kpts2 = jt.array(X['kpts2'], dtype=jt.float32)
        A1    = jt.array(X['A1'], dtype=jt.float32)
        A2    = jt.array(X['A2'], dtype=jt.float32)
        X_out = model(imgs1, imgs2, kpts1, kpts2, A1, A2)
        loss = 0.
        for i in range(batch_size):
            X_out_i = X_out[i]
            X_gt_i = X_gt[i]
            loss += pygm.utils.permutation_loss(X_out_i, X_gt_i)
        # loss = pygm.utils.permutation_loss(X_out, X_gt)
        optimizer.step(loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_size * batch_idx, len(train_loader) ,
                batch_size * 100. * batch_idx / len(train_loader), loss.numpy()[0]))
            
    model.eval()
    
    pred_dict_list = []
    
    for batch_idx, (X, X_gt) in enumerate(test_loader):
        
        if batch_idx == 50:
            break
        # vgg16_cnn = jt.models.vgg16_bn(True)
        # cnn = CNNNet(vgg16_cnn)
        # print(cnn.node_layers)
        # print(list(vgg16_cnn.features.values())[30])
        
        # print(X['kpts1'].shape) (B x 2 x 10)
        # print(X['imgs1'].shape) (B x C x W x H) (16, 3, 256, 256)
        # print(X['A2'])
        # print(cnt)
        # if cnt==59:
        #     print(imgs1.shape)
        # load the data
        imgs1 = jt.array(X['imgs1'], dtype=jt.float32)  # 问题出在要转换数据类型, solved 11/28 20:02
        imgs2 = jt.array(X['imgs2'], dtype=jt.float32)
        kpts1 = jt.array(X['kpts1'], dtype=jt.float32)
        kpts2 = jt.array(X['kpts2'], dtype=jt.float32)
        A1    = jt.array(X['A1'], dtype=jt.float32)
        A2    = jt.array(X['A2'], dtype=jt.float32)
        cls   = X['cls']
        ids   = X['ids']
        # print(ids)
        # print(ids)
        # print(cls)
        # print(ids)
        # print(tuple([ids[0][0].data[0], ids[1][0].data[0]]))
        # print(kpts1)
        # rounded_kpts1 = jt.round_int(kpts1) # shape=(B, 2, 10)
        # rounded_kpts2 = jt.round_int(kpts2) # shape=(B, 2, 10)
        # print(rounded_kpts1)
        
        X_out = model(imgs1, imgs2, kpts1, kpts2, A1, A2)  # (B, 10, 10)
        # print(X_out) jt.var
        loss = 0.
        for i in range(batch_size):
            X_out_i = X_out[i]
            X_gt_i = X_gt[i]
            loss += pygm.utils.permutation_loss(X_out_i, X_gt_i)
            
        pred_dict = {
            'ids'     : None,
            'cls'     : None,
            'perm_mat' : None
        }
        
        for i in range(batch_size):
            pred_dict['ids'] = tuple([ids[0][i], ids[1][i]])
            pred_dict['cls'] = cls[i]
            pred_dict['perm_mat'] = X_gt.numpy()[i]
            print(pred_dict['perm_mat'].shape)
            pred_dict_list.append(pred_dict)
            
        # print(len(pred_dict_list))

        # # print(X_gt)
        # loss = pygm.utils.permutation_loss(X_out, X_gt)
        # print(f'loss={loss:.4f}')
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlenth: {} loss={}'.format(
                epoch, batch_idx, len(test_loader) ,
                100. * batch_idx / len(test_loader), len(pred_dict_list), loss))

    classes = ['Face', 'Motorbike', 'Car', 'Duck', 'Winebottle']    
    eval_val = test_loader.Evaluate(pred_dict_list, classes)
    # eval_val = benchmark.eval(pred_dict_list, classes)
    print(eval_val)
    
test_loader.remove_cache(True)    
# python train_funcs.py

# benchmark.rm_gt_cache(True)