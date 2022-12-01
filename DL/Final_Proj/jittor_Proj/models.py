"""
    filename : model.py
    Author : Atanis
    Description : 基于jittor框架搭建模型
    Reference : jittor document 
    conda env: ycm 
"""

import pygmtools as pygm
import jittor as jt
from jittor.models import Resnet18, densenet121, vgg16_bn, vgg16
import jittor.nn as nn
import numpy as np
jt.flags.use_cuda = jt.flags.use_cuda = jt.has_cuda


def l2norm(node_feat):
    return local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)
    

# 改写自pytorch document    
def local_response_norm(input: jt.Var, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    
    sizes = input.size()
    avgpool_3d = nn.AvgPool3d((size, 1, 1), stride=1)
    dim = input.ndim
    if dim < 3:
        raise ValueError(
            "Expected 3D or higher dimensionality \
                         input (got {} dimensions)".format(
                dim
            )
        )

    if input.numel() == 0:
        return input

    div = nn.matmul(input, input).unsqueeze(1)
    if dim == 3: 
        div = nn.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avgpool_3d(div).squeeze(1)
        div = div.view(sizes)
    
    div = (alpha * div).add(k).pow(beta)
    # div = nn.matmul(div, alpha).add(k).pow(beta)
    # div = div.mul(alpha).add(k).pow(beta)
    return input / div

    
class CNNNet(nn.Module):
    def __init__(self, CNN_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = nn.Sequential(*[_ for _ in list(CNN_module.features.values())[:31]])
        self.edge_layers = nn.Sequential(*[_ for _ in list(CNN_module.features.values())[31:38]])

    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global
    
    
class GMNet(nn.Module):
    
    def __init__(self, obj_resize=(256, 256), cnn_choice='vgg16', pretrained_gm=False):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=pretrained_gm) # fetch the network object
        # self.vgg16_cnn = vgg16_bn(True)
        
        if  cnn_choice =='vgg16':
            self.pretrained_cnn = vgg16(True)
        elif cnn_choice == 'vgg16_bn':
            self.pretrained_cnn = vgg16_bn(False)
        elif cnn_choice == 'Resnet18':
            self.pretrained_cnn = Resnet18(True)
        
        self.cnn = CNNNet(self.pretrained_cnn)
        
        self.obj_resize = obj_resize

    def execute(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        # print(feat1_local.shape) # (B, 512, 8, 8)
        # print(feat1_global.shape) # (B, 512, 8, 8)
         
        # l2norm 
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)
        # print(feat1_local.shape)
        # print(feat1_global.shape)
        # upsample feature map
        feat1_local_upsample = nn.interpolate(feat1_local, self.obj_resize, mode='bilinear')
        feat1_global_upsample = nn.interpolate(feat1_global, self.obj_resize, mode='bilinear')
        feat2_local_upsample = nn.interpolate(feat2_local, self.obj_resize, mode='bilinear')
        feat2_global_upsample = nn.interpolate(feat2_global, self.obj_resize, mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)
        # print(feat1_upsample.shape) (B, 1024, 256, 256)
        # print(feat1_global_upsample.shape) (B, 512, 256, 256)
        
        # assign node features
        rounded_kpts1 = jt.round_int(kpts1) # shape=(B, 2, 10)
        rounded_kpts2 = jt.round_int(kpts2) # shape=(B, 2, 10)
        
        batch_size = rounded_kpts1.shape[0]
        node1 = jt.zeros((batch_size, 10, 1024))
        node2 = jt.zeros((batch_size, 10, 1024))
        
        for i in range(batch_size):
            rounded_kpts1_tmp = rounded_kpts1.data[i]
            rounded_kpts2_tmp = rounded_kpts2.data[i]
            # tmp = feat1_upsample[i, :, rounded_kpts1_tmp[0], rounded_kpts1_tmp[1]].t()
            # print(tmp.data)
            node1[i] = feat1_upsample[i, :, rounded_kpts1_tmp[0], rounded_kpts1_tmp[1]].t()
            node2[i] = feat2_upsample[i, :, rounded_kpts2_tmp[0], rounded_kpts2_tmp[1]].t()
        
        # print(node1)
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        return X

    def show_cnn(self):
        print('node layers:', self.cnn.node_layers.features)
        print('eedge layers:', self.cnn.edge_layers.feaures)