# coding: utf-8
"""
==========================================================
Matching Image Keypoints by Graph Matching Neural Networks
==========================================================

This example shows how to match image keypoints by neural network-based graph matching solvers.
These graph matching solvers are designed to match two individual graphs. The matched images
can be further passed to tackle downstream tasks.
"""

# Author: Runzhong Wang <runzhong.wang@sjtu.edu.cn>
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 3

##############################################################################
# .. note::
#     The following solvers are based on matching two individual graphs, and are included in this example:
#
#     * :func:`~pygmtools.neural_solvers.pca_gm` (neural network solver)
#
#     * :func:`~pygmtools.neural_solvers.ipca_gm` (neural network solver)
#
#     * :func:`~pygmtools.neural_solvers.cie` (neural network solver)
#
import torch # pytorch backend
import torchvision # CV models
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image

pygm.BACKEND = 'pytorch' # set default backend for pygmtools

##############################################################################
# Predicting Matching by Graph Matching Neural Networks
# ------------------------------------------------------
# In this section we show how to do predictions (inference) by graph matching neural networks.
# Let's take PCA-GM (:func:`~pygmtools.neural_solvers.pca_gm`) as an example.
#
# Load the images
# ^^^^^^^^^^^^^^^^
# Images are from the Willow Object Class dataset (this dataset also available with the Benchmark of ``pygmtools``,
# see :class:`~pygmtools.dataset.WillowObject`).
#
# The images are resized to 256x256.
#

obj_resize = (256, 256)
img1 = Image.open('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0001.png')
img2 = Image.open('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0002.png')
kpts1 = torch.tensor(sio.loadmat('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0001.mat')['pts_coord'])
kpts2 = torch.tensor(sio.loadmat('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0002.mat')['pts_coord'])
kpts1[0] = kpts1[0] * obj_resize[0] / img1.size[0]
kpts1[1] = kpts1[1] * obj_resize[1] / img1.size[1]
kpts2[0] = kpts2[0] * obj_resize[0] / img2.size[0]
kpts2[1] = kpts2[1] * obj_resize[1] / img2.size[1]
# print(kpts1)
img1 = img1.resize(obj_resize, resample=Image.BILINEAR)
img2 = img2.resize(obj_resize, resample=Image.BILINEAR)
torch_img1 = torch.from_numpy(np.array(img1, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(0) # shape: BxCxHxW
torch_img2 = torch.from_numpy(np.array(img2, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(0) # shape: BxCxHxW
# print(torch_img1.shape) torch.Size([1, 3, 256, 256])


def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = torch.zeros(len(kpt[0]), len(kpt[0]))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

print(torch_img1.shape)
A1 = delaunay_triangulation(kpts1)
A2 = delaunay_triangulation(kpts2)
# print(A1.shape) (10 x 10)

vgg16_cnn = torchvision.models.vgg16_bn(True)
class CNNNet(torch.nn.Module):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = torch.nn.Sequential(*[_ for _ in vgg16_module.features[:31]])
        self.edge_layers = torch.nn.Sequential(*[_ for _ in vgg16_module.features[31:38]])

    def forward(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global
    
def l2norm(node_feat):
    return torch.nn.functional.local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)
    
class GMNet(torch.nn.Module):
    def __init__(self):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=False) # fetch the network object
        self.cnn = CNNNet(vgg16_cnn)

    def forward(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        print(img1.shape)
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)
        
        # print(feat1_global.shape) (1, 512, 16, 16)
        # print(kpts1.shape) (2, 10)
        # upsample feature map
        feat1_local_upsample = torch.nn.functional.interpolate(feat1_local, obj_resize, mode='bilinear')
        feat1_global_upsample = torch.nn.functional.interpolate(feat1_global, obj_resize, mode='bilinear')
        feat2_local_upsample = torch.nn.functional.interpolate(feat2_local, obj_resize, mode='bilinear')
        feat2_global_upsample = torch.nn.functional.interpolate(feat2_global, obj_resize, mode='bilinear')
        feat1_upsample = torch.cat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = torch.cat((feat2_local_upsample, feat2_global_upsample), dim=1)
        # print(feat1_upsample.shape) (1, 1024, 256, 256)
        # assign node features
        rounded_kpts1 = torch.round(kpts1).to(dtype=torch.long)
        rounded_kpts2 = torch.round(kpts2).to(dtype=torch.long)
        # print(rounded_kpts1.shape) (2, 10)
        # print(rounded_kpts1[0]) tensor([200, 170, 139, 127, 218, 101,  57,  21,  85, 153])
        node1 = feat1_upsample[0, :, rounded_kpts1[0], rounded_kpts1[1]].t()  # shape: NxC
        node2 = feat2_upsample[0, :, rounded_kpts2[0], rounded_kpts2[1]].t()  # shape: NxC

        # PCA-GM matching layers
        # print(node1.shape) (10, 1024)
        # print(node2.shape)
        # print(A1.shape) (10, 10)
        # print(A2.shape)
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        return X


cnt=0
for _ in vgg16_cnn.features[:31]:
    
    print(_)

    cnt+=1
    if cnt==1:
        break
    
model = GMNet()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
X = model(torch_img1, torch_img2, kpts1, kpts2, A1, A2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(t)

X_gt = torch.eye(X.shape[0])
print(X.shape[0])
print(X_gt.shape)

# loss = pygm.utils.permutation_loss(X, X_gt)
# print(f'loss={loss:.4f}')
