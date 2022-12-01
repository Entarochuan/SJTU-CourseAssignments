"""
==========================================================
Precess the giving example step by step
==========================================================

"""

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
import json

benchmark = pygm.benchmark.Benchmark(name='WillowObject', sets='train', obj_resize=(256, 256), problem='2GM')

with open('./data/WillowObject/train.json', 'r') as file:
    str = file.read()
    data = json.loads(str)
    
# print(data[0])


# print(train_list[0])
data_list, perm_mat_dict, ids = benchmark.get_data([data[0], data[1]])
# print(len(data_list))
# # print(ids)
# # print(torch.cuda.is_available())
# # print(data)
# print(perm_mat_dict)
# # python test.py
len = benchmark.compute_img_num()
print(len)