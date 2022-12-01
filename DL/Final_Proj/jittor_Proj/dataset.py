"""
    filename: dataset.py
    description: 数据集加载
    Reference : jittor document 
    conda env: ycm 
"""

import pygmtools as pygm
import numpy as np
import json
from jittor.dataset import Dataset
import scipy.spatial as spa
import itertools
import jittor as jt
from PIL import Image
pygm.BACKEND = 'jittor' # set default backend for pygmtools


# Build graph    
def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = np.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A   

        
class GMdataset(Dataset):
    """_summary_

    Args:
        batch_size 的大小不能太大,超过 8 时占用显存超出范围。 (训练gpu为DL课程提供的RTX 3090ti)
    """
    
    def __init__(self, mode = 'train', obj_resize = (256, 256), batch_size=1, shuffle=False, drop_last=False, num_workers=0, buffer_size=512 * 1024 * 1024, stop_grad=True, keep_numpy_array=False, endless=False):
        super().__init__(batch_size, shuffle, drop_last, num_workers, buffer_size, stop_grad, keep_numpy_array, endless)
        
        self.mode = mode
        self.benchmark = pygm.benchmark.Benchmark(name='WillowObject', sets=mode, obj_resize=(256, 256), problem='2GM')
        self.ids, self.length = self.benchmark.get_id_combination()
        self.class_num = len(self.ids)
        self.resize = obj_resize
        
        # if mode == 'train':
        #     with open('./data/WillowObject/train.json', 'r') as file:
        #         str = file.read()
        #         self.data = json.loads(str)
        
        # elif mode == 'test':
        #     with open('./data/WillowObject/test.json', 'r') as file:
        #         str = file.read()
        #         self.data = json.loads(str)
            
        # else:
        #     print('Invalid mode, switch between train and test.')
    
    def __getitem__(self, index):
        """_summary_

        Args:
            index (int):  输入整数,从id_pair中取出指定位置的pair.
        """
        
        assert index < self.benchmark.compute_length()
        
        for cls in range(self.class_num):

            if index <= len(self.ids[cls])-1:
                id_pair = self.ids[cls][index]
                data_pair, perm_mat_dict, ids = self.benchmark.get_data(list(id_pair), shuffle=True)
                # print(ids)
                
                perm_mat =  perm_mat_dict[(0, 1)].toarray()
                # print(data_pair[0]['kpts'][0]) 第一张图的第一对kpts
                
                # print(data_pair) , print(perm_mat)

                data_dic = {
                    'imgs1' : jt.array(data_pair[0]['img'].transpose(2, 0, 1)), 
                    'imgs2' : jt.array(data_pair[1]['img'].transpose(2, 0, 1)),
                    'kpts1' : np.zeros((2, 10)), 
                    'kpts2' : np.zeros((2, 10)), 
                    'A1'    : None, 
                    'A2'    : None,
                    'cls'   : data_pair[0]['cls'],
                    'ids'   : ids
                }
                
                for i in range(10):
                    # print(data_pair[0]['kpts'][i]['x']) float值
                    data_dic['kpts1'][0][i] = data_pair[0]['kpts'][i]['x']
                    data_dic['kpts1'][1][i] = data_pair[0]['kpts'][i]['y']
                
                for i in range(10):
                    data_dic['kpts2'][0][i] = data_pair[1]['kpts'][i]['x']
                    data_dic['kpts2'][1][i] = data_pair[1]['kpts'][i]['y']
                
                data_dic['kpts1'] = jt.array(data_dic['kpts1'])
                data_dic['kpts2'] = jt.array(data_dic['kpts2'])
                    
                data_dic['A1'] = delaunay_triangulation(data_dic['kpts1'])
                data_dic['A2'] = delaunay_triangulation(data_dic['kpts2'])
                return data_dic, perm_mat
                # return data_pair, perm_mat

            else:
                index = index-len(self.ids[cls])
                continue
            
        return data_dic, perm_mat
    
        # print(id_pair)
        
    def Evaluate(self, pred_dict_list, classes):
        
        assert self.mode == 'test',  'Use test mode to evaluate.'
            
        return self.benchmark.eval(pred_dict_list, classes)
    
    def __len__(self):
        return self.length
    
    def count_loader_num(self):
        return self.benchmark.compute_length()

    def remove_cache(self, last_epoch=False):
        if self.mode == 'test':
            self.benchmark.rm_gt_cache(last_epoch=last_epoch)
    

if __name__ =='__main__':
    
    
    # test_loader = GMdataset(mode=mode, batch_size=2)
    # cnt_list, sum = test_loader.count_loader_num()
    mode = 'train'
    benchmark = pygm.benchmark.Benchmark(name='WillowObject', sets=mode, obj_resize=(256, 256), problem='2GM')
    cnt_list = benchmark.compute_length()
    # import pygmtools.dataset_config as config
    # config.__C.Willo
    print(cnt_list)
    # print(sum)