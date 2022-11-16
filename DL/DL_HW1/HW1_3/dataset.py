"""
    filename : dataset.py
    Author : Ma Yichuan
    Description : 实现切分处理后的数据集， 并提供了评测函数
    Referrence : paddle Cifar10 源码
    conda env: DL on server
"""

import numpy as np
import os
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, normalize
from paddle.io import Dataset
import paddle
import math
import pickle


MODE_FLAG_MAP = {
    'train10': 'data_batch',
    'test10': 'test_batch',
    'train100': 'train',
    'test100': 'test'
}


def Open_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
    
class Cifar_Cut(Dataset):
    
    def __init__(self, Type, transform = None, data_path = None):
        super(Cifar_Cut, self).__init__()
        self.dtype = paddle.get_default_dtype()
        if transform == None:
            transform = Compose([Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), Transpose()]) 
            
        self.transform = transform
        self.data_path = data_path

        if Type == 'Train':
            self.load_train_data()
        elif Type == 'Test':
            self.load_test_data()
        else:
            raise  ValueError('Input type wrong: Train or Test')

    def Random_Cut(self, img):
        
        """
        
        Randomly cut the image into 4 pieces , return the result and labels
        
        Inputs: 
            img :  (3, 32, 32)
        Returns:
            Cut_pictures, Label : (4, 3, 16, 16) ; (4, 4)
            
        """
        
        shape = img.shape  # (3, 32, 32)
        threshold_X = math.floor(shape[1] / 2 )
        threshold_Y = math.floor(shape[2] / 2 )
        # (16, 16)
        
        """
            embed the cut position with 0, 1, 2, 3
        """
        
        img_cut = []
        img_cut.append( img[:, :threshold_X, :threshold_Y ] ) 
        img_cut.append( img[:, :threshold_X, threshold_Y: ] ) 
        img_cut.append( img[:, threshold_X:, :threshold_Y ] ) 
        img_cut.append( img[:, threshold_X:, threshold_Y: ] ) 

        label_Matrix = np.diag([1,1,1,1])
        # print(label)         

        np.random.shuffle(label_Matrix)
        cut_imgs = np.zeros([4, shape[0], threshold_X, threshold_Y])
        
        label = np.zeros([4])
        for i in range(4):
            cut_imgs[i] = img_cut[np.argmax(label_Matrix[i])]
            label[i] = np.argmax(label_Matrix[i])

        return cut_imgs, label


    def load_train_data(self):
        self.data = []
        for batch_idx in range(1, 6):
            batch = Open_batch(os.path.join(self.data_path,'data_batch_'+str(batch_idx)))
            for idx in range(0, len(batch[b'labels'])):
                img = batch[b'data'][idx] 
                img = np.reshape(img, [3, 32, 32])
                img = img.transpose([1, 2, 0])  # img.shape (32, 32, 3)
                if self.transform is not None:
                    img = self.transform(img)
                
                imgs, label = self.Random_Cut(img)
                self.data.append((imgs, label))

    def load_test_data(self):
        self.data = []
        batch = Open_batch(os.path.join(self.data_path,'test_batch')) # only one batch 0.0
        for idx in range(0, len(batch[b'labels'])):
            img = batch[b'data'][idx]
            img = np.reshape(img, [3, 32, 32])
            img = img.transpose([1, 2, 0])  # img.shape (32, 32, 3)
            
            if self.transform is not None:
                img = self.transform(img)
            
            imgs, label = self.Random_Cut(img)
            self.data.append((imgs, label))

    def __getitem__(self, idx):
        image, label = self.data[idx] 
        return image.astype('float32'), np.array(label).astype('int64')
    
    def __len__(self):
        return len(self.data)


def Evaluate(model, test_loader, best_acc):
    """_summary_

    Args:
        model: Tmp Perm model
        test_loader : (batch, (cut_imgs, label))
        best_acc (float32): The best accuracy of the model

    Returns:
        beat_acc
    """
    print('evaluating...')
    model.eval()
    accuracies = []
    result = [0, 0]
    acc_idx = 0
    
    for batch_idx, (img, label) in enumerate(test_loader()):
        # print(img.shape)
        if img.shape[0] < 256 :
            break
        output = model(img)
        
        for i in range(256): # 每一组切分图片
            
            label_i  = label[i]    # 4*1
            output_i = output[i]   # 4*4
            
            result[1] = result[1] + 4 
            for j in range(4):
                # print('predict', np.argmax(output_i[j]))
                # print('truth', label_i[j])
                if np.argmax(output_i[j]) == label_i[j]:
                    result[0] = result[0] + 1
            
               
            # result[0] = result[0] + 1
            # result[1] = result[1] + 1
        
        accuracies.append( result[0] / result[1] )
        
        # if accuracies[batch_idx] > best_acc :
        #     best_acc = accuracies[batch_idx]
        
        acc_idx = batch_idx
        if (batch_idx+1) % 5 == 0 :
            print('batch:[{}/{} ({:.0f}%)]\tAccuracy: {:.6f}'.format(
                    batch_idx+1, int(len(test_loader)),
                    100. * batch_idx / float(len(test_loader)), accuracies[batch_idx]))
    
    if accuracies[acc_idx] > best_acc : 
        best_acc = accuracies[acc_idx]    
           
    print('Best Accuracy: {:.6f}'.format(best_acc))        
    print('End evaluation, returning to training function.')      
    model.train()
    
    return best_acc