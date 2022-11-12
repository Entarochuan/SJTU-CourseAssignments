"""

    filename : Train_and_eval.py
    author : Yichuan Ma
    Date : 2022/11/12
    Description : Train and evaluate the modeels
    Reference :  Spikingjelly Document
    
"""

from spikingjelly.datasets import n_mnist

root_dir = './data/n_mnist'
train_set = n_mnist(root_dir, train=True, use_frame=False)
test_set = n_mnist(root_dir, train=True, use_frame=False)
