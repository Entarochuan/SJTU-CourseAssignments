"""
    filename : Args.py
    Author : Ma Yichuan
    Description : 提供训练参数接口
    Referrence : 
    conda env: DL on server
"""

import argparse
import sys


def init_args(params=sys.argv[1:]):
    
    arg_parser = argparse.ArgumentParser(
                    prog = 'Stitching Net',
                    description = 'Cut pictures and train a Net to stitch it back.')
    
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    
    print('Args loaded, start training.')
    
    return opt


def add_argument_base(arg_parser):
    
    arg_parser.add_argument('--max_epoch', type=int, default=20, help='Type in the max epoch')
    arg_parser.add_argument('--batch_size', type=int, default=256, help='Data batch_size')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    arg_parser.add_argument('--data_path', type=str, default='./cifar/cifar-10-batches-py', help='Path of data')
    arg_parser.add_argument('--hidden_size', type=int, default=256, help='CNN hidden size')
    
    return arg_parser