"""
    filename : Args.py
    Author : Atanis
    Description : 提供训练参数接口
    Reference : 
    conda env: ycm 
"""

import argparse
import sys


def init_args(params=sys.argv[1:]):
    
    arg_parser = argparse.ArgumentParser(
                    prog = 'Graph_Matching Net',
                    description = 'Accomplished a graph-matching net.')
    
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    
    print('Args loaded, start training.')
    
    return opt


def add_argument_base(arg_parser):
    
    arg_parser.add_argument('--max_epoch', type=int, default=15, help='Type in the max epoch')
    arg_parser.add_argument('--batch_size', type=int, default=4, help='Data batch_size, default=16.')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--lr', type=float, default=0.0000015, help='learning rate')
    arg_parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='SGD', help='Choose the optimizer.')
    arg_parser.add_argument('--cnn_type', choices=['vgg16_bn', 'vgg16', 'Resnet18', 'densenet121'], default='vgg16_bn', help='Choose the cnn block.')
    arg_parser.add_argument('--SGD_settings', type=list, default=[0.9, 1e-8], help='Settings for SGD, [momentum:float, weight_decay:float]')
    
    return arg_parser