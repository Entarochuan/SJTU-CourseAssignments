"""

    filename : main.py
    author : Yichuan Ma
    Date : 2022/11/12
    Description : define  main arguments and train
    Reference :  Spikingjelly Document
    
"""

import Train_funcs
import sys
import argparse


def init_args(params=sys.argv[1:]):
    
    arg_parser = argparse.ArgumentParser(
                    prog = 'SNN_CNN Compare',
                    description = 'Compare between SNN&CNN')
    
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    
    print('Args loaded, start training.')
    
    return opt


def add_argument_base(arg_parser):
    
    arg_parser.add_argument('--max_epoch', type=int, default=20, help='Type in the max epoch')
    arg_parser.add_argument('--batch_size', type=int, default=32, help='Data batch_size')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    arg_parser.add_argument('--choose_net', default='CNN', choices=['SNN', 'CNN'], help='choose a net')
    
    return arg_parser


def main():
    
    args = init_args(sys.argv[1:])
    
    device = args.device
    batch_size = args.batch_size
    lr = args.lr
    max_epoch = args.max_epoch
    
    # 下面的参数是为SNN定义的
    T = int(8)
    tau = float(2.0)
    
    if args.choose_net == 'SNN':
        Train_funcs.train_SNN(max_epoch, lr, batch_size, device)
    elif args.choose_net == 'CNN':
        Train_funcs.train_CNN(max_epoch, lr, batch_size, device)
    else:
        print('Training params error,choose from SNN/CNN.')
        
        
if __name__ == '__main__':
    main()