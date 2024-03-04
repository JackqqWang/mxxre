import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()

    

    parser.add_argument('--T', default=1, help="1 TEMP")
    parser.add_argument('--test_pass', default=0, help="1 is we just use small model to test")

    parser.add_argument('--dataset', type=str, default='Fed_IXI', help="name \
                        of dataset, ChestXray, ISIC, ChestXray_binary, ChestXray_seg,Fed_IXI")
    parser.add_argument('--num_users', type=int, default=3,
                        help="number of users: K, ChestXray_seg:3, ChestXray_binary:6, ISIC:6")
    parser.add_argument('--num_classes', type=int, default=2, help="number \
                        of classes, x_ray:14")
    parser.add_argument('--large_model_is', default=1, help="1 is we only use large model for test")
    
    parser.add_argument('--small_model', type=str, default='ResNet20', help='model name')
    parser.add_argument('--large_model', type=str, default='ResNet110', help='model name')

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    
    parser.add_argument('--top_n', type=int, default=2, help='2 for x_ray_binary')
    parser.add_argument('--temperature', type=float, default=1,
                        help='T in KD') 
    
    parser.add_argument('--ours', type=int, default=1, help='1:large, 0 all small')
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')


    
    parser.add_argument('--communication_round', type=int, default=50,
                        help="number of communication rounds")
    parser.add_argument('--sfine', type=int, default=3,
                        help="server fine tune")

    
    parser.add_argument('--frac_large', type=float, default=1,
                        help='the fraction of large clients')    
    

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='weight of the teacher and student')


    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")
    parser.add_argument('--sv_batch_size', type=int, default=5000,
                        help="server batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    

    parser.add_argument('--num_shards', type=int, default=200,
                        help='number of shards')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")

    
    parser.add_argument('--metric_hyper', type=float, default=0.5, help="metric hyper")
    parser.add_argument('--cluster_num', type=int, default=2, help="number \
                        of cluster classes")


    parser.add_argument('--public_dataset', type=str, default='cifar10', help="name \
                        of public dataset")
    parser.add_argument('--input_size', type=int, default=10, help="cifar10: 3*32*32 = 3072, svhn: 3072, mnist: 7841") 

    parser.add_argument('--supervised', default=1, help="1 is supervised, 0 is unsueprvised")


    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to 1. Default set  0 to use CPU.")
    
    
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
