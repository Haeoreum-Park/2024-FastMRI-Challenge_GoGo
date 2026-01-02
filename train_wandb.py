import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part_wandb import wandb_train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

if os.getcwd() + '/utils/mraugment/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/mraugment/')
from utils.mraugment.data_augment import DataAugmentor

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size') 
    parser.add_argument('-e', '--num-epochs', type=int, default=4, help='Number of epochs') 
    parser.add_argument('-l', '--lr', type=float, default = 0.001, help='Learning rate') 
    parser.add_argument('--lam', type=float, default = 0.95, help='exponential') 
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=str, default='test_varnet', help='Name of network')

    parser.add_argument('--cascade', type=int, default=4, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=18, help='Number of channels for cascade U-Net | 18 in original varnet') 
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') 
    parser.add_argument('--sens_pools', type=int, default=4, help='Number of downsampling and upsampling layers for sensitivity map U-Net. | 4 in original varnet') 
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--edge_weight', type=float, default=1.0, help='weight of the edge loss.') 

    parser.add_argument('-m', '--model-mode', type=str, default='VarNet', help='VarNet or NaFNet') 
    parser.add_argument('--grappa', type = int, default=1, help='Grappa usage') 
    parser.add_argument('--kspace_size_h', type = int, default=768, help='h of kspace') 
    parser.add_argument('--kspace_size_w', type = int, default=396, help='w of kspace')
    parser.add_argument('--coil', type = int, default=20, help='number of coil')
    parser.add_argument('--image_mask', type = int, default=0, help='image mask usage')
    parser.add_argument('--attdim', type = int, default = 4, help = 'attention dim') 
    parser.add_argument('--num_head', type = int, default = 4, help = 'the number of head')
    parser.add_argument('--MSRB', type = int, default = 3, help = 'n_MSRB')
    
    parser.add_argument('--width', type = int, default = 16, help = 'nafnet chan') 

    parser = DataAugmentor.add_augmentation_specific_args(parser)
    args, unknown = parser.parse_known_args()

    return args

if __name__ == '__main__':

    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    wandb_train(args)
