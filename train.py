import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train  

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
    parser.add_argument('-e', '--num-epochs', type=int, default=60, help='Number of epochs') 
    parser.add_argument('-l', '--lr', type=float, default=0.0005, help='Learning rate') 
    parser.add_argument('--lam', type=float, default = 0.95, help='exponential')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_NaFNet_2', help='Name of network')
    parser.add_argument('-kt', '--kspace-data-path-train', type=Path, default='/home/Data/train/kspace', help='Directory of train data')
    parser.add_argument('-it', '--image-data-path-train', type=Path, default='/home/NewData/train/image', help='Directory of train data')
    parser.add_argument('-kv', '--kspace-data-path-val', type=Path, default='/home/Data/val/kspace', help='Directory of validation data')
    parser.add_argument('-iv', '--image-data-path-val', type=Path, default='/home/NewData/val/image', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=6, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=12, help='Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens_chans', type=int, default=5, help='Number of channels for sensitivity map U-Net | 8 in original varnet')
    parser.add_argument('--sens_pools', type=int, default=4, help='Number of downsampling and upsampling layers for sensitivity map U-Net. | 4 in original varnet')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--edge_weight', type=float, default=1, help='weight of the edge loss.') 

    parser.add_argument('-m', '--model-mode', type=str, default='NaFNet', help='VarNet, NaFNet, EAMRI or EANaF') 
    parser.add_argument('--grappa', type = int, default=1, help='Grappa usage')
    parser.add_argument('--kspace_size_h', type = int, default=768, help='h of kspace')
    parser.add_argument('--kspace_size_w', type = int, default=396, help='w of kspace') 
    parser.add_argument('--coil', type = int, default=20, help='number of coil') 
    parser.add_argument('--image_mask', type = int, default=0, help='image mask usage') 

    parser.add_argument('--attdim', type = int, default = 4, help = 'attention dim')
    parser.add_argument('--num_head', type = int, default = 4, help = 'the number of head')
    parser.add_argument('--MSRB', type = int, default = 3, help = 'n_MSRB')

    parser.add_argument('--width', type = int, default = 18, help = 'nafnet chan')

    parser.add_argument('--pre', type = int, default = 0, help = 'pretrained model usage')
    parser.add_argument('--pre-name', type = Path, default = 'test_NaFNet_1', help = 'pretrained model path')

    parser.add_argument('--train-multiplier', type = int, default = 2, help = 'how many times the train data was generated')
    parser.add_argument('--val-multiplier', type = int, default = 2, help = 'how many times the validation data was generated')

    parser.add_argument('--stop-epoch', type = int, default = 1000, help = 'when to stop the code')

    parser = DataAugmentor.add_augmentation_specific_args(parser)
    args, unknown = parser.parse_known_args()
    args.aug_on = True
    args.aug_strength = 0.35
    return args

if __name__ == '__main__':

    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.pre_exp_dir = '../result' / args.pre_name / 'checkpoints'

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    
    train(args)