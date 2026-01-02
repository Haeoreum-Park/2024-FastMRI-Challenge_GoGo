import argparse
import torch
import os
import h5py
import re
import sys

import numpy as np
import torchvision.transforms as transforms
from fastmri.data import transforms as T 
from fastmri import fft2c, ifft2c, rss_complex, complex_abs 
from pathlib import Path
from tempfile import NamedTemporaryFile as NTF
from skimage.util import view_as_windows

sys.path.insert(1, os.getcwd())
if os.getcwd() + '/utils/data/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/data/')
from grappa import grappa_torch


def parse():
    parser = argparse.ArgumentParser(description='Recreate Data on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-tn', '--train-num', type=int, default=2, help='multiplier for train data, 2 or 3')
    parser.add_argument('-vn', '--val-num', type=int, default=2, help='multiplier for val data, 2 or 3')
    parser.add_argument('-tp', '--train-path', type=Path, default='home/Data/train', help='path of original train data')
    parser.add_argument('-vp', '--val-path', type=Path, default='home/Data/val', help='path of original val data')
    parser.add_argument('-ntp', '--newtrain-path', type=Path, default='home/NewData/train/image', help='path for new train image data')
    parser.add_argument('-nvp', '--newval-path', type=Path, default='home/NewData/val/image', help='path for new val image data')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()

    train = '../../' / args.newtrain_path
    val = '../../' / args.newval_path
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)
    
    mask = np.zeros((7, 396), dtype = int)
    
    for i in range(4, 11):
      mask[(i-4)][(i-2)::i] = 1
      mask[(i-4)][182 : (182 + 32)] = 1
    mask = torch.from_numpy(mask)
    
    os.chdir("/")
    root = args.train_path
    
    image_files = list(Path(root / "image").iterdir())
    kspace_files = list(Path(root / "kspace").iterdir())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    
    s_kspace = sorted(kspace_files)
    s_image = sorted(image_files)
    
    for file_ind in range(len(kspace_files)):
    
        print(f"file index: {file_ind}")
    
        fname = s_kspace[file_ind]
    
        image_list = [[] for _ in range(args.train_num)] ##
        grappa_list = [[] for _ in range(args.train_num)] ##
    
        with h5py.File(s_image[file_ind], 'r') as f:
          real_image = torch.from_numpy(f['image_label'][:]).to(device)
          my_attrs = dict(f.attrs)
    
        with h5py.File(fname, 'r') as f:
    
            match = re.search(r'acc(\d+)_(\d+)', fname.name)
            acc_number = int(match.group(1))
            ran_index = int(match.group(2))
    
            sort_list = [i for i in range(4, 11) if i != acc_number]
            if args.train_num == 2:
                d = ran_index % 6
                result = (acc_number, sort_list[d]) 
            elif args.train_num == 3:
                d = ran_index % 3
                result = (acc_number, sort_list[d], sort_list[d+3]) 
    
            kspace = torch.from_numpy(f['kspace'][:]).to(device)
            
            kernel_size = (5, 5)
    
            for i in range(args.train_num): 
                mask_ind = result[i]
    
                if kspace.shape[3] == 392:
                    my_mask = mask[mask_ind-4][2:394]
                elif kspace.shape[3] == 396:
                    my_mask = mask[mask_ind-4]
                
                unkspace = kspace * my_mask.to(device)
    
                calib = T.center_crop(unkspace, [384, 384]).to(device)
    
                real_part = unkspace.real
                imag_part = unkspace.imag
                final_kspace = torch.stack((real_part, imag_part), dim=-1).to(device)
    
                image_inner = torch.zeros((final_kspace.shape[0], 384, 384), device=device)
                grappa_inner = torch.zeros((final_kspace.shape[0], 384, 384), device=device)
    
                #start = time.time()
                for j in range(final_kspace.shape[0]):
    
                    my_slice = final_kspace[j].to(device)
                    image_inner[j] = T.center_crop(rss_complex(ifft2c(my_slice)), [384, 384]).to(device)
    
    
                    my_grappa = grappa_torch(unkspace[j], calib[j], kernel_size, coil_axis=0, device = device).to(device)
    
                    real_part = my_grappa.real
                    imag_part = my_grappa.imag
                    final_grappa = torch.stack((real_part, imag_part), dim=-1).to(device)
    
                    grappa_inner[j] = T.center_crop(rss_complex(ifft2c(final_grappa)), [384, 384]).to(device)

                image_list[i] = image_inner
                grappa_list[i] = grappa_inner
    
    
        for acc in result:
            
            my_image = image_list[result.index(acc)]
            my_grappa = grappa_list[result.index(acc)]
            
            if kspace.shape[3] == 392:
              my_mask = mask[acc-4][2:394]
            elif kspace.shape[3] == 396:
              my_mask = mask[acc-4]
                
            wname = f"brain_acc{acc_number}_{ran_index}_{acc}.h5"
        
            save_path = args.newtrain_path
            full_path = os.path.join(save_path, wname)
            with h5py.File(full_path, 'w') as f:
              f.create_dataset('image_input', data = my_image.cpu().numpy())
              f.create_dataset('image_grappa', data = my_grappa.cpu().numpy())
              f.create_dataset('image_label', data = real_image.cpu().numpy())
              f.create_dataset('mask', data = my_mask.cpu().numpy())
              f.attrs['max'] = my_attrs['max']
              f.attrs['norm'] = my_attrs['norm']
    
        for obj in dir():
            if isinstance(eval(obj), torch.Tensor) and eval(obj).is_cuda:
                del globals()[obj]
        
        torch.cuda.empty_cache()
    
    os.chdir("/")
    root = args.val_path
    
    image_files = list(Path(root / "image").iterdir())
    kspace_files = list(Path(root / "kspace").iterdir())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    
    s_kspace = sorted(kspace_files)
    s_image = sorted(image_files)
    
    for file_ind in range(len(kspace_files)):
    
        print(f"file index: {file_ind}")
    
        fname = s_kspace[file_ind]
    
        image_list = [[] for _ in range(args.val_num)]
        grappa_list = [[] for _ in range(args.val_num)]
    
        with h5py.File(s_image[file_ind], 'r') as f:
          real_image = torch.from_numpy(f['image_label'][:]).to(device)
          my_attrs = dict(f.attrs)
    
        with h5py.File(fname, 'r') as f:
    
            match = re.search(r'acc(\d+)_(\d+)', fname.name)
            acc_number = int(match.group(1))
            ran_index = int(match.group(2))
    
            sort_list = [i for i in range(4, 11) if i != acc_number]

            if args.val_num == 2:
                d = ran_index % 6
                result = (acc_number, sort_list[d]) 
            elif args.val_num == 3:
                d = ran_index % 3
                result = (acc_number, sort_list[d], sort_list[d+3]) 
    
            kspace = torch.from_numpy(f['kspace'][:]).to(device)
            
            kernel_size = (5, 5)
    
            for i in range(args.val_num):
                mask_ind = result[i]
    
                if kspace.shape[3] == 392:
                    my_mask = mask[mask_ind-4][2:394]
                elif kspace.shape[3] == 396:
                    my_mask = mask[mask_ind-4]
                
                unkspace = kspace * my_mask.to(device)
    
                calib = T.center_crop(unkspace, [384, 384]).to(device)
    
                real_part = unkspace.real
                imag_part = unkspace.imag
                final_kspace = torch.stack((real_part, imag_part), dim=-1).to(device)
    
                image_inner = torch.zeros((final_kspace.shape[0], 384, 384), device=device)
                grappa_inner = torch.zeros((final_kspace.shape[0], 384, 384), device=device)
    
                #start = time.time()
                for j in range(final_kspace.shape[0]):
    
                    my_slice = final_kspace[j].to(device)
                    image_inner[j] = T.center_crop(rss_complex(ifft2c(my_slice)), [384, 384]).to(device)
    
    
                    my_grappa = grappa_torch(unkspace[j], calib[j], kernel_size, coil_axis=0, device = device).to(device)
    
                    real_part = my_grappa.real
                    imag_part = my_grappa.imag
                    final_grappa = torch.stack((real_part, imag_part), dim=-1).to(device)
    
                    grappa_inner[j] = T.center_crop(rss_complex(ifft2c(final_grappa)), [384, 384]).to(device)

                image_list[i] = image_inner
                grappa_list[i] = grappa_inner
    
    
        for acc in result:
            
            my_image = image_list[result.index(acc)]
            my_grappa = grappa_list[result.index(acc)]
            
            if kspace.shape[3] == 392:
              my_mask = mask[acc-4][2:394]
            elif kspace.shape[3] == 396:
              my_mask = mask[acc-4]
        
            wname = f"brain_acc{acc_number}_{ran_index}_{acc}.h5"
        
            save_path = args.newval_path
            full_path = os.path.join(save_path, wname)
            with h5py.File(full_path, 'w') as f:
              f.create_dataset('image_input', data = my_image.cpu().numpy())
              f.create_dataset('image_grappa', data = my_grappa.cpu().numpy())
              f.create_dataset('image_label', data = real_image.cpu().numpy())
              f.create_dataset('mask', data = my_mask.cpu().numpy())
              f.attrs['max'] = my_attrs['max']
              f.attrs['norm'] = my_attrs['norm']
    
        for obj in dir():
            if isinstance(eval(obj), torch.Tensor) and eval(obj).is_cuda:
                del globals()[obj]
        
        torch.cuda.empty_cache()