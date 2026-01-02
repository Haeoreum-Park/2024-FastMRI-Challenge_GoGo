import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import re

def char_priority(c):
    if c == '_':
        return 0
    elif c.isdigit(): 
        return int(c)+1
    else:
        return 100

def custom_sort_key(s):
    s = str(s)
    return [char_priority(c) for c in s]

class SliceData(Dataset):
    def __init__(self, kspace_root, image_root, transform, input_key, target_key, isgrappa, device, H=None, W=None, coil=None, isimagemask=None, forward=False, v=1, batch=1):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.v = v
        self.device = device
        self.coil = coil
        self.H = H
        self.W = W
        self.isimagemask = isimagemask
        self.isgrappa = isgrappa
        self.batch = batch
        
        if not forward:
            image_files = list(Path(image_root).iterdir())
            

            for fname in sorted(image_files, key = custom_sort_key):
                
                num_slices = self._get_metadata(Path(fname))
        
                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(kspace_root).iterdir())
        kspace_files = kspace_files * v
        
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
                
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):

        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if self.forward:
                mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
            grappa = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                mask =  np.array(hf["mask"])
                if self.isgrappa == 1:
                    grappa = hf['image_grappa'][dataslice]
                else:
                    grappa = -1
                attrs = dict(hf.attrs)
        
        coil = self.coil
        isimagemask = self.isimagemask
        isgrappa = self.isgrappa
        device = self.device
        H = self.H
        W = self.W
        batch = self.batch
            
        return self.transform(mask, input, target, isgrappa, grappa, H, W, coil, isimagemask, device, attrs, kspace_fname.name, dataslice, batch)


def dataFormat(x): # eamri code 
    if len(x.shape) == 3:
        return x 
    elif len(x.shape) == 5: #(B, C, H, W, 2)
        assert x.shape[-1] == 2, "dataFormat last dimension should be 2"
        x = (x**2).sum(dim=-1).sqrt() #(B,-1, H, W)
        x = ((x**2).sum(dim=1)).sqrt() #(B,H,W)
    elif len(x.shape) == 4:
        if x.shape[1] == 1: #(B,1,H,W)
            x = x.squeeze(1)
        elif x.shape[1] == 2: #single coigl (B,2,H,W)
            x = (x**2).sum(dim=1).sqrt()
        else: #(B,C,H,W)
            assert len(x.shape) == 4, "dataFormat do not support dynamic MRI"
            B, C, H, W = x.shape
            x = x.reshape(B,-1,H,W,2)
            x = (x**2).sum(dim=-1).sqrt() #(B,-1, H, W)
            x = ((x**2).sum(dim=1)).sqrt() #(B,H,W)

    return x


def create_data_loaders(args, kspace_data_path, image_data_path=None, device=None, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key

        v = 1
        if 'train' in str(kspace_data_path):
            v = args.train_multiplier
        if 'val' in str(kspace_data_path):
            v = args.val_multiplier
        data_storage = SliceData(
            kspace_root=kspace_data_path,
            image_root=image_data_path,
            transform=DataTransform(isforward, max_key_),
            input_key=args.input_key,
            target_key=target_key_,
            forward = isforward,
            isgrappa = args.grappa,
            device = device,
            H = args.kspace_size_h,
            W = args.kspace_size_w,
            coil = args.coil,
            isimagemask = args.image_mask,
            v = v,
            batch = args.batch_size
        )
    else: 
        max_key_ = -1
        target_key_ = -1
        data_storage = SliceData( 
            kspace_root=kspace_data_path,
            image_root=image_data_path,
            transform=DataTransform(isforward, max_key_),
            input_key=args.input_key,
            target_key=target_key_,
            forward = isforward,
            isgrappa = args.grappa,
            device = device,
            batch = args.batch_size
        )
    

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    
    return data_loader
