import numpy as np
import torch
import torchvision.transforms.functional as TF
#import cv2
from fastmri import fft2c, ifft2c, rss_complex
from utils.data.grappa import grappa_torch
from fastmri.data import transforms as T


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key): 
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, mask, input, target, isgrappa, grappa, H, W, coil, isimagemask, device, attrs, fname, slice, batch):

        if not self.isforward:
            target = to_tensor(target) 
            maximum = attrs[self.max_key]
            if isgrappa==1:
                grappa = to_tensor(grappa)
            
            if batch!= 1:
                w = mask.shape[0]
                p = int((W-w)/2)
                if p != 0:
                    index = np.where(mask==1)
                    a = index[0][0]
                    b = index[0][1]
                    h = b-a
                    mask = np.concatenate((mask[:h][-p:],mask,mask[-h:][:p]))
        else:
            target = -1
            maximum = -1

        kspace = to_tensor(input)

        if self.isforward and isgrappa==1:
            calib = T.center_crop(kspace, [384, 384]).to(device)
            kernel_size = (5, 5)
            my_grappa = grappa_torch(kspace, calib, kernel_size, coil_axis=0, device = device).to(device)
            real_part = my_grappa.real
            imag_part = my_grappa.imag
            final_grappa = torch.stack((real_part, imag_part), dim=-1).to(device)
            grappa = T.center_crop(rss_complex(ifft2c(final_grappa)), [384, 384]).to(device)
            grappa = grappa.cpu()


        if batch!=1 and not self.isforward:                                   

            h = kspace.shape[-2]
            w = kspace.shape[-1]
    
            if [h,w] != [H,W]:
              h_pad = int((H-h)/2)
              w_pad = int((W-w)/2)
              kspace = TF.pad(kspace,(w_pad, h_pad))


        kspace = torch.stack((kspace.real, kspace.imag), dim=-1) 

        if batch!=1 and not self.isforward:
            c = kspace.shape[0]
            if c != coil:
              h = kspace.shape[1]
              w = kspace.shape[2]
              size = (coil-c,h,w,2)
              eps=1e-10
              z = np.random.normal(0, eps, size=size)
              z = torch.from_numpy(z)
              empty_coil = fft2c(z)
              kspace = torch.cat((kspace, empty_coil), dim=0)
        
        mask = torch.from_numpy(mask)

        return mask, kspace, target, grappa, maximum, fname, slice 
