import cv2
import numpy as np
import torch

import os
import torch.nn as nn
import torch.nn.functional as F


def Get_sobel(target):
    
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(target, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(target, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def Normalize_sobel(target): 

    C, H, W = target.shape 
    edge = torch.zeros((C, H, W), dtype=torch.float32)

    target = target.numpy()

    for c in range(C):
      target_batch = target[c]
      if target_batch.max().item() - target_batch.min().item() != 0.0:
          target_normal = (255*(target_batch - target_batch.min()))/(target_batch.max().item() - target_batch.min().item())
          edge_batch = torch.from_numpy(Get_sobel(target_normal)) 
          edge_batch = edge_batch / edge_batch.max() / 255
          edge[c] = edge_batch * (target_batch.max().item() - target_batch.min().item()) + target_batch.min()
    
    return edge
