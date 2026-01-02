import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from utils.model.sensitivity import SensitivityModel # new

from varnet import VarNet 
from nafnet import NAFNet


class StackModel(nn.Module):
  def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        isgrappa: int = 0,
        # img_channel = 2, # ???
        width = 16,
        middle_blk_num = 1, 
        enc_blk_nums = [1, 1, 1, 28], 
        dec_blk_nums = [1, 1, 1, 1]
        ):

    super().__init__()
    self.varnet = VarNet(num_cascades = num_cascades, 
                   chans = chans)
    self.nafnet = NAFNet(img_channel=(isgrappa+1), 
                    width=width, 
                    middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blk_nums, 
                    dec_blk_nums=dec_blk_nums)

    self.sens_net = SensitivityModel(sens_chans, sens_pools)

  def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa) -> torch.Tensor: 
    sens_maps = self.sens_net(masked_kspace, mask)
    
    var_result = self.varnet(masked_kspace, mask, sens_maps) 

    var_result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(var_result)), dim=1)

    height = var_result.shape[-2]
    width = var_result.shape[-1]
    var_result = var_result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]

    if torch.equal(grappa, torch.tensor([-1], device=grappa.device)):
      var_result = var_result.unsqueeze(1)
    else:
      var_result = torch.stack([var_result, grappa], dim = 1)

    naf_result = self.nafnet(var_result).squeeze(1)
      
    return naf_result