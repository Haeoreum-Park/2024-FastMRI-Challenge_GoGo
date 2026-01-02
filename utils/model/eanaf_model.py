import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model.sensitivity import SensitivityModel # new

from nafnet import NAFNet
from eamri import EAMRI


class EANaFModel(nn.Module):
  def __init__(
        self,
        sens_chans: int = 8,
        sens_pools: int = 4,
        isgrappa: int = 0,
        img_channel = 1, 
        attdim: int = 4,
        num_head: int = 4,
        n_MSRB: int = 3,
        width = 16, 
        middle_blk_num = 1, 
        enc_blk_nums = [1, 1, 1, 28], 
        dec_blk_nums = [1, 1, 1, 1]
        ):

    super().__init__()
    
    self.eamri = EAMRI(sens_chans = sens_chans, sens_pools = sens_pools, attdim = attdim, num_head = num_head, n_MSRB = n_MSRB)
      
    self.nafnet = NAFNet(img_channel=(isgrappa+1), 
                    width=width, 
                    middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blk_nums, 
                    dec_blk_nums=dec_blk_nums)
  def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa) -> torch.Tensor:

    aliased = fastmri.rss(fastmri.ifft2c(masked_kspace))

    ea_result = self.eamri(aliased, masked_kspace, mask)

    height = ea_result[0].shape[-2]
    width = ea_result[0].shape[-1]

    for i in range(len(ea_result)):
        ea_result[i] = ea_result[i][:, :, (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
        if i < len(ea_result) - 1:
            ea_result[i] = ea_result[i].squeeze(1)
        else:
            ea_result[i] = fastmri.complex_abs(ea_result[i].permute(0, 2, 3, 1))

    
    if torch.equal(grappa, torch.tensor([-1], device=grappa.device)):
        ea_result[len(ea_result)-1] = ea_result[len(ea_result)-1].unsqueeze(1)
    else:
        ea_result[len(ea_result)-1] = torch.stack([ea_result[len(ea_result)-1], grappa], dim = 1)

    ea_result[len(ea_result)-1] = self.nafnet(ea_result[len(ea_result)-1]).squeeze(1)

    return ea_result