import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model.sensitivity import SensitivityModel

from eamri import EAMRI


class EAStackModel(nn.Module):
  def __init__(
        self,
        sens_chans: int = 8,
        sens_pools: int = 4,
        attdim: int = 4,
        num_head: int = 4,
        n_MSRB: int = 3
        ):

    super().__init__()

    self.eamri = EAMRI(sens_chans = sens_chans, sens_pools = sens_pools, attdim = attdim, num_head = num_head, n_MSRB = n_MSRB)

  def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa = torch.Tensor) -> torch.Tensor: 

    ea_result = self.eamri(grappa, masked_kspace, mask)

    height = ea_result[0].shape[-2]
    width = ea_result[0].shape[-1]

    for i in range(len(ea_result)):
        ea_result[i] = ea_result[i][:, :, (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
        if i < len(ea_result) - 1:
            ea_result[i] = ea_result[i].permute(0, 2, 3, 1)
        elif i == len(ea_result) - 1:
            ea_result[i] = ea_result[i].permute(0, 2, 3, 1) 
    
    return ea_result

