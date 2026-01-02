import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders

from utils.model.varnet import VarNetSen
from utils.model.stack_model import StackModel
from utils.model.eastack_model import EAStackModel
from pathlib import Path

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, grappa, _, fnames, slices) in data_loader:
            mask = mask.numpy()
            mask = torch.from_numpy(mask.reshape(args.batch_size, 1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
            
            kspace = kspace.cuda(non_blocking=True) 
            mask = mask.cuda(non_blocking=True) 
            grappa = grappa.cuda(non_blocking=True) 

            if args.model_mode == 'VarNet':
              output = model(kspace, mask)
            else:
              output = model(kspace, mask, grappa)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) 
    print ('Current cuda device ', torch.cuda.current_device()) 

    if args.model_mode == 'NaFNet': 
      model = StackModel(num_cascades = args.cascade, 
                    chans = args.chans,           
                    sens_chans = args.sens_chans,
                    sens_pools = args.sens_pools,
                    isgrappa = args.grappa)        
    elif args.model_mode == 'VarNet':
      model = VarNetSen(num_cascades = args.cascade,
                    chans = args.chans,
                    sens_chans = args.sens_chans)

    model.to(device=device) 
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(args = args, kspace_data_path = Path(args.data_path / "kspace"), device=device, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
    