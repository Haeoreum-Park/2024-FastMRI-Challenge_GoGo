import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy
import torch.optim as optim
from torch.nn import functional as F

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnetrdn import VarNetSen
from utils.model.stack_model import StackModel
from utils.model.eastack_model import EAStackModel
from utils.model.eanaf_model import EANaFModel

from utils.data.sobel import *
from utils.model.fastmri.math import tensor_to_complex_np
from fastmri import fft2c, ifft2c, rss

from utils.mraugment.data_augment import DataAugmentor
import wandb
import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type): 
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    augmentor = DataAugmentor(args, epoch)

    for iter, data in enumerate(data_loader):

        mask, kspace, target, grappa, maximum, _, _ = data 
        
        kspace, target, grappa = augmentor(kspace, target, grappa, target_size = target.shape) 
        
        B, _, _ = target.shape 

        kspace = tensor_to_complex_np(kspace)

        kspace = torch.from_numpy(kspace * mask.view(mask.shape[0], 1, 1, mask.shape[1]).numpy())
        mask = mask.numpy()
        
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1).unsqueeze(0)

        mask = torch.from_numpy(mask.reshape(kspace.shape[1], 1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        kspace = kspace.squeeze(0)

        kspace = kspace.to(torch.float32)
        grappa = grappa.to(torch.float32)
        target = target.to(torch.float32)

        mask = mask.cuda(non_blocking=True) 
        kspace = kspace.cuda(non_blocking=True) 
        target = target.cuda(non_blocking=True) 
        grappa = grappa.cuda(non_blocking=True) 
        maximum = maximum.cuda(non_blocking=True) 
    
        if args.model_mode == 'VarNet': 
          output = model(kspace, mask)
        else:
          output = model(kspace, mask, grappa)

        loss = loss_type(output, target, maximum) 
        
        loss.backward()

        wandb.log({"train_loss_batch": loss})    

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    count = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            count += 1
            
            if count % 10 == 0:
                print(f"valid fin! {count}")
            mask, kspace, target, grappa, maximum, fnames, slices = data

            kspace = tensor_to_complex_np(kspace)
        
            kspace = torch.from_numpy(kspace * mask.view(mask.shape[0], 1, 1, mask.shape[1]).numpy())
            mask = mask.numpy()

            kspace = torch.stack((kspace.real, kspace.imag), dim=-1).unsqueeze(0)

            mask = torch.from_numpy(mask.reshape(kspace.shape[1], 1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
            kspace = kspace.squeeze(0)

            kspace = kspace.to(torch.float32)
            grappa = grappa.to(torch.float32)

            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)


            if args.model_mode == 'VarNet': 
              output = model(kspace, mask) 
            else:
              output = model(kspace, mask, grappa)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start

def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

def wandb_train(args):
    
    parameters = vars(args)
    sweep_config = dict()
    sweep_config['method'] = 'grid'
    sweep_config['metric'] = {'goal': 'minimize', 'name': 'val_acc'}
    sweep_config['parameters'] = parameters
    for key, value in sweep_config['parameters'].items():
        sweep_config['parameters'][key] = {'value': value}

    sweep_config['parameters'].update({'cascade': {'value': 6},
                                       'chans': {'value': 12},
                                       'sens_chans': {'value': 6},
                                       'sens_pools': {'value': 4},
                                       'aug_strength':{'value': 0.35},
                                       'aug_on': {'value': True},
                                       'lr': {'value': 0.0005},
                                       'lam': {'value': 0.95},
                                       'width': {'values': [18, 20, 22]},
                                       'model_mode': {'value': 'NaFNet'}})


    sweep_id = wandb.sweep(sweep_config, project = 'gomin_goneung_NaFNet')

    wandb.agent(sweep_id, train, count = 4) 

def func(epoch):
    if epoch < 10:
        return 0.93 ** epoch
    elif epoch < 20:
        return 0.94 ** 2
    elif epoch < 30:
        return 0.95 ** epoch
    else:
        return 0.96 ** epoch
        
def train(config = None):
    
    with wandb.init(config = config, name = "NaFNet_width_second"):
        
        args = wandb.config

        args['kspace_data_path_train'] = Path('/home/Data/train/kspace')
        args['image_data_path_train'] = Path('/home/NewData/train/image')
        args['kspace_data_path_val'] = Path('/home/Data/val/kspace')
        args['image_data_path_val'] = Path('/home/NewData/val/image')
        
        device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
        torch.cuda.set_device(device) 
        print('Current cuda device: ', torch.cuda.current_device()) 
    
        if args.model_mode == 'NaFNet': 
          model = StackModel(num_cascades = args.cascade, 
                        chans = args.chans,           
                        sens_chans = args.sens_chans,
                        sens_pools = args.sens_pools,
                        isgrappa = args.grappa)
        
        elif args.model_mode == 'VarNet':
          model = VarNetSen(num_cascades = args.cascade,
                        chans = args.chans,
                        sens_chans = args.sens_chans,
                        sens_pools = args.sens_pools)
    
        model.to(device=device) 
    
        """
        # using pretrained parameter
        VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
        MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
        if not Path(MODEL_FNAMES).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
        
        pretrained = torch.load(MODEL_FNAMES)
        pretrained_copy = copy.deepcopy(pretrained)
        for layer in pretrained_copy.keys():
            if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
                del pretrained[layer]
        model.load_state_dict(pretrained)
        """
    
        loss_type = SSIMLoss().to(device=device) 
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda = func,
                                        last_epoch=-1,
                                        verbose=False)
    
        max_grad_norm = 5. 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
        best_val_loss = 1.
        start_epoch = 0
    
        train_loader = create_data_loaders(args = args, kspace_data_path = args.kspace_data_path_train, image_data_path = args.image_data_path_train, shuffle=True)
        val_loader = create_data_loaders(args = args, kspace_data_path = args.kspace_data_path_val, image_data_path = args.image_data_path_val)
    
        val_loss_log = np.empty((0, 2))
        for epoch in range(start_epoch, args.num_epochs):
            print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
            
            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
            val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

            scheduler.step()
            
            val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)

            train_loss = torch.tensor(train_loss).cuda(non_blocking=True) 
            val_loss = torch.tensor(val_loss).cuda(non_blocking=True) 
            num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True) 
    
            val_loss = val_loss / num_subjects
    
            is_new_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            
            print(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})         
            if is_new_best:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                start = time.perf_counter()
                print(
                    f'ForwardTime = {time.perf_counter() - start:.4f}s',
                )
