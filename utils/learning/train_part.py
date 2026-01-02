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
from utils.model.varnet import VarNetSen 
from utils.model.stack_model import StackModel
from utils.model.eastack_model import EAStackModel
from utils.model.eanaf_model import EANaFModel

from utils.data.sobel import *
from utils.model.fastmri.math import tensor_to_complex_np
from fastmri import fft2c, ifft2c, rss

from utils.mraugment.data_augment import DataAugmentor

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
        
        # edge = Normalize_sobel(target) 

        kspace = kspace.to(torch.float32)
        grappa = grappa.to(torch.float32)
        target = target.to(torch.float32)

        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) 
        grappa = grappa.cuda(non_blocking=True)
        # edge = edge.cuda(non_blocking=True) 
        maximum = maximum.cuda(non_blocking=True) 

        if args.model_mode == 'VarNet': #or args.model_mode == 'EAMRI':
          output = model(kspace, mask)
        else:
          output = model(kspace, mask, grappa)

        # if args.model_mode == 'EANaF' or args.model_mode == 'EAMRI': 
        #   loss = 0.
        #   edge_loss = 0.
        #   im_loss = 0.
        #   for ii, ele in enumerate(output):
        #     # ele= dataFormat(ele)
        #     # edge model
        #     if ii < len(output) - 1: # edge
        #       edge_loss += loss_type(ele, edge, maximum)
        #     else: # image
        #       im_loss += loss_type(ele, target, maximum)
        #   loss = (args.edge_weight * edge_loss + im_loss) # edge_weight = 0.25로 설정...
        # else:
        loss = loss_type(output, target, maximum)

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g}'
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

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, grappa, maximum, fnames, slices = data

            kspace = tensor_to_complex_np(kspace)
        
            kspace = torch.from_numpy(kspace * mask.view(mask.shape[0], 1, 1, mask.shape[1]).numpy())
            mask = mask.numpy()
            
            kspace = torch.stack((kspace.real, kspace.imag), dim=-1).unsqueeze(0)
            mask = torch.from_numpy(mask.reshape(kspace.shape[1], 1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
            kspace = kspace.squeeze(0)

            edge = Normalize_sobel(target)

            kspace = kspace.to(torch.float32)
            grappa = grappa.to(torch.float32)
            
            kspace = kspace.cuda(non_blocking=True) 
            mask = mask.cuda(non_blocking=True) 
            grappa = grappa.cuda(non_blocking=True) 


            if args.model_mode == 'VarNet': # or args.model_mode == 'EAMRI':
              output = model(kspace, mask) 
            else:
              output = model(kspace, mask, grappa)

            # if args.model_mode == 'EANaF' or args.model_mode == 'EAMRI':
            #   output = output[len(output)-1]

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


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


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


        
def train(args):

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
        
    # elif args.model_mode == 'EAMRI':
    #   model = EAStackModel(
    #                 sens_chans = args.sens_chans,
    #                 sens_pools = args.sens_pools,
    #                 attdim = args.attdim,
    #                 num_head = args.num_head,
    #                 n_MSRB = args.MSRB)

    # elif args.model_mode == 'EANaF':
    #   model = EANaFModel(
    #                 sens_chans = args.sens_chans,
    #                 sens_pools = args.sens_pools,
    #                 isgrappa = args.grappa,
    #                 attdim = args.attdim,
    #                 num_head = args.num_head,
    #                 n_MSRB = args.MSRB)
    

    model.to(device=device) 

    best_val_loss = 1.
    start_epoch = 0
    pre_start_epoch = 0
    
    if args.pre == 1:
        checkpoint = torch.load(args.pre_exp_dir / 'best_model.pt', map_location='cpu')
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        model.load_state_dict(checkpoint['model'])
        best_val_loss = checkpoint['best_val_loss'].item()
        pre_start_epoch = checkpoint['epoch']

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
                                lr_lambda=lambda epoch: args.lam ** epoch, 
                                last_epoch=-1,
                                verbose=False)

    if args.pre == 1:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: args.lam ** epoch, 
                                        last_epoch=pre_start_epoch,
                                        verbose=False)

    max_grad_norm = 5.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    train_loader = create_data_loaders(args = args, kspace_data_path = args.kspace_data_path_train, image_data_path = args.image_data_path_train, shuffle=True)
    val_loader = create_data_loaders(args = args, kspace_data_path = args.kspace_data_path_val, image_data_path = args.image_data_path_val)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        #if epoch == args.lr
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        scheduler.step()
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True) 
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True) 
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True) 

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )