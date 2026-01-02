import torch
import numpy as np


def view_as_windows_torch(image, shape, stride=None):
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)
    elif len(shape) != 3:
        raise ValueError("Shape must be a tuple of length 2 or 3")

    if stride is None:
        stride = (1, 1, 1)

    windows = image.unfold(2, shape[0], stride[0])
    windows = windows.unfold(3, shape[1], stride[1])
    windows = windows.unfold(4, shape[2], stride[2])

    return windows

def unravel_index_torch(indices, shape):
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(coord))

def atleast_1d_torch(tensor):
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    return tensor


def grappa_torch(kspace, calib, kernel_size=(5, 5), coil_axis=-1, device = 'cpu', lamda=0.01):
    kspace = kspace.to(device)
    calib = calib.to(device)

    kspace = torch.moveaxis(kspace, coil_axis, -1)
    calib = torch.moveaxis(calib, coil_axis, -1)

    if torch.sum((torch.abs(kspace[..., 0]) == 0).flatten()) == 0:
        return torch.moveaxis(kspace, -1, coil_axis).cpu().numpy()

    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx / 2), int(ky / 2)
    nc = calib.shape[-1]
    adjx = kx % 2
    adjy = ky % 2

    kspace = torch.nn.functional.pad(kspace, (0, 0, ky2, ky2, kx2, kx2))
    calib = torch.nn.functional.pad(calib, (0, 0, ky2, ky2, kx2, kx2))
    mask = torch.abs(kspace[..., 0]) > 0

    P = view_as_windows_torch(mask.reshape((1, 1, mask.shape[0], mask.shape[1])), (kx, ky)).to(device)
    Psh = P.shape[2:6]
    P = P.reshape(-1, kx, ky)
    P, iidx = torch.unique(P, return_inverse=True, dim=0)

    validP = torch.nonzero(~P[:, kx2, ky2], as_tuple=True)[0]
    invalidP = torch.nonzero(torch.all(P== 0, dim=(1, 2)), as_tuple=True)[0]
    validP = validP[~torch.isin(validP, invalidP)]

    P = torch.tile(P[..., None], (1, 1, 1, nc))

    A = view_as_windows_torch(calib.reshape((1, 1, calib.shape[0], calib.shape[1], calib.shape[2])), (kx, ky, nc)).reshape(-1, kx, ky, nc).to(device)
    recon = torch.zeros_like(kspace, device=device)

    for ii in validP:
        S = A[:, P[ii, ...]]
        T = A[:, kx2, ky2, :]

        ShS = torch.matmul(S.conj().t(), S)
        ShT = torch.matmul(S.conj().t(), T)
        lamda0 = lamda * torch.norm(ShS) / ShS.shape[0]
        W = torch.linalg.solve(ShS + lamda0 * torch.eye(ShS.shape[0], device=device), ShT).t()

        idx = unravel_index_torch(torch.nonzero(iidx == ii).squeeze(), Psh[:2])
        x, y = idx[0]+kx2, idx[1]+ky2

        x = atleast_1d_torch(x.squeeze())
        y = atleast_1d_torch(y.squeeze())

        my_x1 = x-kx2
        my_x2 = x+kx2+adjx
        my_y1 = y-ky2
        my_y2 = y+ky2+adjy
        
        max_x_size = (my_x2 - my_x1).max().item()
        max_y_size = (my_y2 - my_y1).max().item()

        my_B = torch.zeros(max_x_size, max_y_size, kspace.shape[2], my_x1.shape[0])

        x_indices = torch.arange(max_x_size, device=device).view(-1, 1, 1).expand(max_x_size, max_y_size, my_x1.shape[0])
        y_indices = torch.arange(max_y_size, device=device).view(1, -1, 1).expand(max_x_size, max_y_size, my_y1.shape[0])

        x_indices = x_indices + my_x1.view(1, 1, -1)
        y_indices = y_indices + my_y1.view(1, 1, -1)

        valid_x_mask = (x_indices < my_x2.view(1, 1, -1)).float()
        valid_y_mask = (y_indices < my_y2.view(1, 1, -1)).float()
        valid_mask = valid_x_mask * valid_y_mask

        x_indices = x_indices * valid_x_mask.long()
        y_indices = y_indices * valid_y_mask.long()

        my_B = kspace[x_indices, y_indices, :].permute(0, 1, 3, 2)

        my_B = my_B[P[ii, ...], :]

        recon[x, y, :] = torch.einsum('ij,jkl->ikl', W, my_B[:, None]).squeeze(1).permute(1, 0)

    recon = torch.moveaxis((recon + kspace)[kx2:-kx2, ky2:-ky2, :], -1, coil_axis)
    return recon