import torch.nn as nn
import numpy as np
from math import log10


def psnr(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)
    psnr = 20 * log10(data_range) - 10 * log10(mse.item())
    return psnr


class PSNR(nn.Module):
    """PSNR"""

    def __init__(self, reduction='none', data_range=1.):
        super(PSNR, self).__init__()
        self.reduction = reduction
        self.data_range = data_range

    def forward(self, input, target):
        mse = (input-target).pow(2).flatten(1).mean(-1)
        ret = 10 * (self.data_range ** 2 / (mse+1e-12)).log10()
        if self.reduction != 'none':
            ret = ret.mean() if self.reduction == 'mean' else ret.sum()
        return ret


def PSNR_np(imgs1, imgs2, data_range=255.):
    """PSNR for numpy image"""
    mse = np.mean(np.square(imgs1.astype(np.float) - imgs2.astype(np.float)))
    psnr = 20 * log10(data_range) - 10 * log10(mse.item())
    return psnr


def mse2psnr(mse, data_range=1.):
    """PSNR for numpy mse"""
    return 20 * log10(data_range) - 10 * log10(mse)
