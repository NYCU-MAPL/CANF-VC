import torch
import torch.nn as nn
from ssim import MS_SSIM, SSIM


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


def yuv420_loss(input, target, method, mode):
    Y_loss = method(input[:, :4], target[:, :4])
    U_loss = method(input[:, 4:5], target[:, 4:5])
    V_loss = method(input[:, 5:6], target[:, 5:6])
    total_loss = Y_loss*(6 if mode == '611' else 4) + U_loss + V_loss
    return total_loss/(8 if mode == '611' else 6), (Y_loss, U_loss, V_loss)


class YUV420Loss(nn.Module):
    def __init__(self, method=nn.functional.mse_loss, weight_mode="611") -> None:
        super().__init__()
        self.method = method
        self.mode = weight_mode

    def forward(self, input, target):
        return yuv420_loss(input, target, self.method, self.mode)
