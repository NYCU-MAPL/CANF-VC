'''
© 2019, JamesChan
forked from https://github.com/One-sixth/ms_ssim_pytorch/ssim.py
'''
import torch
import torch.nn.functional as F


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    """create 1D gauss kernel  
    Args:
        window_size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
        channel (int): input channel
    """
    half_window = window_size // 2
    coords = torch.arange(-half_window, half_window+1).float()

    g = (-(coords ** 2) / (2 * sigma ** 2)).exp_()
    g.div_(g.sum())

    return g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)


@torch.jit.script
def gaussian_blur(x, window, use_padding: bool):
    """Blur input with 1-D gauss kernel  
    Args:
        x (tensor): batch of tensors to be blured
        window (tensor): 1-D gauss kernel
        use_padding (bool): padding image before conv
    """
    C = x.size(1)
    padding = 0 if not use_padding else window.size(3) // 2
    out = F.conv2d(x, window, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window.transpose(2, 3),
                   stride=1, padding=(padding, 0), groups=C)
    return out


@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    """Calculate ssim for X and Y  
    Args:
        X (tensor):Y (tensor): a batch of images, (N, C, H, W)
        window (tensor): 1-D gauss kernel
        data_range (float): value range of input images. (usually 1.0 or 255)
        use_padding (bool, optional): padding image before conv. Defaults to False.
    """
    K1, K2 = 0.01, 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = gaussian_blur(X, window, use_padding)
    mu2 = gaussian_blur(Y, window, use_padding)
    sigma1_sq = gaussian_blur(X * X, window, use_padding)
    sigma2_sq = gaussian_blur(Y * Y, window, use_padding)
    sigma12 = gaussian_blur(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim, cs


@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False):
    """Calculate ms_ssim for X and Y  
    Args:
        X (tensor):Y (tensor): a batch of images, (N, C, H, W)
        window (tensor): 1-D gauss kernel
        data_range (float): value range of input images. (usually 1.0 or 255)
        weights (tensor): weights for different levels
        use_padding (bool, optional): padding image before conv. Defaults to False.
    """
    css, ssims = [], []
    for _ in range(weights.size(0)):
        ssim, cs = ssim(X, Y, window, data_range, use_padding)
        css.append(cs)
        ssims.append(ssim)
        padding = (X.size(-2) % 2, X.size(-1) % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ms_css = torch.stack(css[:-1], dim=0) ** weights[:-1].unsqueeze(1)
    ms_ssim = torch.prod(ms_css * (ssims[-1] ** weights[-1]), dim=0)
    return ms_ssim


class SSIM(torch.jit.ScriptModule):
    """Structural Similarity index  
    Args:
        window_size (int, optional): the size of gauss kernel. Defaults to 11.
        window_sigma (float, optional): sigma of normal distribution. Defaults to 1.5.
        data_range (float, optional): value range of input images. (usually 1.0 or 255). Defaults to 255..
        channel (int, optional): input channels. Defaults to 3.
        use_padding (bool, optional): padding image before conv. Defaults to False.
        reduction (str, optional): reduction mode. Defaults to "none".
    """
    __constants__ = ['data_range', 'use_padding', 'reduction']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, reduction="none"):
        super().__init__()
        self.data_range = data_range
        self.use_padding = use_padding
        self.reduction = reduction

        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

    @torch.jit.script_method
    def forward(self, input, target):
        ret = ssim(input, target, self.window,
                   self.data_range, self.use_padding)[0]
        if self.reduction != 'none':
            ret = ret.mean() if self.reduction == 'mean' else ret.sum()
        return ret


class MS_SSIM(SSIM):
    """Multi-Scale Structural Similarity index  
    Args:
        window_size (int, optional): the size of gauss kernel. Defaults to 11.
        window_sigma (float, optional): sigma of normal distribution. Defaults to 1.5.
        data_range (float, optional): value range of input images. (usually 1.0 or 255). Defaults to 255..
        channel (int, optional): input channels. Defaults to 3.
        use_padding (bool, optional): padding image before conv. Defaults to False.
        reduction (str, optional): reduction mode. Defaults to "none".
        weights (list of float, optional): weights for different levels. Default to [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levels (int, optional): number of downsampling
    """
    __constants__ = ['data_range', 'use_padding', 'reduction']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, reduction="none",
                 weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], levels=None):
        super().__init__(window_size, window_sigma,
                         data_range, channel, use_padding, reduction)

        weights = torch.FloatTensor(weights)
        if levels is not None:
            weights = weights[:levels]
            weights /= weights.sum()
        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, input, target):
        ret = ms_ssim(input, target, self.window, self.data_range,
                      self.weights, self.use_padding)
        if self.reduction != 'none':
            ret = ret.mean() if self.reduction == 'mean' else ret.sum()
        return ret
