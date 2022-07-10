"""Sampler library for Spatial Transform

James Chan
"""
import torch
import torch.nn.functional as F
from torch import nn

from .functional import center_of, getWH, inv3x3, meshgrid
from .warplayer import warp as warplayer

try:
    from .resample2d_package.resample2d import warp as warp_cuda
except:
    warp_cuda = None
try:
    from .resample3d_package.resample3d import warp3d as warp3d_cuda
except:
    warp3d_cuda = None


def cat_grid_z(grid, fill_value: int = 1):
    """concat z axis of grid at last dim , return shape (B, H, W, 3)"""
    return torch.cat([grid, torch.full_like(grid[..., 0:1], fill_value)], dim=-1)


class LinearizedMutilSample():
    num_grid = 8
    noise_strength = 0.5
    need_push_away = True
    fixed_bias = False

    @classmethod
    def hyperparameters(cls):
        return {'num_grid': cls.num_grid, 'noise_strength': cls.noise_strength,
                'need_push_away': cls.need_push_away, 'fixed_bias': cls.fixed_bias}

    @classmethod
    def set_hyperparameters(cls, **kwargs):
        selfparams = cls.hyperparameters()
        for key, value in kwargs.items():
            if selfparams[key] != value:
                setattr(cls, key, value)
                print('Set Linearized Mutil Sample hyperparam:`%s` to %s' %
                      (key, value))

    @classmethod
    def create_auxiliary_grid(cls, grid, inputsize):
        grid = grid.unsqueeze(1).repeat(1, cls.num_grid, 1, 1, 1)

        WH = grid.new_tensor([[grid.size(-2), grid.size(-3)]])
        grid_noise = torch.randn_like(grid[:, 1:]) / WH * cls.noise_strength
        grid[:, 1:] += grid_noise

        if cls.need_push_away:
            input_h, input_w = inputsize[-2:]
            least_offset = grid.new_tensor([2.0 / input_w, 2.0 / input_h])
            noise = torch.randn_like(grid[:, 1:]) * least_offset
            grid[:, 1:] += noise

        return grid

    @classmethod
    def warp_input(cls, input, auxiliary_grid, padding_mode='zeros', align_corners=False):
        assert input.dim() == 4
        assert auxiliary_grid.dim() == 5

        B, num_grid, H, W = auxiliary_grid.size()[:4]
        inputs = input.unsqueeze(1).repeat(1, num_grid, 1, 1, 1).flatten(0, 1)
        grids = auxiliary_grid.flatten(0, 1).detach()
        warped_input = F.grid_sample(inputs, grids, 'bilinear',
                                     padding_mode, align_corners)
        return warped_input.reshape(B, num_grid, -1, H, W)

    @classmethod
    def linearized_fitting(cls, warped_input, auxiliary_grid):
        assert warped_input.size(1) > 1, 'num of grid should be larger than 1'
        assert warped_input.dim() == 5, 'shape should be: B x Grid x C x H x W'
        assert auxiliary_grid.dim() == 5, 'shape should be: B x Grid x H x W x XY'
        assert warped_input.size(1) == auxiliary_grid.size(1)

        center_image = warped_input[:, 0]
        other_image = warped_input[:, 1:]
        center_grid = auxiliary_grid[:, 0]
        other_grid = auxiliary_grid[:, 1:]

        delta_intensity = other_image - center_image.unsqueeze(1)
        delta_grid = other_grid - center_grid.unsqueeze(1)

        # concat z and reshape to [B, H, W, XY1, Grid-1]
        xT = cat_grid_z(delta_grid).permute(0, 2, 3, 4, 1)
        # calculate dI/dX, euqation(7) in paper
        xTx = xT @ xT.transpose(3, 4)  # [B, H, W, XY1, XY1]
        xTx_inv_xT = inv3x3(xTx) @ xT  # [B, H, W, XY1, Grid-1]

        # prevent manifestation from out-of-bound samples mentioned in section 6.1 of paper
        dW, dH = delta_grid.abs().chunk(2, dim=-1)
        delta_mask = ((dW <= 1.0) * (dH <= 1.0)).permute(0, 2, 3, 4, 1)
        xTx_inv_xT = xTx_inv_xT * delta_mask

        # [B, Grid-1, C, H, W] reshape to [B, H, W, Grid-1, C]
        delta_intensity = delta_intensity.permute(0, 3, 4, 1, 2)
        # gradient_intensity shape: [B, H, W, XY1, C]
        gradient_intensity = xTx_inv_xT @ delta_intensity

        # stop gradient shape: [B, H, W, C, XY1]
        gradient_intensity = gradient_intensity.detach().transpose(3, 4)

        # center_grid shape: [B, H, W, XY1]
        grid_xyz_stop = cat_grid_z(center_grid.detach(), int(cls.fixed_bias))
        gradient_grid = cat_grid_z(center_grid) - grid_xyz_stop

        # map to linearized, equation(2) in paper
        return center_image + (gradient_intensity @ gradient_grid.unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)

    @classmethod
    def apply(cls, input, grid, padding_mode='zeros', align_corners=False):
        assert input.size(0) == grid.size(0)
        auxiliary_grid = cls.create_auxiliary_grid(grid, input.size())
        warped_input = cls.warp_input(
            input, auxiliary_grid, padding_mode, align_corners)
        linearized_input = cls.linearized_fitting(warped_input, auxiliary_grid)
        return linearized_input


def linearized_grid_sample(input, grid, padding_mode='zeros', align_corners=False,
                           num_grid=8, noise_strength=.5, need_push_away=True, fixed_bias=False):
    """Linearized multi-sampling

    Args:
        input (tensor): (B, C, H, W)
        grid (tensor): (B, H, W, 2)
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        num_grid (int, optional): multisampling. Defaults to 8.
        noise_strength (float, optional): auxiliary noise. Defaults to 0.5.
        need_push_away (bool, optional): pushaway grid. Defaults to True.
        fixed_bias (bool, optional): Defaults to False.

    Returns:
        tensor: linearized sampled input

    Reference:
        paper: https://arxiv.org/abs/1901.07124
        github: https://github.com/vcg-uvic/linearized_multisampling_release
    """
    LinearizedMutilSample.set_hyperparameters(
        num_grid=num_grid, noise_strength=noise_strength, need_push_away=need_push_away, fixed_bias=fixed_bias)
    return LinearizedMutilSample.apply(input, grid, padding_mode, align_corners)


@torch.jit.script
def u(s, a: float = -0.75):
    s2, s3 = s**2, s**3
    l1 = (a+2)*s3 - (a+3)*s2 + 1
    l2 = a*s3 - (5*a)*s2 + (8*a)*s - 4*a
    return l1.where(s <= 1, l2)


@torch.jit.script
def bicubic_grid_sample(input, grid, padding_mode: str = 'zeros', align_corners: bool = False):
    """bicubic_grid_sample"""
    kernel_size = 4
    if not align_corners:
        grid = grid * getWH(input) / getWH(input).sub_(1)
    center = center_of(input)
    abs_loc = ((grid + 1) * center).unsqueeze(-1)

    locs = abs_loc.floor() + torch.tensor([-1, 0, 1, 2], device=grid.device)

    loc_w, loc_h = locs.detach().flatten(0, 2).unbind(dim=-2)
    loc_w = loc_w.reshape(-1, 1, kernel_size).expand(-1, kernel_size, -1)
    loc_h = loc_h.reshape(-1, kernel_size, 1).expand(-1, -1, kernel_size)
    loc_grid = torch.stack([loc_w, loc_h], dim=-1)
    loc_grid = loc_grid.view(grid.size(0), -1, 1, 2)/center - 1

    selected = F.grid_sample(input, loc_grid.detach(), mode='nearest',
                             padding_mode=padding_mode, align_corners=True)
    patch = selected.view(input.size()[:2]+grid.size()[1:3]+(kernel_size,)*2)

    mat_r, mat_l = u(torch.abs(abs_loc - locs.detach())).unbind(dim=-2)
    output = torch.einsum('bhwl,bchwlr,bhwr->bchw', mat_l, patch, mat_r)
    return output


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    original function prototype:
    torch.nn.functional.grid_sample(
        input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    copy from pytorch 1.3.0 source code
    add linearized_grid_sample and bicubic_grid_sample
    """
    if mode == 'linearized':
        assert input.dim() == grid.dim() == 4
        return LinearizedMutilSample.apply(input, grid, padding_mode, align_corners)
    if mode == 'bicubic':
        assert input.dim() == grid.dim() == 4
        return bicubic_grid_sample(input, grid, padding_mode, align_corners)
    else:
        return F.grid_sample(input, grid, mode, padding_mode, align_corners)


def warp(input, flow, sample_mode='bilinear', padding_mode='border', align_corners=True):
    """warp input with flow"""
    dim = input.dim()
    assert dim == flow.dim() and dim in [4, 5], input.size()
    if (warp_cuda is not None and dim == 4) or (warp3d_cuda is not None and dim == 5):
        if input.is_cuda and input.size()[2:] == flow.size()[2:]:
            assert padding_mode == 'border' and align_corners == True
            return warp_cuda(input, flow, sample_mode) if dim == 4 else warp3d_cuda(input, flow, bilinear=sample_mode == 'bilinear')
    if flow.size(-1) != dim-2:  # (B, 2, H, W) or (B, 3, D, H, W)
        flow = flow.permute(0, *list(range(2, dim)), 1)
        # (B, H, W, 2) or (B, D, H, W, 3)
    grid = meshgrid((flow.size(0), 1) + flow.size()
                    [1:-1], align_corners, device=flow.device) + flow/center_of(input)
    return grid_sample(input, grid, sample_mode, padding_mode, align_corners)


def warp2d(input, flow, sample_mode='bilinear', padding_mode='border', align_corners=True):
    """warp input with flow"""
    assert input.dim() == flow.dim() == 4
    if warp_cuda is not None and input.is_cuda and input.size()[2:] == flow.size()[2:]:
        assert padding_mode == 'border' and align_corners == True
        return warp_cuda(input, flow, sample_mode)
    if flow.size(-1) != 2:  # B, 2, H, W
        flow = flow.permute(0, 2, 3, 1)  # B, H, W, 2
    grid = meshgrid((flow.size(0), 1) + flow.size()
                    [1:-1], align_corners, device=flow.device) + flow/center_of(input)
    return grid_sample(input, grid, sample_mode, padding_mode, align_corners)


def warp3d(input, flow, sample_mode='bilinear', padding_mode='border', align_corners=True):
    assert input.dim() == flow.dim() == 5
    if warp3d_cuda is not None and input.is_cuda and input.size()[2:] == flow.size()[2:]:
        assert padding_mode == 'border' and align_corners == True
        return warp3d_cuda(input, flow, bilinear=sample_mode == 'bilinear')
    if flow.size(-1) != 3:  # B, 3, D, H, W
        flow = flow.permute(0, 2, 3, 4, 1)  # B, D, H, W, 3
    grid = meshgrid((flow.size(0), 1) + flow.size()
                    [1:-1], align_corners, device=flow.device) + flow/center_of(input)
    return grid_sample(input, grid, sample_mode, padding_mode, align_corners)


def warp3d_2(input, flow, sample_mode='bilinear', padding_mode='border', align_corners=False):
    assert input.dim() == flow.dim() == 5
    if flow.size(-1) != 3:  # B, 3, D, H, W
        flow = flow.permute(0, 2, 3, 4, 1)  # B, D, H, W, 3

    assert flow.size(1) == 1
    shifted = warp(input.flatten(1, 2), flow[..., :2].squeeze(1),
                   sample_mode, padding_mode, align_corners).view_as(input)

    B, C, D, H, W = input.size()
    scale = flow[..., -1:].permute(0, 1, 4, 2, 3) + (D-1) / 2
    lb = scale.floor().clamp(0, D-1)
    ub = (lb + 1).clamp(0, D-1)
    alpha = scale - scale.floor()

    lv = shifted.gather(2, lb.long().expand(B, C, -1, H, W))
    uv = shifted.gather(2, ub.long().expand(B, C, -1, H, W))

    val = (1-alpha) * lv + alpha * uv
    return val


class Resampler(nn.Module):
    """Resample image with flow

    Args:
        sample_mode (str): sample mode for gridsample
            'bilinear' | 'linearized' | 'nearest'. Default: 'bilinear'
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'border'

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Flow: :math:`(N, 2, H', W')` or `(N, H', W', 2)`
        - Output: :math:`(N, C, H', W')` (same shape as flow)

    Returns:
        Resampled input
    """

    def __init__(self, sample_mode='bilinear', padding_mode='border', align_corners=True):
        super(Resampler, self).__init__()
        self.sample_mode = sample_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def extra_repr(self):
        return "sample_mode={sample_mode}, padding_mode={padding_mode}".format(**self.__dict__)

    def forward(self, input, flow):
        return warplayer(input, flow, self.sample_mode, self.padding_mode, self.align_corners)


def index_select(input, index, index_mode='coord', select_mode='bilinear'):
    """index select

    Args:
        input: shape(B, C, H, W) or (B, C, D, H, W)
        index: shape(B, K, 2) or (B, K, 3)
        index_mode (str): 'coord' | 'position'. Default: 'coord'
        select_mode (str): sample mode for gridsample
            'bilinear' | 'linearized' | 'nearest'. Default: 'bilinear'

    Returns:
        selected items: shape(B, K, C)
    """
    if index_mode == 'coord':
        grid = index
    elif index_mode == 'position':
        grid = index/center_of(input) - 1

    if grid.min() < -1 or grid.max() > 1:
        raise IndexError("index out of range")

    view_shape = grid.size()[:2] + (1,)*(input.dim()-3) + grid.size()[-1:]
    selected = grid_sample(input, grid.view(view_shape), mode=select_mode,
                           padding_mode='zeros', align_corners=True)
    return selected.view(selected.size()[:3]).transpose(-1, -2)


class IndexSelecter(nn.Module):
    """index select

    Args:
        input: shape(B, C, H, W) or (B, C, D, H, W)
        index: shape(B, K, 2) or (B, K, 3)
        index_mode (str): 'coord' | 'position'. Default: 'coord'
        select_mode (str): sample mode for gridsample
            'bilinear' | 'linearized' | 'nearest'. Default: 'bilinear'

    Returns:
        select items: shape(B, K, *)
    """

    def __init__(self, index_mode='coord', select_mode='bilinear'):
        super().__init__()
        self.index_mode = index_mode
        self.select_mode = select_mode

    def extra_repr(self):
        return 'index_mode={index_mode}, select_mode={select_mode}'.format(**self.__dict__)

    def forward(self, input, index):
        return index_select(input, index, self.index_mode, self.select_mode)


def affine_grid(theta, size, align_corners=True):
    # type: (Tensor, List[int]) -> Tensor
    grid = meshgrid(size, align_corners, device=theta.device)
    return cat_grid_z(grid).flatten(1, 2).bmm(theta.transpose(1, 2)).view_as(grid)


def homography_grid(matrix, size, align_corners=True):
    # type: (Tensor, List[int]) -> Tensor
    grid = cat_grid_z(meshgrid(size, align_corners,
                               device=matrix.device))  # B, H, W, 3
    homography = grid.flatten(1, 2).bmm(matrix.transpose(1, 2)).view_as(grid)
    grid, ZwarpHom = homography.split([2, 1], dim=-1)
    return grid / ZwarpHom.add(1e-8)


def transform_grid(matrix, size, align_corners=True):
    if matrix.size()[1:] in [(2, 3), (3, 4)]:
        return F.affine_grid(matrix, size, align_corners)
    else:
        return homography_grid(matrix, size, align_corners)


def affine(input, theta, size=None, sample_mode='bilinear', padding_mode='border', align_corners=False):
    # type: (Tensor, Tensor, Optional[List[int]], str, str, bool) -> Tensor
    """SPT affine function

    Args:
        input: 4-D tensor (B, C, H, W) or 5-D tensor (B, C, D, H, W)
        theta: 3-D tensor (B, 2, 3) or (B, 3, 4)
        size (Size): output size. Default: input.size()
    """
    assert input.dim() in [4, 5] and theta.dim() == 3
    assert input.size(0) == theta.size(
        0), 'batch size of inputs do not match the batch size of theta'
    if size is None:
        size = input.size()[2:]
    size = (input.size(0), 1) + tuple(size)
    return grid_sample(input, transform_grid(theta, size, align_corners), sample_mode, padding_mode, align_corners)


def shift(input, motion, size=None, sample_mode='bilinear', padding_mode='border', align_corners=False):
    """SPT shift function

    Args:
        input: 4-D tensor (B, C, H, W) or 5-D tensor (B, C, D, H, W)
        motion (motion): motion (B, 2) or (B, 3)
    """
    B = motion.size(0)
    MD = input.dim() - 2

    defo = torch.eye(MD).to(input.device)
    txy = motion.view(B, MD) / center_of(input)
    theta = torch.cat([defo.expand(B, MD, MD), txy.view(B, MD, 1)], dim=2)
    return affine(input, theta, size, sample_mode, padding_mode, align_corners)


class SpatialTransformer(nn.Module):
    """`Spatial Transformer` in `Spatial Transformer Network`

    Args:
        mode (str): SPT mode 'affine' | 'shift'. Default: 'affine'
        sample_mode (str): sample mode for gridsample and affine_grid
            'bilinear' | 'linearized' | 'nearest'. Default: 'bilinear'
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'border'

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Theta: :math:`(N, 2, *)` where `*` means, parameter shape SPT func need
        - Size (Tuple[int, int], Optional): output size. Default: input.size()
        - Output: :math:`(N, C, H, W)` (same shape as `Size`)

    Returns:
        Transformed input
    """

    def __init__(self, mode='affine', sample_mode='bilinear', padding_mode='border', align_corners=False):
        super(SpatialTransformer, self).__init__()
        self.mode = mode
        self.sample_mode = sample_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        if mode == 'affine':
            self.SPT_fn = affine
        elif mode == 'shift':
            self.SPT_fn = shift

    def extra_repr(self):
        return "mode={mode}, sample_mode={sample_mode}, padding_mode={padding_mode}".format(**self.__dict__)

    def forward(self, input, theta, size=None):
        return self.SPT_fn(input, theta, size, self.sample_mode, self.padding_mode, self.align_corners)


def crop(input, crop_center, window, sample_mode='bilinear', padding_mode='border'):
    """SPT crop function

    Args:
        input: input img
        crop_center (position): crop center (0, 0)~(W-1, H-1)
        window (int or list_like): crop size
    """
    if isinstance(window, (tuple, list)):
        newh, neww = window
    elif isinstance(window, int):
        newh, neww = window, window

    B, _, H, W = input.size()

    defo = input.new_tensor([[(neww-1)/(W-1), 0], [0, (newh-1)/(H-1)]])
    txy = crop_center/center_of(input) - 1
    theta = torch.cat([defo.expand(B, 2, 2), txy.view(B, 2, 1)], dim=2)
    return affine(input, theta, (newh, neww), sample_mode, padding_mode, align_corners=True)


def random_crop(input, window, sample_mode='bilinear', padding_mode='margin'):
    """SPT random crop function

    Args:
        input: input img
        window (int or list_like): crop size
    """
    crop_center = torch.empty(input.size(0), 2).to(input.device)
    H, W = input.size()[-2:]
    if padding_mode == 'margin':
        crop_center[:, 0].uniform_(window//2, W-1-window//2)
        crop_center[:, 1].uniform_(window//2, H-1-window//2)
        padding_mode = 'zeros'
    else:
        crop_center[:, 0].uniform_(0, W-1)
        crop_center[:, 1].uniform_(0, H-1)
    return crop(input, crop_center, window, sample_mode, padding_mode)


def crop2(X, crop_center, window):
    if isinstance(window, (tuple, list)):
        newh, neww = window
    elif isinstance(window, int):
        newh, neww = window, window

    B = X.size(0)
    H, W = X.size()[-2:]
    newh, neww = min(newh, H), min(neww, W)

    tmp = X.new_zeros(B, X.size(1), newh, neww)
    halfh, halfw = newh//2, neww//2
    even = 1 if neww % 2 else 0
    for bid, img in enumerate(X):
        cw, ch = crop_center[bid].view(2).long()+1
        pt, pl = max(0-(ch-halfh), 0), max(0-(cw-halfw), 0)
        pd, pr = max((ch+halfh+even)-H, 0), max((cw+halfw+even)-W, 0)
        cw, ch = cw+pl, ch+pt
        pad_img = F.pad(img, (pl, pr, pt, pd), value=0)
        tmp[bid] = pad_img[:, ch-halfh:ch+halfh+even, cw-halfw:cw+halfw+even]
    return tmp
