"""Functional extention

James Chan
"""
import numpy as np
import torch
import torch.nn.functional as F


def torchseed(seed=666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.jit.script
def tile_dim(input, dim: int, n_tile: int):
    """tile input"""
    if dim < 0:
        dim = input.dim() + dim
    expanse = list(input.size())
    expanse.insert(dim, n_tile)
    return input.unsqueeze(dim).expand(expanse).transpose(dim, dim+1).flatten(dim, dim+1)


@torch.jit.script
def tile(input, size):
    # type: (Tensor, List[int]) -> Tensor
    """tile"""
    assert input.dim() == len(size)
    isize = input.size()
    for dim in range(input.dim()):
        i, o = input.size(dim), size[dim]
        if i != o:
            if o % i:
                message = "The tiled size of the tensor ({}) must be a multiple of the existing size ({})" +\
                    " at non-singleton dimension {}.  Target sizes: {}.  Tensor sizes: {}"
                raise RuntimeError(message.format(o, i, dim, size, isize))
            input = tile_dim(input, dim, n_tile=o//i)
    return input


def tile_as(input, other):
    """tile input as other.size()"""
    return tile(input, other.size())


def tile2d(input, n_tile):
    """tile input at last 2 dim (H, W)"""
    assert input.dim() >= 2 and len(n_tile) == 2
    return tile_dim(tile_dim(input, -2, n_tile[0]), -1, n_tile[1])


_shuffle_info = {}


def _get_shuffle_info(input, upscale_factor):
    pixel_dim = input.dim() - 2
    if isinstance(upscale_factor, int):
        upscale_factor = [upscale_factor] * pixel_dim
    else:
        assert len(upscale_factor) == pixel_dim
        if isinstance(upscale_factor, tuple):
            upscale_factor = list(upscale_factor)
    key = (input.size(), *upscale_factor)

    if key not in _shuffle_info:
        size = list(input.size())
        assert size[1] % np.prod(upscale_factor) == 0
        extented_shape = [size[0], -1] + upscale_factor + size[2:]

        permute_dim = [0, 1]
        for dim in range(2, 2+pixel_dim):
            permute_dim += [dim+pixel_dim, dim]

        reshape_size = [size[0], -1]
        for shape, scale in zip(size[2:], upscale_factor):
            reshape_size.append(shape*scale)

        _shuffle_info[key] = (extented_shape, permute_dim, reshape_size)

    return _shuffle_info[key]


def pixel_shuffle(input, upscale_factor):
    """pixel_shuffle"""
    extented_shape, permute_dim, reshape_size = _get_shuffle_info(
        input, upscale_factor)
    extented = input.reshape(*extented_shape)
    transposed = extented.permute(*permute_dim)
    shuffled = transposed.reshape(*reshape_size)
    return shuffled


def depth_to_space(input, block_size):
    """depth_to_space, alias of pixel_shuffle"""
    return pixel_shuffle(input, block_size)


_unshuffle_info = {}


def _get_unshuffle_info(input, block_size):
    pixel_dim = input.dim() - 2
    if isinstance(block_size, int):
        block_size = [block_size] * pixel_dim
    else:
        assert len(block_size) == pixel_dim
        if isinstance(block_size, tuple):
            block_size = list(block_size)
    key = (input.size(), *block_size)

    if key not in _unshuffle_info:
        size = list(input.size())
        reshape_size = [size[0], -1]
        for shape, scale in zip(size[2:], block_size):
            assert shape % scale == 0
            reshape_size += [shape//scale, scale]

        permute_dim = [0, 1]
        for dim in range(pixel_dim+1, 1, -1):
            permute_dim += [dim, dim+pixel_dim]

        merge_shape = [size[0], -1] + \
            reshape_size[slice(2, 2+pixel_dim+1, pixel_dim)]

        _unshuffle_info[key] = (reshape_size, permute_dim, merge_shape)

    return _unshuffle_info[key]


def space_to_depth(input, block_size):
    """space_to_depth"""
    reshape_size, permute_dim, merge_shape = _get_unshuffle_info(
        input, block_size)
    extented = input.reshape(*reshape_size)
    transposed = extented.permute(*permute_dim)
    merged = transposed.reshape(*merge_shape)
    return merged


def cat_k(input):
    """concat second dimesion to batch"""
    return input.flatten(0, 1)


def repair(input1, input2):
    """tile input1 input2.s times at batch dim and flatten input2"""
    return tile_dim(input1, 0, input2.size(1)), cat_k(input2)


def split_k(input, size: int, dim: int = 0):
    """reshape input to original batch size"""
    if dim < 0:
        dim = input.dim() + dim
    split_size = list(input.size())
    split_size[dim] = size
    split_size.insert(dim+1, -1)
    return input.view(split_size)


def get_topkhot(input, k: int, dim: int = -1):
    """return one_hot map of input and index"""
    index = input.topk(k, dim, sorted=False)[1]
    return torch.zeros_like(input).scatter_(dim, index, 1.0), index


def get_onehot(input, dim: int = -1):
    """return one_hot map of input and index"""
    index = input.max(dim, keepdim=True)[1]
    return torch.zeros_like(input).scatter_(dim, index, 1.0), index


def onehot_embedding(input, num_embeddings):
    assert input.dim() == 2
    return input.new_zeros(input.size(0), num_embeddings).scatter_(1, input, 1)


@torch.jit.script
def getWH(input):
    """Get [W, H] tensor from input"""
    H, W = input.size()[-2:]
    return torch.tensor([[W, H]], dtype=torch.float32, device=input.device)
    # return input.new_tensor([W, H])


@torch.jit.script
def center_of(input):
    """return [(W-1)/2, (H-1)/2] tensor of input img"""
    if input.dim() == 4:
        H, W = input.size()[-2:]
        shape = [[W, H]]
    else:
        D, H, W = input.size()[-3:]
        shape = [[W, H, D]]
    return torch.tensor(shape, dtype=torch.float32, device=input.device).sub_(1).div_(2)
    # return input.new_tensor(input.size()[:1:-1]).sub_(1).div_(2)


class GeneratorCache(object):
    def __init__(self, func):
        self.func = func
        self._cache = {}

    def __str__(self):
        return self.func.__name__

    def __call__(self, size, *args, **kwargs):
        assert isinstance(size, (list, tuple))
        if isinstance(size, list):
            size = tuple(size)
        key = (size, args, tuple(kwargs.items()))
        if key not in self._cache:
            self._cache[key] = self.func(size, *args, **kwargs)
        return self._cache[key]


@GeneratorCache
def meshgrid(size, align_corners=True, device='cpu'):
    # type: (List[int]) -> Tensor
    """return meshgrid (B, H, W, 2) of input size(width first, range (-1, -1)~(1, 1))"""
    coords = torch.meshgrid(*[torch.linspace(-1, 1, s)*(1 if align_corners else (s-1)/s)
                              if s > 1 else torch.zeros(1) for s in size[2:]])
    return torch.stack(coords[::-1], dim=-1).repeat(*([size[0]] + [1]*(len(size)-1))).to(device)


def meshgrid_of(input, align_corners=True):
    """return meshgrid (B, H, W, 2) of input size(width first, range (-1, -1)~(1, 1))"""
    return meshgrid(input.size(), align_corners, device=input.device)


def catCoord(input):
    """concatenates meshgrid (B, 2, H, W) of input size(width first, range (-1, -1)~(1, 1)) behind input"""
    return torch.cat([input, (meshgrid_of(input)*center_of(input)).permute(0, 3, 1, 2)], dim=1)


@GeneratorCache
def indices(size, device='cpu'):
    # type: (List[int]) -> Tensor
    """return flatten indices (B, H*W, 2) of input size(width first, range (0, 0)~(W, H))"""
    idxs = torch.meshgrid(*[torch.arange(s) for s in size[2:]])
    return torch.stack(idxs[::-1], dim=-1).repeat(*([size[0]] + [1]*(len(size)-1))).to(device)


def indices_of(input):
    """return flatten indices (B, H*W, 2) of input size(width first, range (0, 0)~(W, H))"""
    return indices(input.size(), device=input.device)


def spatial_average(input, keepdim=False):
    return input.mean([-2, -1], keepdim=keepdim)


def interpolate_as(input, other, mode='nearest', align_corners=False):
    """interpolate_as"""
    if mode == 'nearest':
        align_corners = None
    return F.interpolate(input, size=other.size()[2:], mode=mode, align_corners=align_corners)


def pool_as(input, other, mode='avg'):
    pool = F.adaptive_avg_pool2d if mode == 'avg' else F.adaptive_max_pool2d
    return pool(input, output_size=other.size()[2:])


def abT(a, b=None):
    """return abT. if b is None, return aaT. supple batchly"""
    b = a if b is None else b
    return a @ b.transpose(-2, -1)


def det3x3(mat):
    """calculate the determinant of a 3x3 matrix, support batch."""
    M = mat.reshape(-1, 3, 3).permute(1, 2, 0)

    detM = M[0, 0] * (M[1, 1] * M[2, 2] - M[2, 1] * M[1, 2]) \
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) \
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    return detM.reshape(*mat.size()[:-2]) if mat.dim() > 2 else detM.contiguous()


def inv3x3(mat):
    """calculate the inverse of a 3x3 matrix, support batch."""
    M = mat.reshape(-1, 3, 3).permute(1, 2, 0)

    # Divide the matrix with it's maximum element
    max_vals = M.flatten(0, 1).max(0)[0]
    M = M / max_vals

    adjM = torch.empty_like(M)
    adjM[0, 0] = M[1, 1] * M[2, 2] - M[2, 1] * M[1, 2]
    adjM[0, 1] = M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]
    adjM[0, 2] = M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]
    adjM[1, 0] = M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]
    adjM[1, 1] = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]
    adjM[1, 2] = M[1, 0] * M[0, 2] - M[0, 0] * M[1, 2]
    adjM[2, 0] = M[1, 0] * M[2, 1] - M[2, 0] * M[1, 1]
    adjM[2, 1] = M[2, 0] * M[0, 1] - M[0, 0] * M[2, 1]
    adjM[2, 2] = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1]

    # Divide the maximum value once more
    invM = adjM / (det3x3(M.permute(2, 0, 1)) * max_vals)
    return invM.permute(2, 0, 1).reshape_as(mat)


def l1norm(input, dim=-1, eps=1e-12):
    """l1-norm along dim"""
    return F.normalize(input, p=1., dim=dim, eps=eps)


def has_nan(input):
    """check whether input contains nan."""
    return bool(torch.isnan(input).any())


def check_finity(name, input):
    infinity = torch.logical_not(torch.isfinite(input))
    if infinity.any():
        message = 'grad infinity waring :' + name + \
            '\tType: ' + ('nan' if torch.isnan(input).any() else 'inf')
        import warnings
        warnings.warn(message)


def check_full_rank(input):
    """check input matrix is full rank. if not, print rank."""
    if input.dim() < 3:
        input = input.unsqueeze(0)
    C = input.size(1)
    for mat in input:
        rank = torch.matrix_rank(mat)
        if rank < C:
            print(f'Rank: {rank}/{C}')
            return


def check_range(name, input, dim=None):
    """check range along dim, if dim is None, check whole Tensor"""
    if dim is not None:
        ret = ""
        for i, subt in enumerate(input.unbind(dim)):
            ret += check_range(name+'_%d' % i, subt)
            ret += "\n" if i < input.size(dim)-1 else ""
        return ret
    return f"{name} ({input.min().item():.4f} ~ {input.max().item():.4f})"


def parallel(module):
    """parallelize module"""
    return torch.nn.DataParallel(module).to("cuda:0")
